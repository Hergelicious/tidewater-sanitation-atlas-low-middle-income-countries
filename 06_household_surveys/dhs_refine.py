"""Refinements requested by the v0.5 review (aggregated outputs only).
1. Poverty gradient with an asset index that excludes toilet type and drinking water (first principal component of
   household assets and housing materials, urban households, per survey), and with household head education.
2. GPS displacement simulation restricted to land within the coastal zone.
Writes supplementary/Table_S22_dhs_refinements.csv"""
import sys, glob, numpy as np, pandas as pd, geopandas as gpd
import statsmodels.formula.api as smf
sys.path.insert(0, "/home/claude/atlas")
P = "/home/claude/dhs_private"
rng = np.random.default_rng(7)
V = ["hv001", "hv002", "hv005", "hv009", "hv025", "hv205", "hv270a", "hv206", "hv207", "hv208", "hv209", "hv210", "hv211",
     "hv212", "hv221", "hv243a", "hv243e", "hv213", "hv214", "hv215", "hv226", "hv216", "hv106_01"]
hr = []
for f in sorted(glob.glob(f"{P}/hr/*/*.[dD][tT][aA]")):
    with pd.io.stata.StataReader(f) as r:
        cols = [c for c in V if c in r.variable_labels()]
    x = pd.read_stata(f, columns=cols, convert_categoricals=False)
    hr.append(x.assign(cc=f.split("/")[-2][:2]))
hr = pd.concat(hr, ignore_index=True)
u = hr[hr.hv025 == 1].copy()
yes = ["hv206", "hv207", "hv208", "hv209", "hv210", "hv211", "hv212", "hv221", "hv243a", "hv243e"]
for c in yes:
    u[c] = (u[c] == 1).astype(float)
u["floor"] = (u.hv213 >= 30).astype(float); u["wall"] = (u.hv214 >= 30).astype(float); u["roof"] = (u.hv215 >= 30).astype(float)
u["fuel"] = u.hv226.isin([1, 2, 3, 4]).astype(float)
u["crowd"] = (u.hv009 / u.hv216.where(u.hv216 > 0)).fillna(u.hv009)
X = yes + ["floor", "wall", "roof", "fuel"]
u["w"] = u.hv005 / 1e6 * u.hv009
idx = []
for cc, x in u.groupby("cc"):
    Z = x[X].values - x[X].values.mean(0)
    s = Z.std(0); Z = Z[:, s > 0] / s[s > 0]
    vals, vecs = np.linalg.eigh(np.cov(Z.T, aweights=x.w))
    pc = Z @ vecs[:, -1]
    if np.corrcoef(pc, x.hv208)[0, 1] < 0:
        pc = -pc
    o = np.argsort(pc); cw = np.cumsum(x.w.values[o]) / x.w.sum()
    q = np.empty(len(x)); q[o] = np.minimum(5, np.floor(cw * 5) + 1)
    idx.append(pd.Series(q, index=x.index))
u["q_assets"] = pd.concat(idx)
u["head_ed"] = u.hv106_01.where(u.hv106_01 <= 3)
cw = pd.read_csv(f"{P}/cluster_wet.csv"); cw["cc"] = cw.survey.str[:2]; cw = cw[cw.in_zone]
e = u.merge(cw, left_on=["cc", "hv001"], right_on=["cc", "DHSCLUST"])
e["clu"] = e.cc + "_" + e.hv001.astype(str); e["city"] = e.ID_UC_G0.astype(str)
rows = []
for var, ref, lab in (("hv270a", 5, "DHS urban wealth quintile (includes toilet)"),
                      ("q_assets", 5, "Asset index without toilet and water"),
                      ("head_ed", 3, "Household head education (none to higher)")):
    x = e.dropna(subset=[var])
    m = smf.wls(f"wet_2km ~ C({var}, Treatment({ref})) + C(city)", data=x, weights=x.w).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(x.city)[0]})
    for k in m.params.index:
        if k.startswith(f"C({var}"):
            lvl = k.split("[T.")[1].rstrip("]")
            rows.append(dict(measure=lab, level=lvl, reference=ref, diff_pp=100 * m.params[k],
                             ci_lo=100 * (m.params[k] - 1.96 * m.bse[k]), ci_hi=100 * (m.params[k] + 1.96 * m.bse[k])))
print("corr asset quintile vs DHS quintile:", round(e[["q_assets", "hv270a"]].corr(method="spearman").iloc[0, 1], 2))

# 2. displacement restricted to land within the coastal zone
exec(open("/home/claude/atlas_dhs/health_model.py").read().split('print("children"')[0])
d = d[d.toilet != "other"].copy()
from metrics import bathtub, coast_masks, ocean_seeds, load, grids, cities, esl, grid_transform, geometry_mask
import metrics as M
link = open("/home/claude/atlas_dhs/link_clusters.py").read()
exec(link[link.index("def masks"):link.index("rows = []")])
g = pd.concat([gpd.read_file(f).assign(survey=f.split("/")[-2]) for f in sorted(glob.glob(f"{P}/ge/*/*.shp"))])
cw2 = pd.read_csv(f"{P}/cluster_wet.csv"); cw2 = cw2[cw2.in_zone].merge(g[["survey", "DHSCLUST", "LATNUM", "LONGNUM"]], on=["survey", "DHSCLUST"])
NS = 50; sim = {}
for uid, sub in cw2.groupby("ID_UC_G0"):
    pop, zone, wet, gw, fl, t = masks(uid)
    h, w_ = pop.shape
    yy = t.f + t.e * (np.arange(h) + 0.5); xx = t.c + t.a * (np.arange(w_) + 0.5)
    for r in sub.itertuples():
        vals = []
        tries = 0
        while len(vals) < NS and tries < 2000:
            tries += 1
            ang, dist = rng.uniform(0, 2 * np.pi), rng.uniform(0, 2.0)
            a = r.LATNUM + dist * np.sin(ang) / 111.32; b = r.LONGNUM + dist * np.cos(ang) / (111.32 * np.cos(np.radians(r.LATNUM)))
            ri, ci = int((a - t.f) / t.e), int((b - t.c) / t.a)
            if not (0 <= ri < h and 0 <= ci < w_) or not zone[ri, ci]:
                continue                                   # reject points in the sea or outside the coastal zone
            dy = (yy - a)[:, None] * 111.32; dx = (xx - b)[None, :] * 111.32 * np.cos(np.radians(a))
            m_ = (np.sqrt(dy ** 2 + dx ** 2) <= 0.5) & zone
            wp = pop[m_].sum()
            vals.append(pop[m_ & wet].sum() / wp if wp > 0 else np.nan)
        vals += [np.nan] * (NS - len(vals))
        sim[(r.survey[:2], r.DHSCLUST)] = np.array(vals)
base = "C(age) + C(b4) + C(v106) + C(v190a) + C(water) + C(city) + C(month)"
draws = []
for s in range(NS):
    d["E"] = [sim.get((c, v), np.full(NS, np.nan))[s] for c, v in zip(d.cc, d.v001)]
    dd = d.dropna(subset=["E"])
    m = smf.wls(f"diar ~ E * soil + C(toilet) + {base}", data=dd, weights=dd.w).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(dd.clu)[0]})
    draws.append(100 * m.params["E:soil"])
draws = np.array(draws)
rows.append(dict(measure="Displacement simulation restricted to land in the coastal zone (interaction, pp)", level="mean and 2.5-97.5 percentile of draws",
                 reference="", diff_pp=np.nanmean(draws), ci_lo=np.nanpercentile(draws, 2.5), ci_hi=np.nanpercentile(draws, 97.5)))
t = pd.DataFrame(rows).round(2)
pd.set_option("display.width", 220, "display.max_colwidth", 90)
print(t.to_string(index=False))
t.to_csv("/home/claude/atlas/supplementary/Table_S22_dhs_refinements.csv", index=False)
