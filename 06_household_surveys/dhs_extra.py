"""DHS sensitivity analyses requested by the review (aggregated outputs only).
1. Health model: standard errors clustered by city; minimum detectable interaction (80% power, two-sided 5%).
2. GPS displacement: 50 simulated true locations per cluster (uniform direction, distance 0 to 2 km), wet-ground share within
   0.5 km of each, health model refitted per draw.
3. Poverty gradient (poorest vs richest quintile, city fixed effects) by country.
Writes supplementary/Table_S17_dhs_sensitivity.csv"""
import sys, glob, numpy as np, pandas as pd, geopandas as gpd
import statsmodels.formula.api as smf
sys.path.insert(0, "/home/claude/atlas"); sys.path.insert(0, "/home/claude/atlas_dhs")
P = "/home/claude/dhs_private"
rng = np.random.default_rng(20261106)
src = open("/home/claude/atlas_dhs/health_model.py").read()
exec(src.split('print("children"')[0])            # builds d (analysis sample) with the same definitions
d = d[d.toilet != "other"].copy()
base = "C(age) + C(b4) + C(v106) + C(v190a) + C(water) + C(city) + C(month)"
rows = []


def est(dd, expo, groups):
    dd = dd.dropna(subset=[expo]).copy(); dd["E"] = dd[expo]
    m = smf.wls(f"diar ~ E * soil + C(toilet) + {base}", data=dd, weights=dd.w).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(dd[groups])[0]})
    return m.params["E:soil"], m.bse["E:soil"], m.params["E"], m.params["E"] + m.params["E:soil"]


for g in ("clu", "city"):
    b, se, e0, e1 = est(d, "wet_2km", g)
    rows.append(dict(analysis=f"Primary model, SE clustered by {'survey cluster' if g == 'clu' else 'city'}", interaction_pp=100 * b,
                     ci_lo=100 * (b - 1.96 * se), ci_hi=100 * (b + 1.96 * se), mde_80_pp=100 * 2.8 * se))
w_sd = d.groupby("city").wet_2km.transform(lambda x: x - x.mean()).std()
print("within-city SD of wet_2km:", round(w_sd, 3), "share of cities with any variation:",
      round((d.groupby("city").wet_2km.std() > 0.05).mean(), 2))

# 2. displacement simulation
import metrics as M
from metrics import bathtub, coast_masks, ocean_seeds, load, grids, cities, esl, grid_transform, geometry_mask
cw = pd.read_csv(f"{P}/cluster_wet.csv"); cw = cw[cw.in_zone]
g = pd.concat([gpd.read_file(f).assign(survey=f.split("/")[-2]) for f in sorted(glob.glob(f"{P}/ge/*/*.shp"))])
cw = cw.merge(g[["survey", "DHSCLUST", "LATNUM", "LONGNUM"]], on=["survey", "DHSCLUST"])
link = open("/home/claude/atlas_dhs/link_clusters.py").read()
exec(link[link.index("def masks"):link.index("rows = []")])
NS = 50
sim = {}
for uid, sub in cw.groupby("ID_UC_G0"):
    pop, zone, wet, gw, fl, t = masks(uid)
    h, w_ = pop.shape
    yy = t.f + t.e * (np.arange(h) + 0.5); xx = t.c + t.a * (np.arange(w_) + 0.5)
    for r in sub.itertuples():
        ang = rng.uniform(0, 2 * np.pi, NS); dist = rng.uniform(0, 2.0, NS)
        la = r.LATNUM + dist * np.sin(ang) / 111.32
        lo = r.LONGNUM + dist * np.cos(ang) / (111.32 * np.cos(np.radians(r.LATNUM)))
        vals = []
        for a, b in zip(la, lo):
            dy = (yy - a)[:, None] * 111.32; dx = (xx - b)[None, :] * 111.32 * np.cos(np.radians(a))
            m = (np.sqrt(dy ** 2 + dx ** 2) <= 0.5) & zone
            wp = pop[m].sum()
            vals.append(pop[m & wet].sum() / wp if wp > 0 else np.nan)
        sim[(r.survey[:2], r.DHSCLUST)] = np.array(vals)
draws = []
for s in range(NS):
    d["E_sim"] = [sim.get((c, v), np.full(NS, np.nan))[s] for c, v in zip(d.cc, d.v001)]
    b, se, e0, e1 = est(d, "E_sim", "clu")
    draws.append(b)
draws = np.array(draws) * 100
rows.append(dict(analysis="Displacement simulation (50 draws, 0.5 km around simulated true locations)",
                 interaction_pp=np.nanmean(draws), ci_lo=np.nanpercentile(draws, 2.5), ci_hi=np.nanpercentile(draws, 97.5), mde_80_pp=np.nan))

# 3. poverty gradient by country (households)
_keep = rows
hr_src = open("/home/claude/atlas_dhs/hr_analyses.py").read()
exec(hr_src.split("# 2. exposure gradient")[0].replace('print(t.to_string(index=False))', '').replace(
    't.to_csv("/home/claude/atlas/supplementary/Table_S6_dhs_metro_sanitation.csv", index=False)', ''))
rows = _keep
cw2 = pd.read_csv(f"{P}/cluster_wet.csv"); cw2["cc"] = cw2.survey.str[:2]; cw2 = cw2[cw2.in_zone]
e = u.merge(cw2, left_on=["cc", "hv001"], right_on=["cc", "DHSCLUST"])
e["clu"] = e.cc + "_" + e.hv001.astype(str); e["city"] = e.ID_UC_G0.astype(str); e["q"] = e.hv270a.fillna(e.hv270)
e["poor"] = (e.q <= 2).astype(float)
for cc, x in e.groupby("cc"):
    if x.clu.nunique() < 30:
        continue
    m = smf.wls("wet_2km ~ poor + C(city)", data=x, weights=x.w).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(x.clu)[0]})
    b, se = m.params["poor"], m.bse["poor"]
    rows.append(dict(analysis=f"Poverty gradient, poorest two vs other quintiles, {cc} ({x.clu.nunique()} clusters, {x.city.nunique()} cities)",
                     interaction_pp=100 * b, ci_lo=100 * (b - 1.96 * se), ci_hi=100 * (b + 1.96 * se), mde_80_pp=np.nan))
m = smf.wls("wet_2km ~ poor + C(city)", data=e, weights=e.w).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(e.city)[0]})
rows.append(dict(analysis="Poverty gradient, all countries, SE clustered by city", interaction_pp=100 * m.params["poor"],
                 ci_lo=100 * (m.params["poor"] - 1.96 * m.bse["poor"]), ci_hi=100 * (m.params["poor"] + 1.96 * m.bse["poor"]), mde_80_pp=np.nan))
out = pd.DataFrame(rows).round(2)
pd.set_option("display.width", 220, "display.max_colwidth", 110)
print(out.to_string(index=False))
out.to_csv("/home/claude/atlas/supplementary/Table_S17_dhs_sensitivity.csv", index=False)
