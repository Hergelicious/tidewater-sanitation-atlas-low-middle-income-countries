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
rows = []
# 2. displacement restricted to land within the coastal zone
exec(open("/home/claude/atlas_dhs/health_model.py").read().split('print("children"')[0])
d = d[d.toilet != "other"].copy()
from metrics import bathtub, coast_masks, ocean_seeds, load, grids, cities, esl, grid_transform, geometry_mask
import metrics as M
link = open("/home/claude/atlas_dhs/link_clusters.py").read()
exec(link[link.index("def masks"):link.index("rows = []")])
g = pd.concat([gpd.read_file(f).assign(survey=f.split("/")[-2]) for f in sorted(glob.glob(f"{P}/ge/*/*.shp"))])
from shapely.geometry import Point
from shapely.prepared import prep
ISO = {"NG": "NGA", "GH": "GHA", "CI": "CIV", "MZ": "MOZ", "TZ": "TZA", "BD": "BGD", "PH": "PHL", "HT": "HTI"}
ADM = {k: gpd.read_file(f"/home/claude/data/geob/{v}_ADM2.geojson").to_crs(4326) for k, v in ISO.items()}
SIDX = {k: v.sindex for k, v in ADM.items()}
def adm_poly(cc, lat, lon):
    p = Point(lon, lat); a = ADM[cc]
    for i in SIDX[cc].query(p):
        if a.geometry.iloc[i].contains(p):
            return prep(a.geometry.iloc[i])
    return None
n_noadm = 0
cw2 = pd.read_csv(f"{P}/cluster_wet.csv"); cw2 = cw2[cw2.in_zone].merge(g[["survey", "DHSCLUST", "LATNUM", "LONGNUM"]], on=["survey", "DHSCLUST"])
NS = 50; sim = {}
for uid, sub in cw2.groupby("ID_UC_G0"):
    pop, zone, wet, gw, fl, t = masks(uid)
    h, w_ = pop.shape
    yy = t.f + t.e * (np.arange(h) + 0.5); xx = t.c + t.a * (np.arange(w_) + 0.5)
    for r in sub.itertuples():
        poly = adm_poly(r.survey[:2], r.LATNUM, r.LONGNUM)
        if poly is None:
            n_noadm += 1
        vals = []
        tries = 0
        while len(vals) < NS and tries < 2000:
            tries += 1
            ang, dist = rng.uniform(0, 2 * np.pi), rng.uniform(0, 2.0)
            a = r.LATNUM + dist * np.sin(ang) / 111.32; b = r.LONGNUM + dist * np.cos(ang) / (111.32 * np.cos(np.radians(r.LATNUM)))
            ri, ci = int((a - t.f) / t.e), int((b - t.c) / t.a)
            if not (0 <= ri < h and 0 <= ci < w_) or not zone[ri, ci]:
                continue                                   # reject points in the sea or outside the coastal zone
            if poly is not None and not poly.contains(Point(b, a)):
                continue                                   # reject points outside the second-level administrative unit
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
rows.append(dict(measure="Displacement simulation restricted to land in the coastal zone and the same second-level administrative unit (geoBoundaries ADM2; interaction, pp)", level="mean and 2.5-97.5 percentile of draws",
                 reference="", diff_pp=np.nanmean(draws), ci_lo=np.nanpercentile(draws, 2.5), ci_hi=np.nanpercentile(draws, 97.5)))
print("clusters without ADM2 match:", n_noadm)
old = pd.read_csv("/home/claude/atlas/supplementary/Table_S22_dhs_refinements.csv")
new = pd.DataFrame(rows).round(2)
pd.concat([old, new], ignore_index=True).to_csv("/home/claude/atlas/supplementary/Table_S22_dhs_refinements.csv", index=False)
print(new.to_string(index=False))
