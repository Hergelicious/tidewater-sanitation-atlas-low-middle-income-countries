"""Link DHS cluster GPS points (urban, within atlas urban centres) to wet ground.
For each cluster: population-weighted share of wet ground (and of groundwater contact, flood only) among
3 arc-second cells within 1, 2 and 5 km, restricted to cells in the 10 km coastal zone.
Output (private, cluster level, no household data): /home/claude/dhs_private/cluster_wet.csv"""
import sys, os
import numpy as np, pandas as pd, geopandas as gpd, glob
sys.path.insert(0, "/home/claude/atlas")
import metrics as M
from metrics import bathtub, coast_masks, ocean_seeds, load, grids, cities, esl, grid_transform, geometry_mask

P = "/home/claude/dhs_private"
g = pd.concat([gpd.read_file(f).assign(survey=f.split("/")[-2]) for f in sorted(glob.glob(f"{P}/ge/*/*.shp"))])
g = g[(g.LATNUM != 0) & (g.URBAN_RURA == "U")].set_crs(4326, allow_override=True)
c = cities.reset_index()[["ID_UC_G0", "geometry"]]
j = gpd.sjoin(g, c, predicate="within", how="inner")
print("urban clusters inside atlas urban centres:", len(j), j.survey.value_counts().to_dict(), flush=True)


def masks(uid):
    gg = grids[uid]; shape = (gg["h"], gg["w"])
    inside = ~geometry_mask([cities.loc[uid, "geometry"]], out_shape=shape, transform=grid_transform(gg))
    pop = np.nan_to_num(load(uid, "pop", shape)) * inside
    dtm = load(uid, M.DTMNAME, shape)
    sea = np.nan_to_num(load(uid, "sea", shape)) >= 0.5
    riv = np.nan_to_num(load(uid, "riv", shape))
    d = np.where(np.isfinite(dtm), dtm, 9999.0)
    e = esl.loc[uid]
    near, ocean = coast_masks(uid, gg, shape)
    seeds = ocean_seeds(sea, ocean)
    land, n10 = ~sea, near[10]
    rv = land & (riv > 0) & (riv < 100) & n10
    coast = bathtub(d, seeds, e.esl_base, n10) & land
    gw = land & n10 & (d <= e.hat_base + 3)
    wet = coast | gw | rv
    return pop, land & n10, wet, gw, wet & ~gw, grid_transform(gg)


rows = []
for uid, sub in j.groupby("ID_UC_G0"):
    pop, zone, wet, gw, fl, t = masks(uid)
    h, w = pop.shape
    yy = t.f + t.e * (np.arange(h) + 0.5)
    xx = t.c + t.a * (np.arange(w) + 0.5)
    for r in sub.itertuples():
        ci = int((r.LONGNUM - t.c) / t.a); ri = int((r.LATNUM - t.f) / t.e)
        in_zone = 0 <= ri < h and 0 <= ci < w and bool(zone[ri, ci])
        o = {"survey": r.survey, "DHSCLUST": int(r.DHSCLUST), "ID_UC_G0": uid, "in_zone": in_zone}
        dy = (yy - r.LATNUM)[:, None] * 111.32
        dx = (xx - r.LONGNUM)[None, :] * 111.32 * np.cos(np.radians(r.LATNUM))
        dist = np.sqrt(dy ** 2 + dx ** 2)
        for km in (1, 2, 5):
            m = (dist <= km) & zone
            wp = pop[m].sum()
            o[f"wet_{km}km"] = float(pop[m & wet].sum() / wp) if wp > 0 else np.nan
            o[f"gw_{km}km"] = float(pop[m & gw].sum() / wp) if wp > 0 else np.nan
            o[f"flood_{km}km"] = float(pop[m & fl].sum() / wp) if wp > 0 else np.nan
            o[f"zonepop_{km}km"] = float(wp)
        rows.append(o)
out = pd.DataFrame(rows)
out.to_csv(f"{P}/cluster_wet.csv", index=False)
print(out.groupby("survey").agg(n=("DHSCLUST", "size"), in_zone=("in_zone", "sum"), wet2=("wet_2km", "mean")).round(2))
