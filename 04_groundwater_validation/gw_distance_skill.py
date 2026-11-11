"""Tidal rule skill by distance from the coastline (Natural Earth 1:10 m), for all validation sets, mean-sea-level referenced.
Also open coast versus estuary (well farther than 40 km from the generalised 1:110 m coastline is treated as estuarine).
Writes supplementary/Table_S21_gw_skill_by_distance.csv"""
import numpy as np, pandas as pd, geopandas as gpd
from scipy.spatial import cKDTree
from datum_dtu import offset_at

w = pd.read_csv("/home/claude/data/gw/wells_all_msl.csv")[["lon", "lat", "dtw", "e_msl", "set", "iso3"]]
c = pd.read_csv("/home/claude/data/cgwl/cgwl_lmic_wells_validation.csv")
c = c[c.dtw.between(-2, 200) & c.e_rel.notna()]
c = pd.DataFrame({"lon": c.lon, "lat": c.lat, "dtw": c.dtw, "e_msl": c.e_rel - offset_at(c.lat.values, c.lon.values),
                  "set": "Low and middle income (CGWL)", "iso3": c.country})
w = pd.concat([w, c], ignore_index=True)


def xyz(la, lo):
    la, lo = np.radians(la), np.radians(lo)
    return np.c_[np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]


def vertices(path, step_km=0.2):
    g = gpd.read_file(path).to_crs(4326)
    pts = []
    for geom in g.geometry.explode(index_parts=False):
        x, y = np.asarray(geom.coords.xy)
        for i in range(len(x) - 1):
            n = max(1, int(np.hypot((x[i + 1] - x[i]) * 111 * np.cos(np.radians(y[i])), (y[i + 1] - y[i]) * 111) / step_km))
            t = np.linspace(0, 1, n, endpoint=False)
            pts.append(np.c_[y[i] + t * (y[i + 1] - y[i]), x[i] + t * (x[i + 1] - x[i])])
    return np.vstack(pts)


v10 = vertices("/home/claude/data/ne/ne_10m_coastline.shp")
v110 = vertices("/home/claude/data/ne/ne_110m_coastline.shp", 1.0)
w["dist_km"] = cKDTree(xyz(v10[:, 0], v10[:, 1])).query(xyz(w.lat.values, w.lon.values))[0] * 6371
w["dist110_km"] = cKDTree(xyz(v110[:, 0], v110[:, 1])).query(xyz(w.lat.values, w.lon.values))[0] * 6371
w["band"] = pd.cut(w.dist_km, [-0.01, 1, 3, 5, 10.5], labels=["0-1 km", "1-3 km", "3-5 km", "5-10 km"])
w["setting"] = np.where(w.dist110_km > 40, "estuarine", "open coast")
w["grp"] = np.where(w.set.str.startswith("Low"), "Low and middle income (IGRAC + CGWL)", w.set)


def sk(x):
    p, o = x.e_msl <= 3, x.dtw <= 3
    tp, fp, fn, tn = (p & o).sum(), (p & ~o).sum(), (~p & o).sum(), (~p & ~o).sum()
    return pd.Series(dict(wells=len(x), flagged_pct=100 * p.mean(), obs_shallow_pct=100 * o.mean(),
                          PPV=tp / max(tp + fp, 1), sensitivity=tp / max(tp + fn, 1), specificity=tn / max(tn + fp, 1),
                          PPV_5m=((p & (x.dtw <= 5)).sum()) / max(p.sum(), 1)))


a = w.groupby(["grp", "band"], observed=True).apply(sk).reset_index()
b = w.groupby(["grp", "setting"]).apply(sk).reset_index().rename(columns={"setting": "band"})
t = pd.concat([a, b]).round(2)
pd.set_option("display.width", 220)
print(t.to_string(index=False))
t.to_csv("supplementary/Table_S21_gw_skill_by_distance.csv", index=False)
w.to_csv("/home/claude/data/gw/wells_all_msl_dist.csv", index=False)
