"""Layers requested by the v0.6 review (raw population per city; non-sewered shares applied in summarise_v07.py).
1. Datum robustness: land shifted by -1.2 to +1.2 m relative to mean sea level; wet ground (D) and surface flooding (B).
2. Restricted open-coast subset: within 5 km of the 1:10 m coastline and within 40 km of the generalised 1:110 m coastline.
3. Location-specific sea level for 2020: relative sea-level rise interpolated per city from the JRC 2050 and 2100 medians
   (quadratic through zero at the 1997 baseline centre), with the projected change in highest astronomical tide
   (tidal amplitude) kept separate from mean sea level.
Writes city_exposure_v07.csv"""
import numpy as np
import pandas as pd
import geopandas as gpd
from multiprocessing import Pool
from scipy.spatial import cKDTree
from rasterio.features import geometry_mask
import metrics as M
from pipeline import grid_transform

E = "/home/claude/data/esl/"


def xyz(la, lo):
    la, lo = np.radians(la), np.radians(lo)
    return np.c_[np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]


# city-level projections not in city_esl.csv (same 3-nearest-point rule)
c = gpd.read_file("/home/claude/atlas/cities.gpkg")
cent = c.to_crs(3857).geometry.centroid.to_crs(4326)
X = pd.DataFrame({"ID_UC_G0": c.ID_UC_G0})
for k, f in {"hat_85_2050": "max_Astronomical_Tide_RCP85-2050", "rslr_85_2100": "RSLR_RCP85-2100"}.items():
    d = pd.read_csv(E + f + ".csv").dropna(subset=["valueMedian"])
    _, i = cKDTree(xyz(d.latitude.values, d.longitude.values)).query(xyz(cent.y.values, cent.x.values), k=3)
    X[k] = d.valueMedian.values[i].mean(axis=1)
X = X.set_index("ID_UC_G0").join(M.esl[["hat_base", "rslr_85_2050", "esl_base", "esl_85_2050"]])
x50, x100, x20 = 2050 - 1997, 2100 - 1997, 2020 - 1997
b = (X.rslr_85_2100 - X.rslr_85_2050 * x100 / x50) / (x100 ** 2 - x50 * x100)
a = (X.rslr_85_2050 - b * x50 ** 2) / x50
X["r20"] = np.clip(a * x20 + b * x20 ** 2, 0, X.rslr_85_2050)
X["dhat50"] = X.hat_85_2050 - X.hat_base
X["dhat20"] = X.dhat50 * x20 / x50
X.to_csv("/home/claude/atlas/city_sl2020.csv")

V110 = None


def city(uid):
    global V110
    if V110 is None:
        import gw_vertices
        V110 = cKDTree(xyz(*gw_vertices.v110().T))
    g = M.grids[uid]
    shape = (g["h"], g["w"])
    inside = ~geometry_mask([M.cities.loc[uid, "geometry"]], out_shape=shape, transform=grid_transform(g))
    pop = np.nan_to_num(M.load(uid, "pop", shape)) * inside
    pop50 = np.nan_to_num(M.load(uid, "pop50", shape)) * inside
    dtm = M.load(uid, M.DTMNAME, shape)
    sea = np.nan_to_num(M.load(uid, "sea", shape)) >= 0.5
    riv = np.nan_to_num(M.load(uid, "riv", shape))
    d = np.where(np.isfinite(dtm), dtm, 9999.0)
    e, s = M.esl.loc[uid], X.loc[uid]
    near, ocean = M.coast_masks(uid, g, shape)
    seeds = M.ocean_seeds(sea, ocean)
    land, n10 = ~sea, near[10]
    rv = land & (riv > 0) & (riv < 100) & n10

    def layers(dd, esl, hat):
        C = M.bathtub(dd, seeds, esl, n10) & land
        G = land & n10 & (dd <= hat + 3)
        return C | G | rv, C | rv
    lay = {}
    for sh in (-1.2, -0.8, -0.4, 0.0, 0.4, 0.8, 1.2):
        D, B = layers(d + sh, e.esl_base, e.hat_base)
        tag = f"s{sh:+.1f}".replace(".", "")
        lay[f"D_{tag}"], lay[f"B_{tag}"] = D, B
    # open coast within 5 km
    oc = np.zeros(shape, bool)
    m = near[5] & (pop > 0)
    if m.any():
        rr, cc = np.where(m)
        lon, lat = grid_transform(g) * (cc + 0.5, rr + 0.5)
        dist = V110.query(xyz(np.asarray(lat), np.asarray(lon)))[0] * 6371
        oc[rr[dist <= 40], cc[dist <= 40]] = True
    D0, B0 = lay["D_s+00"], lay["B_s+00"]
    lay["D_oc5"], lay["B_oc5"], lay["zone_oc5"] = D0 & oc, B0 & oc, land & oc
    lay["D_5km"], lay["B_5km"] = D0 & near[5], B0 & near[5]
    # sea level and tide referenced to 2020, and 2050 with tidal amplitude change kept separate
    lay["D_now20"], lay["B_now20"] = layers(d, e.esl_base + s.r20 + s.dhat20, e.hat_base + s.r20 + s.dhat20)
    lay["D_85t"], lay["B_85t"] = layers(d, e.esl_85_2050, e.hat_base + s.dhat50 + e.rslr_85_2050)
    lay["D_85"], _ = layers(d, e.esl_85_2050, e.hat_base + e.rslr_85_2050)
    out = {"ID_UC_G0": uid}
    for k, msk in lay.items():
        out[f"pop_{k}"] = float(pop[msk].sum())
        if k in ("D_now20", "D_85t", "D_85", "D_s+00"):
            out[f"pop50_{k}"] = float(pop50[msk].sum())
    return out


if __name__ == "__main__":
    ids = list(M.grids)
    with Pool(2) as p:
        res = p.map(city, ids, chunksize=8)
    pd.DataFrame(res).to_csv("/home/claude/atlas/city_exposure_v07.csv", index=False)
    print("done", len(res))
