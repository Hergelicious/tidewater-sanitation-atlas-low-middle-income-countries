"""Independent groundwater check: Fan et al. (2017) 30 arc-second annual-mean water table depth.
Per city, the Fan model is sampled (nearest cell) on the 3 arc-second grid, and its shallow-groundwater classes
(depth <= 1, 3, 5 m) are compared with the tidal rule (elevation <= HAT + 3 m) within the 10 km coastal zone.
Writes city_fan_compare.csv."""
import glob
import os
import numpy as np
DTMNAME = os.environ.get("DTMNAME", "dtmmsl")  # DeltaDTM referenced to local mean sea level (central); 'dtm' = EGM2008
import pandas as pd
import xarray as xr
from multiprocessing import Pool

from metrics import bathtub, coast_masks, ocean_seeds, load, grids, cities, esl, grid_transform, geometry_mask

ATL = "/home/claude/atlas"
FAN = {}
for f in sorted(glob.glob("/home/claude/data/fanwtd/*_WTD_annualmean.nc")):
    d = xr.open_dataset(f)
    FAN[f] = (d.lat.values.astype("float64"), d.lon.values.astype("float64"))


def nearest(axis, x):
    i = np.clip(np.searchsorted(axis, x), 1, axis.size - 1)
    i = np.where(np.abs(axis[i - 1] - x) <= np.abs(axis[i] - x), i - 1, i)
    ok = (x >= axis[0] - 0.005) & (x <= axis[-1] + 0.005)
    return i, ok


def fan_on_grid(g, shape):
    """Depth to water table (m, positive down) on the city grid; NaN outside the model domain."""
    t = grid_transform(g)
    lon = t.c + t.a * (np.arange(shape[1]) + 0.5)
    lat = t.f + t.e * (np.arange(shape[0]) + 0.5)
    out = np.full(shape, np.nan, "float32")
    for f, (la, lo) in FAN.items():
        ii, oki = nearest(la, lat)
        jj, okj = nearest(lo, lon)
        if not oki.any() or not okj.any():
            continue
        i0, i1, j0, j1 = ii[oki].min(), ii[oki].max() + 1, jj[okj].min(), jj[okj].max() + 1
        with xr.open_dataset(f) as d:
            w = d.WTD.isel(time=0, lat=slice(i0, i1), lon=slice(j0, j1)).values.astype("float32")
            m = d["mask"].isel(lat=slice(i0, i1), lon=slice(j0, j1)).values
        w[m == 0] = np.nan
        sub = np.full(shape, np.nan, "float32")
        r = np.where(oki)[0]
        c = np.where(okj)[0]
        sub[np.ix_(r, c)] = w[np.ix_(ii[oki] - i0, jj[okj] - j0)]
        out = np.where(np.isnan(out), -sub, out)
    return out


def city(uid):
    g = grids[uid]
    shape = (g["h"], g["w"])
    inside = ~geometry_mask([cities.loc[uid, "geometry"]], out_shape=shape, transform=grid_transform(g))
    pop = np.nan_to_num(load(uid, "pop", shape)) * inside
    dtm = load(uid, DTMNAME, shape)
    sea = np.nan_to_num(load(uid, "sea", shape)) >= 0.5
    riv = np.nan_to_num(load(uid, "riv", shape))
    d = np.where(np.isfinite(dtm), dtm, 9999.0)
    e = esl.loc[uid]
    near, ocean = coast_masks(uid, g, shape)
    seeds = ocean_seeds(sea, ocean)
    land, n10 = ~sea, near[10]
    rv = land & (riv > 0) & (riv < 100) & n10
    coast = bathtub(d, seeds, e.esl_base, n10) & land
    gw3 = land & n10 & (d <= e.hat_base + 3)
    wtd = fan_on_grid(g, shape)
    has = np.isfinite(wtd) & land & n10
    o = {"ID_UC_G0": uid, "pop_zone": float(pop[land & n10].sum()), "pop_fan_cover": float(pop[has].sum()),
         "pop_gw3_cov": float(pop[gw3 & has].sum())}
    for k in (1, 3, 5):
        fk = has & (wtd <= k)
        o[f"pop_fan{k}"] = float(pop[fk].sum())
        o[f"pop_wetfan{k}"] = float(pop[coast | fk | rv].sum())
    f3 = has & (wtd <= 3)
    o["pop_both3"] = float(pop[gw3 & f3].sum())
    o["pop_gw3_only"] = float(pop[gw3 & has & ~f3].sum())
    o["pop_fan3_only"] = float(pop[f3 & ~gw3].sum())
    o["pop_neither"] = float(pop[has & ~gw3 & ~f3].sum())
    o["pop_wet_tidal_or_fan3"] = float(pop[coast | gw3 | f3 | rv].sum())
    o["pop_wtd_median_on_gw3"] = float(np.nanmedian(wtd[gw3 & has & (pop > 0)])) if (gw3 & has & (pop > 0)).any() else np.nan
    return o


if __name__ == "__main__":
    with Pool(2) as p:
        res = p.map(city, list(grids), chunksize=8)
    out = pd.DataFrame(res)
    out.to_csv(f"{ATL}/city_fan_compare.csv", index=False)
    print(out.describe().T.round(0).to_string())
