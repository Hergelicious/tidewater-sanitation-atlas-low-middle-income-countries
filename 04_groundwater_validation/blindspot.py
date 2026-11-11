"""Size of the elevated-aquifer blind spot: residents where the Fan et al. water table lies within 1 m (or 2 m) but the
ground is not wet ground under the tidal rule, coastal or river flooding; checked at wells and at the published sites.
Writes city_blindspot.csv and supplementary/Table_S30_blindspot.csv"""
import glob
import numpy as np
import pandas as pd
from multiprocessing import Pool
from scipy.spatial import cKDTree
import metrics as M
from fan_compare import fan_on_grid
from pipeline import grid_transform
from rasterio.features import geometry_mask


def city(uid):
    g = M.grids[uid]
    shape = (g["h"], g["w"])
    inside = ~geometry_mask([M.cities.loc[uid, "geometry"]], out_shape=shape, transform=grid_transform(g))
    pop = np.nan_to_num(M.load(uid, "pop", shape)) * inside
    dtm = M.load(uid, M.DTMNAME, shape)
    sea = np.nan_to_num(M.load(uid, "sea", shape)) >= 0.5
    riv = np.nan_to_num(M.load(uid, "riv", shape))
    d = np.where(np.isfinite(dtm), dtm, 9999.0)
    e = M.esl.loc[uid]
    near, ocean = M.coast_masks(uid, g, shape)
    seeds = M.ocean_seeds(sea, ocean)
    land, n10 = ~sea, near[10]
    D = (M.bathtub(d, seeds, e.esl_base, n10) & land) | (land & n10 & (d <= e.hat_base + 3)) | (land & (riv > 0) & (riv < 100) & n10)
    wtd = fan_on_grid(g, shape)
    has = np.isfinite(wtd) & land & n10
    o = {"ID_UC_G0": uid, "pop_zone_fancover": float(pop[has].sum())}
    for k in (1, 2):
        m = has & (wtd <= k) & ~D
        o[f"pop_fan{k}_notwet"] = float(pop[m].sum())
        o[f"pop_fan{k}_notwet_hi10"] = float(pop[m & (d > e.hat_base + 10)].sum())
    return o


if __name__ == "__main__":
    with Pool(2) as p:
        res = pd.DataFrame(p.map(city, list(M.grids), chunksize=8))
    res.to_csv("/home/claude/atlas/city_blindspot.csv", index=False)
    a = pd.read_csv("/home/claude/atlas/atlas_cities.csv").set_index("ID_UC_G0")
    r = res.set_index("ID_UC_G0").join(a[["nonsewer_frac", "GC_DEV_USR_2025"]])
    rows = []
    for k in (1, 2):
        ns = (r[f"pop_fan{k}_notwet"] * r.nonsewer_frac)
        rows.append(dict(quantity=f"Non-sewered residents outside wet ground with modelled water table within {k} m (millions)",
                         value=round(ns.sum() / 1e6, 1)))
        rows.append(dict(quantity=f"  of which on ground more than 10 m above HAT (millions)",
                         value=round((r[f"pop_fan{k}_notwet_hi10"] * r.nonsewer_frac).sum() / 1e6, 1)))
        for reg, v in ns.groupby(r.GC_DEV_USR_2025).sum().sort_values(ascending=False).items():
            if v > 5e5:
                rows.append(dict(quantity=f"  {reg} (within {k} m)", value=round(v / 1e6, 1)))
    # wells: does the Fan model add skill where the rule does not flag?
    exec(open("/home/claude/atlas/ml_groundwater.py").read().split("def xyz")[0].split("w = pd.read_csv")[0])
    w = pd.read_csv("/home/claude/data/gw/wells_all_msl_dist.csv")
    src = open("/home/claude/atlas/ml_groundwater.py").read()
    exec(src[src.index('w["fan"] = np.nan'):src.index("def xyz")])
    L = w[w.set.str.startswith("Low")].copy()
    y = L.dtw <= 3
    rule = L.e_msl <= 3
    for k in (1, 2):
        f = (L.fan <= k) & ~rule
        rows.append(dict(quantity=f"LMIC wells not flagged by the rule but with modelled water table within {k} m: n, share shallow (%)",
                         value=f"{int(f.sum())}, {100 * (y & f).sum() / max(f.sum(), 1):.0f}"))
        u = rule | (L.fan <= k)
        rows.append(dict(quantity=f"Rule OR model within {k} m at LMIC wells: precision, detection (%)",
                         value=f"{100 * (y & u).sum() / u.sum():.0f}, {100 * (y & u).sum() / y.sum():.0f}"))
    rows.append(dict(quantity="LMIC wells not flagged by the rule: share shallow (%) (comparison)", value=f"{100 * y[~rule].mean():.0f}"))
    rows.append(dict(quantity="Rule alone at LMIC wells: precision, detection (%)",
                     value=f"{100 * (y & rule).sum() / rule.sum():.0f}, {100 * (y & rule).sum() / y.sum():.0f}"))
    # published sites
    s = pd.read_csv("/home/claude/atlas/supplementary/Table_S29_site_evidence.csv")
    s["fan_wtd_m"] = np.nan
    for i, rr in s.iterrows():
        for f in sorted(glob.glob("/home/claude/data/fanwtd/*_WTD_annualmean.nc")):
            import xarray as xr
            with xr.open_dataset(f) as dd:
                la, lo = dd.lat.values, dd.lon.values
                if la[0] <= rr.lat <= la[-1] and lo[0] <= rr.lon <= lo[-1]:
                    ii, jj = int(np.argmin(abs(la - rr.lat))), int(np.argmin(abs(lo - rr.lon)))
                    win = -dd.WTD.isel(time=0, lat=slice(max(ii - 2, 0), ii + 3), lon=slice(max(jj - 2, 0), jj + 3)).values
                    mk = dd["mask"].isel(lat=slice(max(ii - 2, 0), ii + 3), lon=slice(max(jj - 2, 0), jj + 3)).values
                    win = np.where(mk == 1, win, np.nan)
                    s.loc[i, "fan_wtd_m"] = round(float(np.nanmedian(win)), 1)
                    break
    s.to_csv("/home/claude/atlas/supplementary/Table_S29_site_evidence.csv", index=False)
    print(s[["city", "site", "wet_ground_pct", "median_elev_above_HAT_m", "fan_wtd_m"]].to_string(index=False))
    t = pd.DataFrame(rows)
    t.to_csv("/home/claude/atlas/supplementary/Table_S30_blindspot.csv", index=False)
    print(t.to_string(index=False))
