"""Extra city-level layers requested by the second review.
1. Land subsidence to 2050: hybrid vertical land motion (Oelsmann et al. 2026) minus GIA, which the JRC relative
   sea-level projections already contain; mean of the three nearest coastal segments; applied uniformly over 30 years.
2. 2050 population growth on wet ground split into cells already populated in 2020 (at least 1 resident) and cells
   that were empty in 2020.
3. 2050 population disaggregation test: 2020 pattern scaled uniformly by each city zone's growth ratio.
Writes city_exposure_extra.csv."""
import os
import numpy as np
DTMNAME = os.environ.get("DTMNAME", "dtmmsl")  # DeltaDTM referenced to local mean sea level (central); 'dtm' = EGM2008
import pandas as pd
import xarray as xr
from multiprocessing import Pool
from scipy.spatial import cKDTree

import metrics as M
from metrics import bathtub, coast_masks, ocean_seeds, load, grids, cities, esl, grid_transform, geometry_mask

ATL = "/home/claude/atlas"
v = xr.open_dataset("/home/claude/data/vlm/Global_VLM_data_Oelsmann_2025_data_supplement.nc")
seg = pd.DataFrame({"lat": v.lat.values, "lon": v.lon.values,
                    "vlm": v.OE24_GPS_InSAR_GIA.values - v.GIA.values, "type": v.datatype.values}).dropna(subset=["vlm"])


def xyz(la, lo):
    la, lo = np.radians(la), np.radians(lo)
    return np.c_[np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]


atl = pd.read_csv(f"{ATL}/atlas_cities.csv").set_index("ID_UC_G0")
cc = cities.loc[atl.index].geometry.centroid
lat = atl.zone_lat.fillna(pd.Series(cc.y.values, index=atl.index))
lon = atl.zone_lon.fillna(pd.Series(cc.x.values, index=atl.index))
dist, idx = cKDTree(xyz(seg.lat.values, seg.lon.values)).query(xyz(lat.values, lon.values), k=3)
VLM = pd.Series(seg.vlm.values[idx].mean(1), index=atl.index)
VLM_KM = pd.Series(dist.mean(1) * 6371, index=atl.index)
YEARS = 30


def city(uid):
    g = grids[uid]
    shape = (g["h"], g["w"])
    inside = ~geometry_mask([cities.loc[uid, "geometry"]], out_shape=shape, transform=grid_transform(g))
    pop = np.nan_to_num(load(uid, "pop", shape)) * inside
    pop50 = np.nan_to_num(load(uid, "pop50", shape)) * inside
    dtm = load(uid, DTMNAME, shape)
    sea = np.nan_to_num(load(uid, "sea", shape)) >= 0.5
    riv = np.nan_to_num(load(uid, "riv", shape))
    d = np.where(np.isfinite(dtm), dtm, 9999.0)
    e = esl.loc[uid]
    near, ocean = coast_masks(uid, g, shape)
    seeds = ocean_seeds(sea, ocean)
    land, n10 = ~sea, near[10]
    rv = land & (riv > 0) & (riv < 100) & n10

    def wet(dd, esl_level, gw_level):
        return (bathtub(dd, seeds, esl_level, n10) & land) | (land & n10 & (dd <= gw_level)) | rv

    w85 = wet(d, e.esl_85_2050, e.hat_base + e.rslr_85_2050 + 3)
    gw85 = land & n10 & (d <= e.hat_base + e.rslr_85_2050 + 3)
    dz = -VLM[uid] * YEARS / 1000.0          # positive when land sinks
    ds = d - dz
    w85s = wet(ds, e.esl_85_2050, e.hat_base + e.rslr_85_2050 + 3)
    gw85s = land & n10 & (ds <= e.hat_base + e.rslr_85_2050 + 3)
    w_now = wet(d, e.esl_base, e.hat_base + 3)
    old = pop >= 1
    zp, zp50 = pop[n10].sum(), pop50[n10].sum()
    ratio = zp50 / zp if zp > 0 else np.nan
    return {"ID_UC_G0": uid, "vlm_mm_yr": float(VLM[uid]), "vlm_dist_km": float(VLM_KM[uid]),
            "pop50_wet10_85_subs": float(pop50[w85s].sum()), "pop_wet10_85_subs": float(pop[w85s].sum()),
            "pop50_gw3_85_subs": float(pop50[gw85s].sum()), "pop50_gw3_85_chk": float(pop50[gw85].sum()),
            "pop50_wet10_85_chk": float(pop50[w85].sum()),
            "pop50_wet10_85_oldcells": float(pop50[w85 & old].sum()), "pop50_wet10_85_newcells": float(pop50[w85 & ~old].sum()),
            "pop_wet10_85_oldcells": float(pop[w85 & old].sum()),
            "pop50_wet10_now_mask": float(pop50[w_now].sum()),
            "pop50u_wet10_85": float((pop[w85].sum() * ratio) if np.isfinite(ratio) else 0.0)}


if __name__ == "__main__":
    with Pool(2) as p:
        res = p.map(city, list(grids), chunksize=8)
    out = pd.DataFrame(res)
    out.to_csv(f"{ATL}/city_exposure_extra.csv", index=False)
    print(out.describe().T.round(1).to_string())
