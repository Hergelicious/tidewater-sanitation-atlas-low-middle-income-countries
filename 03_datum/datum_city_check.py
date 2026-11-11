"""Exposure in four test cities with Seeger and Minderhoud's sea-level-referenced DeltaDTM (MDT HYBRID-CNES-CLS22, GOCO06s)
in place of our DTU22MDT-referenced elevation, within the area covered by the test tiles.
Writes supplementary/Table_S19b_datum_city_exposure.csv"""
import numpy as np, pandas as pd, rasterio
from rasterio.warp import reproject, Resampling
import metrics as M
from metrics import bathtub, coast_masks, ocean_seeds, load, grids, cities, esl, grid_transform, geometry_mask
from pipeline import CRS

atl = pd.read_csv("atlas_cities.csv")
nm = atl.GC_UCN_MAI_2025.str.replace(r"\s*\[.*\]", "", regex=True)
rows = []
for city, tile in [("Lagos", "N06E003"), ("Jakarta", "S07E106"), ("Manila", "N14E120"), ("Chattogram", "N22E091")]:
    uid = int(atl.loc[nm == city, "ID_UC_G0"].iloc[0]); nf = float(atl.loc[nm == city, "nonsewer_frac"].iloc[0])
    g = grids[uid]; shape = (g["h"], g["w"]); T = grid_transform(g)
    inside = ~geometry_mask([cities.loc[uid, "geometry"]], out_shape=shape, transform=T)
    pop = np.nan_to_num(load(uid, "pop", shape)) * inside
    sea = np.nan_to_num(load(uid, "sea", shape)) >= 0.5; riv = np.nan_to_num(load(uid, "riv", shape))
    e = esl.loc[uid]; near, ocean = coast_masks(uid, g, shape); seeds = ocean_seeds(sea, ocean); land, n10 = ~sea, near[10]
    rv = land & (riv > 0) & (riv < 100) & n10
    with rasterio.open(f"/mnt/user-data/uploads/DeltaDTM_v1_1_{tile}_GOCO06s_MDT.tif") as s:
        a = s.read(1).astype("float32"); a[(a < -1e30) | (a <= -9990)] = np.nan
        sg = np.full(shape, np.nan, "float32")
        reproject(a, sg, src_transform=s.transform, src_crs=CRS, src_nodata=np.nan, dst_transform=T, dst_crs=CRS, dst_nodata=np.nan, resampling=Resampling.average)
    ours = load(uid, M.DTMNAME, shape)
    cov = np.isfinite(sg) & np.isfinite(ours)
    out = {"city": city, "pop_share_covered": float(pop[cov & n10].sum() / max(pop[n10].sum(), 1))}
    for lab, dem in (("ours_DTU22MDT", ours), ("Seeger_Minderhoud", sg)):
        d = np.where(np.isfinite(dem), dem, 9999.0)
        wet = (bathtub(d, seeds, e.esl_base, n10) & land) | (land & n10 & (d <= e.hat_base + 3)) | rv
        out[f"nonsewered_wet_M_{lab}"] = float((pop[wet & cov]).sum() * nf / 1e6)
    out["median_elev_diff_m"] = float(np.nanmedian((ours - sg)[cov & n10 & (pop > 0)]))
    rows.append(out)
t = pd.DataFrame(rows)
t["change_pct"] = 100 * (t.nonsewered_wet_M_Seeger_Minderhoud / t.nonsewered_wet_M_ours_DTU22MDT - 1)
print(t.round(3).to_string(index=False))
t.round(3).to_csv("supplementary/Table_S19b_datum_city_exposure.csv", index=False)
