"""Additional validation wells from the Coastal GroundWater Level dataset (CGWL; Nolte et al. 2026): low- and middle-income
countries within 10 km of the coast, records extending past 2000, deduplicated against IGRAC wells within 200 m."""
import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from concurrent.futures import ThreadPoolExecutor
from scipy.spatial import cKDTree

sys.path.insert(0, "/home/claude/atlas")
from httpzip import open_remote_zip  # noqa: E402
from pipeline import DTM_BASE, DTM_ZIPS  # noqa: E402

D = "/home/claude/data/cgwl"
_w = pd.read_csv(f"{D}/03_Wells/wells.csv", low_memory=False)
_s = pd.read_csv(f"{D}/02_Statistics/statistics_monthly.csv")
w = _w[_w.country.isin(["ZAF", "IND", "MEX", "MOZ"])].merge(_s, on="well_id")
w = w[(w.distance <= 10) & (w.end_date >= "2000")].rename(columns={"lat_deg": "lat", "lon_deg": "lon", "GWL_median": "dtw"})
w["id"] = w.well_id
_ig = pd.read_csv("/home/claude/data/igrac/lmic_wells_validation.csv")
def _xyz(la, lo):
    la, lo = np.radians(la), np.radians(lo)
    return np.c_[np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]
_d, _ = cKDTree(_xyz(_ig.lat.values, _ig.lon.values)).query(_xyz(w.lat.values, w.lon.values))
w = w[_d * 6371 > 0.2].copy()
print("CGWL LMIC wells after dedup:", len(w), w.country.value_counts().to_dict(), flush=True)
w = w[w.dtw.between(-2, 200)].copy()
w["tlon"], w["tlat"] = np.floor(w.lon).astype(int), np.floor(w.lat).astype(int)
t = gpd.read_file("/home/claude/data/deltadtm/tiles.gpkg")
tiles = w.groupby(["tlat", "tlon"]).size().rename("n").reset_index().merge(
    t[["longitude", "latitude", "tile", "zipfile"]], left_on=["tlon", "tlat"], right_on=["longitude", "latitude"])
out_f = f"{D}/wells_dtm.csv"
done = set(pd.read_csv(out_f).tile) if os.path.exists(out_f) else set()
tiles = tiles[~tiles.tile.isin(done)]
members = {z: {os.path.basename(m): m for m in open_remote_zip(DTM_BASE + DTM_ZIPS[z]).namelist()} for z in tiles.zipfile.unique()}


def fetch(row):
    return row, open_remote_zip(DTM_BASE + DTM_ZIPS[row.zipfile]).read(members[row.zipfile][row.tile])


rows = [r for r in tiles.itertuples() if r.tile in members[r.zipfile]]
for k in range(0, len(rows), 8):
    with ThreadPoolExecutor(4) as ex:
        res = list(ex.map(fetch, rows[k:k + 8]))
    for row, data in res:
        ww = w[(w.tlon == row.tlon) & (w.tlat == row.tlat)]
        with rasterio.MemoryFile(data) as mf, mf.open() as src:
            a = src.read(1).astype("float32")
            a[a == -9999] = np.nan
            r_, c_ = rasterio.transform.rowcol(src.transform, ww.lon.values, ww.lat.values)
            r_, c_ = np.clip(r_, 0, a.shape[0] - 1), np.clip(c_, 0, a.shape[1] - 1)
            win = [np.nanmean(a[max(i - 1, 0):i + 2, max(j - 1, 0):j + 2]) if np.isfinite(a[max(i - 1, 0):i + 2, max(j - 1, 0):j + 2]).any() else np.nan
                   for i, j in zip(r_, c_)]
        pd.DataFrame({"id": ww.id.values, "tile": row.tile, "dtm_90m": win}).to_csv(
            out_f, mode="a", header=not os.path.exists(out_f), index=False)
    del res
    print(f"{min(k + 8, len(rows))}/{len(rows)} tiles", flush=True)

s = pd.read_csv(out_f).drop_duplicates("id")
v = w.merge(s, on="id")
v = v[v.dtm_90m.notna()].copy()
hat = pd.read_csv("/home/claude/data/esl/max_Astronomical_Tide_Baseline.csv").dropna(subset=["valueMedian"])


def xyz(la, lo):
    la, lo = np.radians(la), np.radians(lo)
    return np.c_[np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]


_, i = cKDTree(xyz(hat.latitude.values, hat.longitude.values)).query(xyz(v.lat.values, v.lon.values), k=3)
v["hat"] = hat.valueMedian.values[i].mean(axis=1)
v["e_rel"] = v.dtm_90m - v.hat
v.to_csv(f"{D}/cgwl_lmic_wells_validation.csv", index=False)
print(len(v), v.country.value_counts().to_dict())


