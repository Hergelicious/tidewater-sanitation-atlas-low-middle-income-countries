"""Sample DeltaDTM bare-earth elevation at coastal monitoring wells (Jasechko et al. 2024)."""
import os
import sys
import numpy as np
import pandas as pd
import rasterio
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "/home/claude/atlas")
from httpzip import open_remote_zip  # noqa: E402
from pipeline import DTM_BASE, DTM_ZIPS  # noqa: E402

GW = "/home/claude/data/gw"
wells = pd.read_csv(f"{GW}/coastal_wells.csv")
wells = wells[wells.dtw.between(-2, 200)].copy()
wells["tlon"], wells["tlat"] = np.floor(wells.lon).astype(int), np.floor(wells.lat).astype(int)
tiles = pd.read_csv(f"{GW}/gw_tiles.csv")
out_f = f"{GW}/wells_dtm.csv"
done = set(pd.read_csv(out_f).tile) if os.path.exists(out_f) else set()
tiles = tiles[~tiles.tile.isin(done)]
members = {}
for z in tiles.zipfile.unique():
    zf = open_remote_zip(DTM_BASE + DTM_ZIPS[z])
    members[z] = {os.path.basename(m): m for m in zf.namelist()}


def fetch(row):
    zf = open_remote_zip(DTM_BASE + DTM_ZIPS[row.zipfile])
    return row, zf.read(members[row.zipfile][row.tile])


def sample(row, data):
    w = wells[(wells.tlon == row.tlon) & (wells.tlat == row.tlat)]
    with rasterio.MemoryFile(data) as mf, mf.open() as src:
        a = src.read(1).astype("float32")
        a[a == -9999] = np.nan
        r, c = rasterio.transform.rowcol(src.transform, w.lon.values, w.lat.values)
        r, c = np.clip(r, 0, a.shape[0] - 1), np.clip(c, 0, a.shape[1] - 1)
        pt = a[r, c]
        win = np.array([np.nanmean(a[max(i - 1, 0):i + 2, max(j - 1, 0):j + 2]) if np.isfinite(a[max(i - 1, 0):i + 2, max(j - 1, 0):j + 2]).any() else np.nan
                        for i, j in zip(r, c)])
    return pd.DataFrame({"StnID": w.StnID.values, "tile": row.tile, "dtm_pt": pt, "dtm_90m": win})


rows = list(tiles.itertuples())
for k in range(0, len(rows), 8):  # bounded batches keep memory low
    with ThreadPoolExecutor(4) as ex:
        res = list(ex.map(fetch, rows[k:k + 8]))
    for row, data in res:
        df = sample(row, data)
        df.to_csv(out_f, mode="a", header=not os.path.exists(out_f), index=False)
    del res
    print(f"{min(k + 8, len(rows))}/{len(rows)}", flush=True)
