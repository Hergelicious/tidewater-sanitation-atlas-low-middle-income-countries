"""Stage DeltaDTM v1.1 referenced to local mean sea level (Seeger & Minderhoud 2026; MDT HYBRID-CNES-CLS22 with GOCO06s)
into each city grid as dtmmsl.npy, with the same 3 arc-second averaging as the original EGM2008 stage.
Also records the per-tile median offset (EGM2008 elevation minus MSL elevation) in dtm_msl_offsets.csv."""
import os
import sys
import urllib.request
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "/home/claude/atlas")
from pipeline import load_grids, grid_transform, intersects, arr_path, get_arr, first_valid, CRS  # noqa: E402

ATL = "/home/claude/atlas"
TMP = "/home/claude/tmp_msl"
os.makedirs(TMP, exist_ok=True)
urls = pd.read_csv("/home/claude/data/mdt/DeltaDTM_v1.1.csv")
urls["key"] = [u.split("/")[-1].replace("_GOCO06s_MDT.tif", ".tif") for u in urls.url]
tiles = pd.read_csv(f"{ATL}/dtm_tiles.csv")
tiles = tiles.merge(urls[["key", "url"]], left_on="tile", right_on="key", how="left")
print("tiles", len(tiles), "with MSL version", tiles.url.notna().sum(), flush=True)
done_f = f"{ATL}/dtm_msl_done.txt"
done = set(open(done_f).read().split()) if os.path.exists(done_f) else set()
todo = tiles[tiles.url.notna() & ~tiles.tile.isin(done)]
grids = load_grids()


def fetch(r):
    p = f"{TMP}/{r.tile}"
    if not os.path.exists(p + ".ok"):
        for attempt in range(4):
            try:
                urllib.request.urlretrieve(r.url, p)
                open(p + ".ok", "w").close()
                break
            except Exception as e:  # noqa: BLE001
                err = e
        else:
            raise err
    return r, p


rows = list(todo.itertuples())
for k in range(0, len(rows), 8):
    with ThreadPoolExecutor(4) as ex:
        res = list(ex.map(fetch, rows[k:k + 8]))
    for r, p in res:
        with rasterio.open(p) as src:
            a = src.read(1).astype("float32")
            nd = src.nodata
            bad = (a <= -9990) | (~np.isfinite(a)) | ((a == nd) if nd is not None else False)
            a[bad] = np.nan
            for uid, g in grids.items():
                if not intersects(src.bounds, g):
                    continue
                dst = np.full((g["h"], g["w"]), np.nan, "float32")
                reproject(a, dst, src_transform=src.transform, src_crs=CRS, src_nodata=np.nan,
                          dst_transform=grid_transform(g), dst_crs=CRS, dst_nodata=np.nan, resampling=Resampling.average)
                cur = get_arr(uid, g, "dtmmsl", np.nan)
                np.save(arr_path(uid, "dtmmsl"), first_valid(cur, dst))
        os.remove(p)
        with open(done_f, "a") as fh:
            fh.write(r.tile + "\n")
    print(f"{min(k + 8, len(rows))}/{len(rows)}", flush=True)
