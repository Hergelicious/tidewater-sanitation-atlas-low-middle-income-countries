"""Tidewater Sanitation Atlas: per-city layer extraction on the GHS-POP 3 arc-second lattice.

Usage: python pipeline.py grids|pop|river|dtm|vous
Each city gets a folder cities/<ID>/ with float32 arrays on a common 3" grid:
  pop.npy (people, GHS-POP 2020), riv.npy (GloFAS RP100 depth, m),
  dtm.npy (DeltaDTM v1.1 mean elevation m; -9999 sea; NaN no tile), vous.npy (RP100 surge depth, m)
"""
import io
import json
import math
import os
import sys
import zipfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from affine import Affine
from rasterio.warp import reproject, Resampling

sys.path.insert(0, "/home/claude/atlas")
from httpzip import open_remote_zip  # noqa: E402

ROOT = "/home/claude/atlas"
CDIR = f"{ROOT}/cities"
RES = 0.0008333333300326821
X0, Y0 = -180.0079159313, 89.0995831776456
PAD = 0.01
CRS = "EPSG:4326"
GHSL = ("https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_POP_GLOBE_R2023A/"
        "GHS_POP_E2020_GLOBE_R2023A_4326_3ss/V1-0/tiles/")
RIVER = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/CEMS-GLOFAS/flood_hazard/RP100/"
DTM_ZIPS = {
    "Africa.zip": "22ffa027-184b-4f67-9979-c182f3dfb1ab", "Asia.zip": "672eba4c-1334-44c6-8119-8879ded25912",
    "Europe.zip": "cb0b8ee3-b018-4828-a74e-2fb05020b1b6", "North_America.zip": "037664c6-1494-4889-9689-a56570728320",
    "Oceania.zip": "de972de1-26bd-4303-afdf-21a90a232cff", "South_America.zip": "db980f00-63cd-4a07-a4df-55ab06510594",
    "Seven_seas_(open_ocean).zip": "fe986ba6-3db9-40e2-8a49-0fcdb341244a"}
DTM_BASE = "https://data.4tu.nl/file/1da2e70f-6c4d-4b03-86bd-b53e789cc629/"
VOUS = "/home/claude/data/vous"


def log(*a):
    print(*a, flush=True)


def load_grids():
    with open(f"{ROOT}/grids.json") as f:
        return {int(k): v for k, v in json.load(f).items()}


def grid_transform(g):
    return Affine(RES, 0, X0 + g["col0"] * RES, 0, -RES, Y0 - g["row0"] * RES)


def grid_bounds(g):
    t = grid_transform(g)
    return (t.c, t.f - g["h"] * RES, t.c + g["w"] * RES, t.f)


def intersects(b, g):
    gb = grid_bounds(g)
    return not (b[2] <= gb[0] or b[0] >= gb[2] or b[3] <= gb[1] or b[1] >= gb[3])


def arr_path(uid, name):
    return f"{CDIR}/{uid}/{name}.npy"


def get_arr(uid, g, name, fill):
    p = arr_path(uid, name)
    if os.path.exists(p):
        return np.load(p)
    return np.full((g["h"], g["w"]), fill, "float32")


def warp_into(src_ds, uid, g, name, fill, resampling, combine):
    dst = np.full((g["h"], g["w"]), np.nan, "float32")
    reproject(rasterio.band(src_ds, 1), dst, dst_transform=grid_transform(g), dst_crs=CRS,
              dst_nodata=np.nan, resampling=resampling)
    cur = get_arr(uid, g, name, fill)
    cur = combine(cur, dst)
    np.save(arr_path(uid, name), cur)


def stage_grids():
    c = gpd.read_file(f"{ROOT}/cities.gpkg").to_crs(4326)
    grids = {}
    for _, r in c.iterrows():
        x0, y0, x1, y1 = r.geometry.bounds
        col0 = int(math.floor((x0 - PAD - X0) / RES))
        col1 = int(math.ceil((x1 + PAD - X0) / RES))
        row0 = int(math.floor((Y0 - (y1 + PAD)) / RES))
        row1 = int(math.ceil((Y0 - (y0 - PAD)) / RES))
        grids[int(r.ID_UC_G0)] = {"col0": col0, "row0": row0, "w": col1 - col0, "h": row1 - row0}
        os.makedirs(f"{CDIR}/{int(r.ID_UC_G0)}", exist_ok=True)
    with open(f"{ROOT}/grids.json", "w") as f:
        json.dump(grids, f)
    log("grids", len(grids))


def first_valid(cur, new):
    return np.where(np.isnan(cur) & ~np.isnan(new), new, cur)


def max_valid(cur, new):
    return np.where(~np.isnan(new), np.fmax(cur, new), cur)


def stage_pop():
    grids = load_grids()
    tiles = pd.read_csv(f"{ROOT}/pop_tiles.csv").iloc[:, 0].tolist()
    os.makedirs("/home/claude/tmp_pop", exist_ok=True)

    def fetch(t):
        p = f"/home/claude/tmp_pop/{t}.zip"
        if not os.path.exists(p + ".done"):
            urllib.request.urlretrieve(f"{GHSL}GHS_POP_E2020_GLOBE_R2023A_4326_3ss_V1_0_{t}.zip", p)
            open(p + ".done", "w").close()
        return t, p

    with ThreadPoolExecutor(3) as ex:
        futs = [ex.submit(fetch, t) for t in tiles]
        for i, f in enumerate(as_completed(futs)):
            try:
                t, p = f.result()
            except Exception as e:
                log("POP FAIL", e)
                continue
            with zipfile.ZipFile(p) as z:
                member = [n for n in z.namelist() if n.endswith(".tif")][0]
            with rasterio.open(f"zip://{p}!/{member}") as src:
                for uid, g in grids.items():
                    if intersects(src.bounds, g):
                        warp_into(src, uid, g, "pop", np.nan, Resampling.nearest, first_valid)
            os.remove(p)
            log(f"pop {i + 1}/{len(tiles)} {t}")


def stage_river():
    grids = load_grids()
    tiles = pd.read_csv(f"{ROOT}/river_tiles.csv")

    def fetch(row):
        url = f"{RIVER}ID{row.id}_{row.name}_RP100_depth.tif"
        return row.name, urllib.request.urlopen(url, timeout=300).read()

    with ThreadPoolExecutor(4) as ex:
        futs = [ex.submit(fetch, r) for r in tiles.itertuples()]
        for i, f in enumerate(as_completed(futs)):
            try:
                name, data = f.result()
            except Exception as e:
                log("RIVER FAIL", e)
                continue
            with rasterio.MemoryFile(data) as mf, mf.open() as src:
                for uid, g in grids.items():
                    if intersects(src.bounds, g):
                        warp_into(src, uid, g, "riv", np.nan, Resampling.nearest, max_valid)
            log(f"river {i + 1}/{len(tiles)} {name}")


def stage_dtm():
    grids = load_grids()
    tiles = pd.read_csv(f"{ROOT}/dtm_tiles.csv")
    done_f = f"{ROOT}/dtm_done.txt"
    done = set(open(done_f).read().split()) if os.path.exists(done_f) else set()
    tiles = tiles[~tiles.tile.isin(done)]
    zips = {}

    def zf(name):
        if name not in zips:
            zips[name] = open_remote_zip(DTM_BASE + DTM_ZIPS[name])
        return zips[name]

    for name in tiles.zipfile.unique():
        zf(name)
    members = {n: {os.path.basename(m): m for m in zips[n].namelist()} for n in zips}

    def fetch(row):
        z = open_remote_zip(DTM_BASE + DTM_ZIPS[row.zipfile])  # own handle per thread
        return row.tile, z.read(members[row.zipfile][row.tile])

    with ThreadPoolExecutor(4) as ex:
        futs = [ex.submit(fetch, r) for r in tiles.itertuples() if r.tile in members[r.zipfile]]
        for i, f in enumerate(as_completed(futs)):
            try:
                tile, data = f.result()
            except Exception as e:
                log("DTM FAIL", e)
                continue
            with rasterio.MemoryFile(data) as mf, mf.open() as src:
                arr = src.read(1)
                sea = arr == -9999
                arr = np.where(sea, np.nan, arr).astype("float32")
                for uid, g in grids.items():
                    if not intersects(src.bounds, g):
                        continue
                    for nm, a, rs in (("dtm", arr, Resampling.average), ("sea", sea.astype("float32"), Resampling.average)):
                        dst = np.full((g["h"], g["w"]), np.nan, "float32")
                        reproject(a, dst, src_transform=src.transform, src_crs=CRS, src_nodata=np.nan,
                                  dst_transform=grid_transform(g), dst_crs=CRS, dst_nodata=np.nan, resampling=rs)
                        cur = get_arr(uid, g, nm, np.nan)
                        np.save(arr_path(uid, nm), first_valid(cur, dst))
            with open(done_f, "a") as fh:
                fh.write(tile + "\n")
            log(f"dtm {i + 1}/{len(futs)} {tile}")


def stage_vous():
    grids = load_grids()
    files = [os.path.join(VOUS, f) for f in os.listdir(VOUS) if f.endswith(".tif")]
    for f in files:
        with rasterio.open(f) as src:
            for uid, g in grids.items():
                if intersects(src.bounds, g):
                    warp_into(src, uid, g, "vous", np.nan, Resampling.nearest, max_valid)
        log("vous", os.path.basename(f))


def stage_growth():
    """2050 population on the 3" grid: GHS-POP 2020 scaled by the 1 km GHS-WUP 2050/2020 ratio;
    1 km cells empty in 2020 receive their 2050 population spread evenly."""
    grids = load_grids()
    w20 = rasterio.open("/home/claude/data/wup/2020/GHS_WUP_POP_E2020_GLOBE_R2025A_4326_30ss_V1_0.tif")
    w50 = rasterio.open("/home/claude/data/wup/2050/GHS_WUP_POP_E2050_GLOBE_R2025A_4326_30ss_V1_0.tif")
    for k, (uid, g) in enumerate(grids.items()):
        tr = grid_transform(g)
        x0, y1 = tr.c, tr.f
        bounds = (x0 - 0.02, y1 - g["h"] * RES - 0.02, x0 + g["w"] * RES + 0.02, y1 + 0.02)
        win = rasterio.windows.from_bounds(*bounds, transform=w20.transform).round_offsets().round_lengths()
        a20 = w20.read(1, window=win, boundless=True, fill_value=0).astype("float64")
        a50 = w50.read(1, window=win, boundless=True, fill_value=0).astype("float64")
        a20[a20 < 0], a50[a50 < 0] = 0, 0
        ratio = np.where(a20 > 0, a50 / np.where(a20 > 0, a20, 1), np.nan).astype("float32")
        newpop = np.where(a20 <= 0, a50, 0).astype("float32")
        wtr = w20.window_transform(win)
        r3 = np.full((g["h"], g["w"]), np.nan, "float32")
        n3 = np.zeros((g["h"], g["w"]), "float32")
        reproject(ratio, r3, src_transform=wtr, src_crs=CRS, src_nodata=np.nan, dst_transform=tr, dst_crs=CRS,
                  dst_nodata=np.nan, resampling=Resampling.nearest)
        reproject(newpop, n3, src_transform=wtr, src_crs=CRS, dst_transform=tr, dst_crs=CRS,
                  resampling=Resampling.nearest)
        cell_ratio = (abs(wtr.a) / RES) * (abs(wtr.e) / RES)  # 3" pixels per 1 km cell (~100)
        p20 = np.nan_to_num(np.load(arr_path(uid, "pop")))
        p50 = np.where(np.isfinite(r3), p20 * r3, 0) + n3 / cell_ratio
        np.save(arr_path(uid, "pop50"), p50.astype("float32"))
        if k % 200 == 0:
            log("growth", k)


if __name__ == "__main__":
    {"grids": stage_grids, "pop": stage_pop, "river": stage_river,
     "dtm": stage_dtm, "vous": stage_vous, "growth": stage_growth}[sys.argv[1]]()
