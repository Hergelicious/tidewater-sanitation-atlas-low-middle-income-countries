"""Validate the shallow-groundwater rule at coastal wells in low- and middle-income countries (IGRAC GGMN)."""
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

D = "/home/claude/data/igrac"
w = pd.read_csv(f"{D}/lmic_wells_dtw.csv")
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
v.to_csv(f"{D}/lmic_wells_validation.csv", index=False)


def auc(score, y):
    r = pd.Series(-score).rank().values
    n1, n0 = y.sum(), (~y).sum()
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan


def ev(d, tr=3.0, to=3.0):
    y, p = (d.dtw <= to).values, (d.e_rel <= tr).values
    tp, fp, fn, tn = (p & y).sum(), (p & ~y).sum(), (~p & y).sum(), (~p & ~y).sum()
    return {"n": len(d), "flagged_pct": round(100 * p.mean(), 1), "obs_shallow_pct": round(100 * y.mean(), 1),
            "sensitivity": round(tp / max(tp + fn, 1), 2), "specificity": round(tn / max(tn + fp, 1), 2),
            "PPV": round(tp / max(tp + fp, 1), 2), "NPV": round(tn / max(tn + fn, 1), 2), "AUC": round(auc(v.e_rel.values if d is v else d.e_rel.values, y), 2)}


res = []
for name, d in [("All LMIC wells", v)] + [(c, v[v.country == c]) for c in v.country.value_counts().index if (v.country == c).sum() >= 50]:
    for tr, to in [(3, 3), (3, 5)]:
        rr = ev(d, tr, to)
        rr.update({"subset": name, "rule_m": tr, "obs_m": to})
        res.append(rr)
res = pd.DataFrame(res)[["subset", "rule_m", "obs_m", "n", "flagged_pct", "obs_shallow_pct", "sensitivity", "specificity", "PPV", "NPV", "AUC"]]
res.to_csv(f"{D}/lmic_validation_metrics.csv", index=False)
pd.set_option("display.width", 200)
print(res.to_string(index=False))
print(v.groupby(pd.cut(v.e_rel, [-5, 0, 1, 2, 3, 5, 10, 20, 30]), observed=True).dtw.describe(percentiles=[.25, .5, .75])[["count", "25%", "50%", "75%"]].round(2))
