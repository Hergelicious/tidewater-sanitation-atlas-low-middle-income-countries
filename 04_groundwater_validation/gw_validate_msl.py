"""Re-run the well validation with elevation referenced to local mean sea level (DTU22MDT offset, datum_dtu.py).
Offset at each well: weighted median, within 150 km (300 km if fewer than 10 wells), of the difference between
DeltaDTM (EGM2008, 90 m mean at the well) and GLL-DTM v2 (mean sea level) at the well.
Writes supplementary/Table_S3_groundwater_validation.csv."""
import numpy as np
import pandas as pd
import rasterio
from scipy.spatial import cKDTree

ig = pd.read_csv("/home/claude/data/igrac/lmic_wells_validation.csv").assign(set="Low and middle income (IGRAC)")
ja = pd.read_csv("/home/claude/data/gw/wells_validation.csv")
ja = ja.assign(set=np.where(ja.region == "United States", "United States", "Other high income"))
w = pd.concat([ig[["lon", "lat", "dtw", "e_rel", "dtm_90m", "set", "iso3"]], ja[["lon", "lat", "dtw", "e_rel", "dtm_90m", "set", "iso3"]]])
w = w[w.dtw.between(-2, 200) & w.e_rel.notna()].reset_index(drop=True)
from datum_dtu import offset_at
w["offset"] = offset_at(w.lat.values, w.lon.values)
w["e_msl"] = w.e_rel - w.offset.fillna(0)
print(w.groupby("set").offset.describe().round(2))


def skill(pred, obs):
    tp, fp, fn, tn = (pred & obs).sum(), (pred & ~obs).sum(), (~pred & obs).sum(), (~pred & ~obs).sum()
    return dict(flagged_pct=100 * pred.mean(), PPV=tp / max(tp + fp, 1), sensitivity=tp / max(tp + fn, 1), specificity=tn / max(tn + fp, 1))


rows = []
for s_, x in w.groupby("set"):
    for thr in (3, 5):
        for lab, e in (("EGM2008", x.e_rel), ("Mean sea level", x.e_msl)):
            rows.append(dict(set=s_, obs_threshold_m=thr, datum=lab, n=len(x), obs_shallow_pct=100 * (x.dtw <= thr).mean(),
                             median_depth_flagged=x.loc[e <= 3, "dtw"].median(), **skill((e <= 3).values, (x.dtw <= thr).values)))
x = w[w.set == "Low and middle income (IGRAC)"]
for nm, m in (("India", x.iso3 == "IND"), ("Excluding India", x.iso3 != "IND")):
    for lab, e in (("EGM2008", x.e_rel), ("Mean sea level", x.e_msl)):
        rows.append(dict(set="LMIC: " + nm, obs_threshold_m=3, datum=lab, n=int(m.sum()), obs_shallow_pct=100 * (x.dtw[m] <= 3).mean(),
                         median_depth_flagged=x.loc[m & (e <= 3), "dtw"].median(), **skill((e[m] <= 3).values, (x.dtw[m] <= 3).values)))
t = pd.DataFrame(rows).round(3)
pd.set_option("display.width", 220)
print(t.to_string(index=False))
t.to_csv("supplementary/Table_S3_groundwater_validation.csv", index=False)
w.to_csv("/home/claude/data/gw/wells_all_msl.csv", index=False)
