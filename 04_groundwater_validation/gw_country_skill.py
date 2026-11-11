"""Tidal rule skill by country group at low- and middle-income wells, with Wilson 95% intervals.
Writes supplementary/Table_S13_gw_skill_by_country.csv."""
import numpy as np
import pandas as pd

v = pd.read_csv("/home/claude/data/igrac/lmic_wells_validation.csv")
v = v[v.dtw.between(-2, 200) & v.e_rel.notna()].reset_index(drop=True)
a = pd.read_csv("/home/claude/data/gw/wells_all_msl.csv")
a = a[a.set == "Low and middle income (IGRAC)"].reset_index(drop=True)
assert len(a) == len(v) and np.allclose(a.lon, v.lon)
v["e_rel"] = a.e_msl.values  # mean-sea-level referenced elevation (central)


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, c - h, c + h


groups = {"India": v.iso3 == "IND", "Low and middle income excluding India": v.iso3 != "IND"}
for c in ["JAM", "ZAF", "CUB", "THA", "BRA"]:
    groups[v.loc[v.iso3 == c, "country"].iloc[0]] = v.iso3 == c
rows = []
for name, m in groups.items():
    x = v[m]
    for thr in (3, 5):
        pred, obs = x.e_rel <= 3, x.dtw <= thr
        tp, fp, fn, tn = (pred & obs).sum(), (pred & ~obs).sum(), (~pred & obs).sum(), (~pred & ~obs).sum()
        r = dict(group=name, obs_threshold_m=thr, wells=len(x), flagged=int(pred.sum()), obs_shallow_pct=round(100 * obs.mean(), 1))
        for lab, k, n in (("PPV", tp, tp + fp), ("sensitivity", tp, tp + fn), ("specificity", tn, tn + fp)):
            p, lo, hi = wilson(k, n)
            r[lab] = f"{p:.2f} ({lo:.2f} to {hi:.2f})" if n else "n/a"
        rows.append(r)
t = pd.DataFrame(rows)
pd.set_option("display.width", 220)
print(t.to_string(index=False))
t.to_csv("supplementary/Table_S13_gw_skill_by_country.csv", index=False)
