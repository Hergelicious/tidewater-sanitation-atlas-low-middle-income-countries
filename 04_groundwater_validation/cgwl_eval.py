"""Score the tidal rule (mean-sea-level referenced) at the additional CGWL low- and middle-income wells.
Writes supplementary/Table_S20_cgwl_validation.csv"""
import numpy as np, pandas as pd
from datum_dtu import offset_at
v = pd.read_csv("/home/claude/data/cgwl/cgwl_lmic_wells_validation.csv")
v = v[v.dtw.between(-2, 200) & v.e_rel.notna()].copy()
v["e_msl"] = v.e_rel - offset_at(v.lat.values, v.lon.values)


def wilson(k, n, z=1.96):
    p = k / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return f"{p:.2f} ({c - h:.2f} to {c + h:.2f})"


rows = []
for name, x in [("CGWL, all", v), ("CGWL, India", v[v.country == "IND"]), ("CGWL, South Africa", v[v.country == "ZAF"])]:
    for thr in (3, 5):
        p, o = x.e_msl <= 3, x.dtw <= thr
        tp, fp, fn, tn = (p & o).sum(), (p & ~o).sum(), (~p & o).sum(), (~p & ~o).sum()
        rows.append(dict(set=name, obs_threshold_m=thr, wells=len(x), flagged_pct=round(100 * p.mean(), 1), obs_shallow_pct=round(100 * o.mean(), 1),
                         PPV=wilson(tp, tp + fp) if tp + fp else "n/a", sensitivity=wilson(tp, tp + fn) if tp + fn else "n/a",
                         specificity=wilson(tn, tn + fp) if tn + fp else "n/a",
                         median_depth_flagged=round(x.loc[p, "dtw"].median(), 2) if p.any() else np.nan))
t = pd.DataFrame(rows)
pd.set_option("display.width", 220)
print(t.to_string(index=False))
t.to_csv("supplementary/Table_S20_cgwl_validation.csv", index=False)
