"""Groundwater seasonality at IGRAC coastal wells in low- and middle-income countries.
For wells with repeated measurements since 2000, the share of observations with water within 3 m and 5 m of the surface,
classed as persistent (at least 75% of observations), intermittent (some, below 75%) or never.
Writes supplementary/Table_S7_groundwater_seasonality.csv."""
import json
import os
import numpy as np
import pandas as pd

D = "/home/claude/data/igrac"
T2000 = pd.Timestamp("2000-01-01").timestamp()
v = pd.read_csv(f"{D}/lmic_wells_validation.csv")
meta = pd.read_csv(f"{D}/lmic_coastal_wells.csv", usecols=["id", "ground_surface_elevation"]).drop_duplicates("id")
v = v.merge(meta, on="id", how="left")
a = pd.read_csv("/home/claude/data/gw/wells_all_msl.csv")
a = a[a.set == "Low and middle income (IGRAC)"].reset_index(drop=True)
vv = v[v.dtw.between(-2, 200) & v.e_rel.notna()].reset_index(drop=True)
assert len(a) == len(vv) and np.allclose(a.lon, vv.lon)
vv["e_rel"] = a.e_msl.values  # mean-sea-level referenced elevation (central)
v = vv

rec = []
for r in v.itertuples():
    p = f"{D}/levels/{r.id}.json"
    if not os.path.exists(p):
        continue
    df = pd.DataFrame(json.load(open(p)).get("data", []))
    if df.empty:
        continue
    df = df[df.dt >= T2000]
    dep = df[df.par.str.contains("ground surface", case=False, na=False)]
    if dep.empty:
        el = df[df.par.str.contains("a.m.s.l", case=False, na=False)]
        if el.empty or not np.isfinite(r.ground_surface_elevation):
            continue
        dep = el.assign(v=r.ground_surface_elevation - el.v)
    dep = dep.assign(t=pd.to_datetime(dep.dt, unit="s")).drop_duplicates("t")
    x = dep.v.values
    x = x[np.isfinite(x) & (x > -2) & (x < 200)]
    if x.size < 8 or dep.t.dt.month.nunique() < 3:
        continue
    rec.append(dict(id=r.id, iso3=r.iso3, e_rel=r.e_rel, n=x.size, months=dep.t.dt.month.nunique(),
                    median=np.median(x), p10=np.quantile(x, 0.1), f3=(x <= 3).mean(), f5=(x <= 5).mean()))
s = pd.DataFrame(rec)


def cls(f):
    return np.select([f >= 0.75, f > 0], ["persistent", "intermittent"], "never")


s["c3"], s["c5"] = cls(s.f3), cls(s.f5)
s["flag3"] = s.e_rel <= 3
print("Wells with repeated records:", len(s), s.iso3.value_counts().to_dict())
tab = pd.crosstab(s.flag3.map({True: "flagged (HAT + 3 m)", False: "not flagged"}), s.c3, normalize="index").round(3)
cnt = pd.crosstab(s.flag3.map({True: "flagged (HAT + 3 m)", False: "not flagged"}), s.c3)
print(cnt, "\n", tab)
nf = s[~s.flag3]
print("Not flagged, median deeper than 3 m but reach 3 m in at least 10% of observations:",
      int(((nf["median"] > 3) & (nf.f3 >= 0.1)).sum()), "of", int((nf["median"] > 3).sum()))
fl = s[s.flag3]
print("Flagged wells: median share of observations within 3 m", round(fl.f3.median(), 2), "; within 5 m", round(fl.f5.median(), 2))
out = pd.concat([cnt.add_suffix("_n"), tab.add_suffix("_share")], axis=1)
out.to_csv("/home/claude/atlas/supplementary/Table_S7_groundwater_seasonality.csv")
