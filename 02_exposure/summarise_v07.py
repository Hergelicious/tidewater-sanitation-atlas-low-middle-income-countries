"""Summaries for the v0.6 review layers (non-sewered = population x 2020 non-sewered share).
Writes supplementary/Table_S26_datum_robustness.csv, Table_S27_open_coast_subset.csv, Table_S28_sea_level_2020.csv"""
import pandas as pd, numpy as np
a = pd.read_csv("atlas_cities.csv").set_index("ID_UC_G0")
v = pd.read_csv("city_exposure_v07.csv").set_index("ID_UC_G0").join(a[["nonsewer_frac", "estuarine", "GC_UCN_MAI_2025", "ns_wet10_now"]])
f = v.nonsewer_frac
S = lambda col, m=None: (v[col] * f)[m].sum() / 1e6 if m is not None else (v[col] * f).sum() / 1e6
# 1 datum
rows = []
for sh in (-1.2, -0.8, -0.4, 0.0, 0.4, 0.8, 1.2):
    t = f"s{sh:+.1f}".replace(".", "")
    D, B = S(f"pop_D_{t}"), S(f"pop_B_{t}")
    rows.append(dict(land_shift_m=sh, wet_ground_M=round(D, 1), surface_flood_M=round(B, 1), groundwater_only_M=round(D - B, 1),
                     groundwater_only_share_pct=round(100 * (D - B) / D, 1)))
t1 = pd.DataFrame(rows)
# top-20 cities: each city's shift applied alone (one city wrong at a time) -> effect on total and ranks
top = v.sort_values("ns_wet10_now", ascending=False).head(20)
man = v[v.GC_UCN_MAI_2025 == "Manila"].iloc[0]
extra = []
for nm, col in (("Metro Manila, land 1.2 m lower", "pop_D_s-12"), ("Metro Manila, land 1.2 m higher", "pop_D_s+12"), ("Metro Manila, central", "pop_D_s+00")):
    extra.append(dict(land_shift_m=nm, wet_ground_M=round(man[col] * man.nonsewer_frac / 1e6, 2)))
# worst case: top 20 cities all shifted 1.2 m in the same direction, others central
for sh, tag in ((-1.2, "s-12"), (1.2, "s+12")):
    tot = S("pop_D_s+00") - (top["pop_D_s+00"] * top.nonsewer_frac).sum() / 1e6 + (top[f"pop_D_{tag}"] * top.nonsewer_frac).sum() / 1e6
    totB = S("pop_B_s+00") - (top["pop_B_s+00"] * top.nonsewer_frac).sum() / 1e6 + (top[f"pop_B_{tag}"] * top.nonsewer_frac).sum() / 1e6
    extra.append(dict(land_shift_m=f"20 most exposed cities all shifted {sh:+.1f} m, others central", wet_ground_M=round(tot, 1),
                      surface_flood_M=round(totB, 1), groundwater_only_M=round(tot - totB, 1)))
# rank stability of top 20 under global shifts
cen = (v["pop_D_s+00"] * f)
for tag in ("s-12", "s+12", "s-04", "s+04"):
    alt = v[f"pop_D_{tag}"] * f
    r = cen.rank().corr(alt.rank())
    o = len(set(cen.nlargest(20).index) & set(alt.nlargest(20).index))
    extra.append(dict(land_shift_m=f"rank check {tag}", spearman=round(r, 3), top20_overlap=o))
t1 = pd.concat([t1, pd.DataFrame(extra)])
t1.to_csv("supplementary/Table_S26_datum_robustness.csv", index=False)
print(t1.to_string())
# 2 open coast
est = v.estuarine.astype(bool)
r2 = []
for nm, D, B, Z, m in (("Full atlas (10 km zone)", "pop_D_s+00", "pop_B_s+00", None, None),
                       ("Within 5 km of the coastline", "pop_D_5km", "pop_B_5km", None, None),
                       ("Open coast within 5 km (<= 40 km from generalised coastline)", "pop_D_oc5", "pop_B_oc5", "pop_zone_oc5", None),
                       ("Open coast within 5 km, estuary and delta cities removed", "pop_D_oc5", "pop_B_oc5", "pop_zone_oc5", ~est)):
    d_, b_ = S(D, m), S(B, m)
    r2.append(dict(subset=nm, wet_ground_M=round(d_, 1), surface_flood_M=round(b_, 1), groundwater_only_M=round(d_ - b_, 1),
                   groundwater_only_share_pct=round(100 * (d_ - b_) / d_, 1), wet_share_of_zone_pct=round(100 * d_ / S(Z, m), 1) if Z else np.nan))
t2 = pd.DataFrame(r2); t2.to_csv("supplementary/Table_S27_open_coast_subset.csv", index=False); print(t2.to_string())
# 3 sea level 2020
sl = pd.read_csv("city_sl2020.csv").set_index("ID_UC_G0")
w = lambda col: np.average(sl[col], weights=v.loc[sl.index, "ns_wet10_now"])
now20, now = S("pop_D_now20"), S("pop_D_s+00")
p50 = lambda c: (v[c] * f).sum() / 1e6
r3 = [dict(quantity="Present day, JRC baseline (1980-2014)", value=round(now, 1)),
      dict(quantity="Present day, sea level and tide moved to 2020 (city-specific)", value=round(now20, 1)),
      dict(quantity="Surface flooding only (2020 referenced)", value=round(S("pop_B_now20"), 1)),
      dict(quantity="Groundwater contact only (2020 referenced)", value=round(now20 - S("pop_B_now20"), 1)),
      dict(quantity="2050 RCP8.5, 2050 population, tidal amplitude constant", value=round(p50("pop50_D_85"), 1)),
      dict(quantity="2050 RCP8.5, 2050 population, tidal amplitude change included", value=round(p50("pop50_D_85t"), 1)),
      dict(quantity="Change 2020 to 2050 (from 2020-referenced present)", value=round(p50("pop50_D_85t") - now20, 1)),
      dict(quantity="  population growth (2050 population, 2020 sea level)", value=round(p50("pop50_D_now20") - now20, 1)),
      dict(quantity="  relative sea-level rise and tide 2020 to 2050 (2020 population)", value=round(S("pop_D_85t") - now20, 1)),
      dict(quantity="  interaction", value=round(p50("pop50_D_85t") - p50("pop50_D_now20") - S("pop_D_85t") + now20, 1)),
      dict(quantity="Exposure-weighted relative sea-level rise baseline to 2020 (m)", value=round(w("r20"), 3)),
      dict(quantity="Range of city relative sea-level rise baseline to 2020 (m)", value=f"{sl.r20.min():.3f} to {sl.r20.max():.3f}"),
      dict(quantity="Exposure-weighted relative sea-level rise 2020 to 2050 (m)", value=round(w("rslr_85_2050") - w("r20"), 3)),
      dict(quantity="Exposure-weighted change in highest astronomical tide to 2050 (m)", value=round(w("dhat50"), 3)),
      dict(quantity="Cities with tidal amplitude change > 0.1 m in absolute value", value=int((sl.dhat50.abs() > 0.1).sum()))]
t3 = pd.DataFrame(r3); t3.to_csv("supplementary/Table_S28_sea_level_2020.csv", index=False); print(t3.to_string())
