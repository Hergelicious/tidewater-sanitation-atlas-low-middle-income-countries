"""Supplementary Table 24 (uncertainty by source, one factor at a time), 25 (like-for-like flood comparison) and
sanitation-technology scenarios (appended to Table 14c)."""
import pandas as pd
d = pd.read_csv("atlas_cities.csv"); M = 1e6
S = lambda c: round(d[c].sum() / M, 1)  # noqa: E731
tech = pd.read_csv("supplementary/Table_S14_technology_adaptation.csv")
sep = float(tech[(tech.group == "Technology") & (tech.item == "septic")].value.iloc[0])
odf = float(tech[(tech.group == "Technology") & (tech.item == "open defecation")].value.iloc[0])  # no facility: not soil-dependent
fan = pd.read_csv("supplementary/Table_S12_fan_comparison.csv").set_index("quantity").nonsewered_millions
city = pd.read_csv("supplementary/Table_S19b_datum_city_exposure.csv")
c = S("ns_wet10_now")
rows = [("Central estimate", c, c),
        ("Elevation error, sea-level percentiles and groundwater threshold combined (assumption range)", S("ns_wet10_low"), S("ns_wet10_high")),
        ("Datum: no sea-level referencing, or DTU22MDT offset +/-0.4 m", S("ns_wet10_egm"), S("ns_wet10_dm04")),
        ("Datum: test cities with the Seeger-Minderhoud product (% change)", round(city.change_pct.min(), 1), round(city.change_pct.max(), 1)),
        ("Present-day sea level moved from the 1980-2014 baseline to 2020 (+0.08 m)", c, S("ns_wet10_sl08")),
        ("Groundwater threshold HAT + 1 m to HAT + 5 m", S("ns_wet10_gw1"), S("ns_wet10_gw5")),
        ("Groundwater model instead of tidal rule (Fan et al., 1 m to 3 m)", float(fan["Wet ground with Fan 1 m in place of tidal rule"]), float(fan["Wet ground with Fan 3 m in place of tidal rule"])),
        ("Coastal zone 5 km to 20 km", S("ns_wet5km"), S("ns_wet20km")),
        ("Coastal flood layer: removed, or hydrodynamic map", S("ns_def_now"), S("ns_wet10_surgemodel")),
        ("Sanitation shares 2024 instead of 2020", S("ns24_wet10_now"), c),
        ("Soil-dependent users: no septic tank soil-dependent, or all septic tanks soil-dependent (no-facility users excluded)", round(c - sep - odf, 1), round(c - odf, 1)),
        ("Soil-dependent users: half of septic tanks soil-dependent (no-facility users excluded)", round(c - sep / 2 - odf, 1), round(c - sep / 2 - odf, 1))]
t = pd.DataFrame(rows, columns=["source", "low", "high"])
# v0.6 review additions (from summarise_v07.py outputs)
_dr = pd.read_csv("supplementary/Table_S26_datum_robustness.csv")
_g = lambda s: float(_dr[_dr.land_shift_m.astype(str) == s].wet_ground_M.iloc[0])
_sl = pd.read_csv("supplementary/Table_S28_sea_level_2020.csv").set_index("quantity").value
_add = pd.DataFrame([("Datum: offset error of 1.2 m (largest test-tile discrepancy) applied to all cities", _g("1.2"), _g("-1.2")),
                     ("Present-day sea level and tide moved to 2020, city-specific", float(_sl.iloc[1]), float(_sl.iloc[1]))], columns=t.columns)
t = pd.concat([t.iloc[:4], _add, t.iloc[4:]], ignore_index=True)
_mc = pd.read_csv("supplementary/Table_S31_joint_uncertainty.csv").set_index("quantity")
t = pd.concat([t, pd.DataFrame([("Soil-dependent residents, septic discharge from SFD reports (joint 90% interval)", _mc.loc["soil_dependent", "p05"], _mc.loc["soil_dependent", "p95"]),
                                ("Joint uncertainty of exposure (90% interval, Supplementary Table 31)", _mc.loc["exposure", "p05"], _mc.loc["exposure", "p95"])], columns=t.columns)], ignore_index=True)
print(t.to_string(index=False)); t.to_csv("supplementary/Table_S24_uncertainty_by_source.csv", index=False)
lf = pd.DataFrame([("A. Coastal inundation only (same DEM, bathtub)", S("ns_coast_now")),
                   ("B. Coastal and river flooding", S("ns_cr_now")),
                   ("C. Potential groundwater contact only", S("ns_gw3_now")),
                   ("D. All three mechanisms", c),
                   ("D minus B: groundwater contact without surface flooding", round((d["ns_wet10_now"].sum() - d["ns_cr_now"].sum()) / M, 1)),
                   ("E. Published 100-year storm-surge map", S("ns_vous_10")),
                   ("E within D (surge-map population also on wet ground)", S("ns_vous_in_wet")),
                   ("Share of E inside D (%)", round(100 * S("ns_vous_in_wet") / S("ns_vous_10"), 1)),
                   ("Share of D captured by E (%)", round(100 * S("ns_vous_in_wet") / c, 1))], columns=["quantity", "value"])
print(lf.to_string(index=False)); lf.to_csv("supplementary/Table_S25_like_for_like.csv", index=False)
