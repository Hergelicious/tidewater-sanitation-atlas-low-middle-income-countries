"""Summary of the Fan et al. comparison: totals, agreement, regions and city ranks. Writes supplementary/Table_S12_fan_comparison.csv."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

d = pd.read_csv("atlas_cities.csv").merge(pd.read_csv("city_fan_compare.csv"), on="ID_UC_G0")
f, M = d.nonsewer_frac, 1e6
for c in ["pop_fan1", "pop_fan3", "pop_fan5", "pop_wetfan1", "pop_wetfan3", "pop_wetfan5", "pop_both3", "pop_gw3_only",
          "pop_fan3_only", "pop_neither", "pop_wet_tidal_or_fan3", "pop_gw3_cov", "pop_fan_cover"]:
    d["ns" + c[3:]] = d[c] * f
S = lambda c, x=d: x[c].sum() / M  # noqa: E731
rows = [("Tidal rule, groundwater contact (HAT + 3 m)", S("ns_gw3_now")),
        ("Fan model, water table within 1 m", S("ns_fan1")), ("Fan model, within 3 m", S("ns_fan3")),
        ("Fan model, within 5 m", S("ns_fan5")),
        ("Both methods agree (contact)", S("ns_both3")), ("Tidal rule only", S("ns_gw3_only")),
        ("Fan model only", S("ns_fan3_only")),
        ("Wet ground, central (tidal rule)", S("ns_wet10_now")),
        ("Wet ground with Fan 1 m in place of tidal rule", S("ns_wetfan1")),
        ("Wet ground with Fan 3 m in place of tidal rule", S("ns_wetfan3")),
        ("Wet ground with Fan 5 m in place of tidal rule", S("ns_wetfan5")),
        ("Wet ground, tidal rule or Fan 3 m", S("ns_wet_tidal_or_fan3"))]
t = pd.DataFrame(rows, columns=["quantity", "nonsewered_millions"]).round(1)
print(t.to_string(index=False))
print("Share of tidal-rule contact population also within 3 m in Fan:", round(d.pop_both3.sum() / d.pop_gw3_cov.sum(), 3))
print("Population coverage of Fan in zone:", round(d.pop_fan_cover.sum() / d.pop_zone.sum(), 3))
print("Zone population within 3 m in Fan:", round(d.pop_fan3.sum() / d.pop_fan_cover.sum(), 3), "vs tidal rule",
      round(d.pop_gw3_cov.sum() / d.pop_fan_cover.sum(), 3))
reg = d.groupby("GC_DEV_USR_2025")[["ns_wet10_now", "ns_wetfan1", "ns_wetfan3", "ns_gw3_now", "ns_both3"]].sum() / M
reg["agree_share_of_rule"] = reg.ns_both3 / reg.ns_gw3_now
print(reg[reg.ns_wet10_now > 0.5].sort_values("ns_wet10_now", ascending=False).round(2))
for c in ["ns_wetfan1", "ns_wetfan3", "ns_wetfan5"]:
    o20 = len(set(d.ns_wet10_now.nlargest(20).index) & set(d[c].nlargest(20).index))
    o50 = len(set(d.ns_wet10_now.nlargest(50).index) & set(d[c].nlargest(50).index))
    print(c, "spearman", round(spearmanr(d.ns_wet10_now, d[c]).statistic, 3), "top20", o20, "top50", o50)
t.to_csv("supplementary/Table_S12_fan_comparison.csv", index=False)
