"""Why the exposure, and the gap with storm-surge maps, differ between regions.
Exposure per coastal-zone resident = (share of zone residents on wet ground) x (non-sewered share among them).
Surge maps capture little where storm surge above the highest astronomical tide is small but land lies within a few metres of it.
Writes supplementary/Table_S9_regional_drivers.csv."""
import numpy as np
import pandas as pd

d = pd.read_csv("atlas_cities.csv")
d["surge_m"] = (d.esl_base - d.hat_base).clip(lower=0)
w = d.pop_within10km


def agg(x):
    ww = x.pop_within10km
    return pd.Series({
        "cities": len(x), "zone_pop_M": ww.sum() / 1e6,
        "wet_share_pct": 100 * x.pop_wet10_now.sum() / ww.sum(),
        "nonsewer_among_wet_pct": 100 * x.ns_wet10_now.sum() / x.pop_wet10_now.sum(),
        "exposed_per_100_residents": 100 * x.ns_wet10_now.sum() / ww.sum(),
        "groundwater_contact_share_pct": 100 * x.ns_gw3_now.sum() / x.ns_wet10_now.sum(),
        "river_only_share_pct": 100 * (x.ns_wet10_now - x.ns_tidal_now).clip(lower=0).sum() / x.ns_wet10_now.sum(),
        "surge_above_HAT_m": np.average(x.surge_m, weights=ww) if ww.sum() > 0 else np.nan,
        "HAT_m": np.average(x.hat_base, weights=ww) if ww.sum() > 0 else np.nan,
        "surge_map_capture_pct": 100 * x.ns_vous_10.sum() / x.ns_wet10_now.sum(),
        "nonsewered_exposed_M": x.ns_wet10_now.sum() / 1e6})


reg = d.groupby("GC_DEV_USR_2025").apply(agg)
reg.loc["All"] = agg(d)
reg = reg[reg.nonsewered_exposed_M >= 0.5].sort_values("nonsewered_exposed_M", ascending=False)
pd.set_option("display.width", 250)
print(reg.round(2).to_string())
# Log decomposition of exposure per resident relative to all cities.
g = reg.loc["All"]
dec = pd.DataFrame({"wet_share": np.log(reg.wet_share_pct / g.wet_share_pct),
                    "nonsewer": np.log(reg.nonsewer_among_wet_pct / g.nonsewer_among_wet_pct)})
dec["total"] = dec.sum(1)
print((100 * dec).round(0))
# City level: surge capture against surge height.
c = d[d.ns_wet10_now >= 1e4].copy()
c["capture"] = c.ns_vous_10 / c.ns_wet10_now
c["bin"] = pd.cut(c.surge_m, [0, 0.5, 1, 1.5, 2, 3, 10])
print(c.groupby("bin", observed=True).apply(lambda x: pd.Series({"cities": len(x), "capture_pct": 100 * x.ns_vous_10.sum() / x.ns_wet10_now.sum()})).round(1))
reg.round(3).to_csv("supplementary/Table_S9_regional_drivers.csv")
