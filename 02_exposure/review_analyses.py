"""Analyses requested by the external review.
1. Persistent groundwater contact versus episodic flooding only.
2. Rank stability of priority cities across assumption scenarios.
3. Per-city share of non-sewered residents on wet ground.
Writes supplementary/Table_S4_rank_stability.csv and supplementary/Table_S5_persistent_episodic.csv."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

d = pd.read_csv("atlas_cities.csv")
M = 1e6

# 1. Persistent versus episodic. gw3 lies inside the wet union, so the remainder is flood only.
d["ns_persistent"] = d.ns_gw3_now
d["ns_episodic_only"] = (d.ns_wet10_now - d.ns_gw3_now).clip(lower=0)
tot = d[["ns_wet10_now", "ns_persistent", "ns_episodic_only", "ns_gw5_now", "ns_gw1_now"]].sum() / M
print("Totals (M):\n", tot.round(1))
reg = d.groupby("GC_DEV_USR_2025")[["ns_wet10_now", "ns_persistent", "ns_episodic_only"]].sum() / M
reg["persistent_share"] = reg.ns_persistent / reg.ns_wet10_now
reg = reg.sort_values("ns_wet10_now", ascending=False)
print(reg.round(2))
reg.round(3).to_csv("supplementary/Table_S5_persistent_episodic.csv")

# 2. Rank stability.
scen = {"Central": "ns_wet10_now", "Low sea levels (p5)": "ns_wet10_low", "High sea levels (p95)": "ns_wet10_high",
        "Groundwater 1 m": "ns_wet10_gw1", "Groundwater 5 m": "ns_wet10_gw5", "Land 0.3 m lower": "ns_wet10_datum03", "Elevation on EGM2008 geoid": "ns_wet10_egm",
        "Zone 5 km": "ns_wet5km", "Zone 20 km": "ns_wet20km", "Hydrodynamic surge layer": "ns_wet10_surgemodel",
        "Tidal mechanisms only": "ns_tidal_now", "2050 RCP8.5, 2050 population": "ns50_wet10_85"}
base = d[scen["Central"]]
top20, top50 = set(base.nlargest(20).index), set(base.nlargest(50).index)
rows = []
for k, c in scen.items():
    s = d[c]
    rho = spearmanr(base, s).statistic
    nz = (base > 0) | (s > 0)
    rho_nz = spearmanr(base[nz], s[nz]).statistic
    o20 = len(top20 & set(s.nlargest(20).index))
    o50 = len(top50 & set(s.nlargest(50).index))
    rows.append(dict(scenario=k, total_M=s.sum() / M, spearman_all=rho, spearman_nonzero=rho_nz,
                     top20_overlap=o20, top50_overlap=o50))
rs = pd.DataFrame(rows)
print(rs.round(3).to_string(index=False))
rs.round(3).to_csv("supplementary/Table_S4_rank_stability.csv", index=False)

# Cities in the central top 50 that stay in the top 50 in every scenario.
always = set(top50)
for c in scen.values():
    always &= set(d[c].nlargest(50).index)
print("Cities in top 50 under every scenario:", len(always))

# 3. Per-city share of non-sewered residents exposed.
d["share_ns_exposed"] = np.where(d.ns_within10km > 0, d.ns_wet10_now / d.ns_within10km, np.nan)
big = d[d.ns_within10km >= 1e5]
print("Cities with >=100k non-sewered in zone:", len(big))
print("Median share exposed:", round(big.share_ns_exposed.median(), 3),
      "IQR", big.share_ns_exposed.quantile([.25, .75]).round(3).tolist())
print("Cities with share >= 50%:", int((big.share_ns_exposed >= 0.5).sum()))
nm = d.GC_UCN_MAI_2025.str.replace(r"\s*\[.*\]", "", regex=True)
top = big.assign(name=nm).nlargest(15, "share_ns_exposed")[["name", "iso3", "ns_within10km", "ns_wet10_now", "share_ns_exposed"]]
print(top.round(3).to_string(index=False))
d[["ID_UC_G0", "share_ns_exposed", "ns_persistent", "ns_episodic_only"]].to_csv("city_review_fields.csv", index=False)
