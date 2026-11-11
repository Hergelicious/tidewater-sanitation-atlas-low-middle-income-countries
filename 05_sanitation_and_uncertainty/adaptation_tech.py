"""Sanitation technology split and a mechanism-matched adaptation scenario (coverage, not cost).
Technology: JMP 2020 national urban shares of septic tanks, latrines and other improved, unimproved, open defecation,
rescaled to the non-sewered share. Writes supplementary/Table_S14_technology_adaptation.csv."""
import numpy as np
import pandas as pd

d = pd.read_csv("atlas_cities.csv")
M = 1e6
T = d.ns_wet10_now.sum() / M
cats = {"septic": "jmp20_septic", "latrines_other_improved": "jmp20_latrine_other",
        "unimproved": "jmp20_unimproved", "open_defecation": "jmp20_open_def"}
sh = d[list(cats.values())].fillna(np.nan)
tot = sh.sum(1)
rows = []
for k, c in cats.items():
    frac = (d[c] / tot).where(tot > 0)
    frac = frac.fillna(frac.groupby(d.GC_DEV_USR_2025).transform("median"))
    rows.append(("Technology", k.replace("_", " "), (d.ns_wet10_now * frac).sum() / M))
    rows.append(("Technology, potential groundwater contact", k.replace("_", " "), (d.ns_gw3_now * frac).sum() / M))
s = d.ns_wet10_now.sort_values(ascending=False).cumsum() / d.ns_wet10_now.sum()
n50, n75 = int((s < 0.5).sum() + 1), int((s < 0.75).sum() + 1)
rows += [("Mechanism", "All non-sewered residents on wet ground", T),
         ("Mechanism", "Reached only by the 100-year coastal flood (coastal flood protection alone)", T - d.ns_def_now.sum() / M),
         ("Mechanism", "Reached only by 100-year coastal or river floods (all flood protection)", T - d.ns_gw3_now.sum() / M),
         ("Mechanism", "On potential groundwater contact ground (raised or sealed containment, container-based or sewered)", d.ns_gw3_now.sum() / M),
         ("Targeting", "Top 20 cities", d.ns_wet10_now.nlargest(20).sum() / M),
         ("Targeting", "Top 50 cities", d.ns_wet10_now.nlargest(50).sum() / M),
         ("Targeting", "Cities needed for half of the exposure", n50),
         ("Targeting", "Cities needed for three quarters of the exposure", n75),
         ("2050", "Constant sewer coverage", d.ns50_wet10_85.sum() / M),
         ("2050", "Sewer trend 2015 to 2024 continued", d.ns50t_wet10_85.sum() / M),
         ("2050", "Wet ground in the 50 most exposed cities (2050) served by sewer or sealed containment",
          d.ns50_wet10_85.sum() / M - d.ns50_wet10_85.nlargest(50).sum() / M)]
t = pd.DataFrame(rows, columns=["group", "item", "value"])
t["value"] = t.value.round(1)
pd.set_option("display.width", 220, "display.max_colwidth", 120)
print(t.to_string(index=False))
t.to_csv("supplementary/Table_S14_technology_adaptation.csv", index=False)

# Targeted versus distributed upgrades (coverage only): N people moved to sewer or sealed, emptied containment.
ns_zone = d.ns_within10km.clip(lower=0)
expo = d.ns_wet10_now
top50 = expo.nlargest(50).sum()
rows2 = []
for N in (10e6, 25e6, 50e6):
    distributed = N * expo.sum() / ns_zone.sum()                    # spread evenly over all non-sewered coastal residents
    within = np.minimum(N * ns_zone / ns_zone.sum(), expo).sum()     # each city's share, wet ground first
    targeted = min(N, top50)                                         # wet ground of the 50 most exposed cities first
    ranked = min(N, expo.sum())                                      # wet ground anywhere, most exposed cities first
    rows2.append(dict(upgrades_millions=N / 1e6, distributed_evenly=distributed / 1e6, wet_ground_first_within_each_city=within / 1e6,
                      top50_cities_wet_ground=targeted / 1e6))
t2 = pd.DataFrame(rows2).round(1)
print(t2.to_string(index=False))
t2.to_csv("supplementary/Table_S14b_targeting.csv", index=False)
