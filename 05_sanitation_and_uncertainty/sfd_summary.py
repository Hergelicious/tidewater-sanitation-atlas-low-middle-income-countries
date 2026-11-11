"""Share of septic-tank users whose tank discharges to soil (soak pit, infiltration, open-bottom tank), from city SFD reports.
Rules: soil = septic to soak pit + lined tank with open bottom; not soil = septic to drain or water body + sealed or fully
lined tank + septic to sewer or to open ground (Bangladesh SFD Lite 'C8'), + sealed tank to treatment (Nonthaburi).
Cities without an outlet split are excluded. Writes sfd_city_psoil.csv, sfd_psoil.json and Table_S32_sfd_septic_discharge.csv"""
import json
import numpy as np
import pandas as pd

d = pd.read_csv("/home/claude/data/sfd/sfd_containment.csv")
f = lambda c: d[c].fillna(0)
other_septic = np.where(d.iso3.eq("BGD") | d.atlas_name.isin(["Đà Nẵng", "Dhaka", "Bangkok"]), f("other_pct"), 0)
soil = f("septic_soakpit_pct") + f("lined_openbottom_pct")
nots = f("septic_drain_pct") + f("septic_sealed_pct") + other_septic
d["septic_total_pct"] = soil + nots
d["p_soil"] = np.where(d.septic_total_pct > 0, soil / d.septic_total_pct, np.nan)
d["p_drain"] = np.where(d.septic_total_pct > 0, f("septic_drain_pct") / d.septic_total_pct, np.nan)
d["convention"] = d.atlas_name.isin(["Chennai", "Lima"])          # tanks classed as sealed by assumption in the report
d.loc[d.atlas_name.isin(["Upenekang", "Kochi"]), "p_soil"] = np.nan  # outlet split not quantified
u = d[d.p_soil.notna()].copy()
c = u.groupby("iso3").p_soil.mean()
summ = dict(cities=int(len(u)), countries=int(c.size), median_city=float(u.p_soil.median()),
            mean_city=float(u.p_soil.mean()), mean_of_country_means=float(c.mean()),
            q25=float(u.p_soil.quantile(0.25)), q75=float(u.p_soil.quantile(0.75)), sd_city=float(u.p_soil.std()),
            mean_excl_convention=float(u[~u.convention].p_soil.mean()), median_drain=float(u.p_drain.median()))
json.dump(summ, open("sfd_psoil.json", "w"), indent=1)
u[["atlas_rank", "atlas_name", "iso3", "sfd_city", "sfd_year", "sfd_level", "p_soil", "p_drain", "convention", "url"]].to_csv("sfd_city_psoil.csv", index=False)
t = u[["atlas_name", "iso3", "sfd_city", "sfd_year", "sfd_level", "septic_total_pct", "p_soil", "p_drain", "convention", "url"]].copy()
t[["p_soil", "p_drain"]] = (100 * t[["p_soil", "p_drain"]]).round(0)
t.to_csv("supplementary/Table_S32_sfd_septic_discharge.csv", index=False)
print(t.drop(columns="url").to_string(index=False)); print(json.dumps(summ, indent=1))
