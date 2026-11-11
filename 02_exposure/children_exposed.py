"""Children under five among non-sewered residents on wet ground.
National share of the population aged 0 to 4 (World Bank WDI, latest year, from UN World Population Prospects)
applied to each city's non-sewered exposed population. Urban age structures are usually slightly older, so this is an upper-leaning approximation."""
import json
import urllib.request
import pandas as pd

def wdi(ind):
    url = f"https://api.worldbank.org/v2/country/all/indicator/{ind}?format=json&mrnev=1&per_page=20000"
    js = json.load(urllib.request.urlopen(url, timeout=120))[1]
    return pd.Series({r["countryiso3code"]: r["value"] for r in js if r["countryiso3code"]}, name=ind)

f, m, fem = wdi("SP.POP.0004.FE.5Y"), wdi("SP.POP.0004.MA.5Y"), wdi("SP.POP.TOTL.FE.ZS")
u5 = ((f * fem + m * (100 - fem)) / 100).rename("u5_share_pct")
d = pd.read_csv("atlas_cities.csv").merge(u5, left_on="iso3", right_index=True, how="left")
miss = d[d.u5_share_pct.isna()].iso3.unique()
print("Countries without WDI age data:", list(miss))
d["u5_share_pct"] = d.u5_share_pct.fillna(d.groupby("GC_DEV_USR_2025").u5_share_pct.transform("median"))
for c in ["ns_wet10_now", "ns_gw3_now", "ns_wet10_low", "ns_wet10_high", "ns_vous_10"]:
    d["u5_" + c] = d[c] * d.u5_share_pct / 100
tot = d[[c for c in d.columns if c.startswith("u5_ns")]].sum() / 1e6
print(tot.round(2))
reg = (d.groupby("GC_DEV_USR_2025")[["u5_ns_wet10_now"]].sum() / 1e6).round(2).sort_values("u5_ns_wet10_now", ascending=False)
print(reg)
d[["ID_UC_G0", "u5_share_pct", "u5_ns_wet10_now"]].to_csv("city_children.csv", index=False)
