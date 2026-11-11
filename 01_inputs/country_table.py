"""Country-level covariates: JMP urban sanitation (2024), fish share of animal protein (FAOSTAT via
gcamfaostat, 2018-2022 mean), WASH-attributable mortality and under-5 mortality (World Bank WDI)."""
import json
import urllib.request
import numpy as np
import openpyxl
import pandas as pd

ATL = "/home/claude/atlas"
iso = pd.read_csv(f"{ATL}/country_iso3.csv")
eng = list(openpyxl.load_workbook("/home/claude/data/jmp/NGA.xlsx", read_only=True, data_only=True)
           ["Estimates"].iter_rows(values_only=True))
HDR = [" ".join(str(h).split()) if h else "" for h in eng[3]]
KEYS = {"sewer": "Sewer connection", "septic": "Septic tanks", "latrine_other": "Latrines and other",
        "unimproved": "Unimproved sanitation", "open_def": "Open defecation", "improved": "Improved sanitation"}
URBAN = ("Urban", "Urbain", "Urbano", "الحضري", "Город")
TOTAL = ("Total", "المجموع", "общий")


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def jmp(c):
    """Latest urban estimate with a sewer value; falls back to the national total when the
    country file has no urban disaggregation (flagged in jmp_setting)."""
    rows = list(openpyxl.load_workbook(f"/home/claude/data/jmp/{c}.xlsx", read_only=True, data_only=True)
                ["Estimates"].iter_rows(values_only=True))
    for labels, setting in ((URBAN, "urban"), (TOTAL, "national")):
        for year in range(2024, 2014, -1):  # latest year with a sewer estimate
            r = [r for r in rows[4:] if num(r[1]) == year and str(r[2]).strip() in labels]
            if r and not np.isnan(num(r[0][HDR.index("Sewer connection")])):
                out = {f"jmp_{k}": num(r[0][HDR.index(v)]) for k, v in KEYS.items()}
                out["jmp_year"], out["jmp_setting"] = year, setting
                return out
    return {}


rec = []
for _, r in iso.iterrows():
    d = {"country": r.country, "iso3": r.iso3}
    d.update(jmp(r.iso3))
    rec.append(d)
df = pd.DataFrame(rec)

# fish share of animal protein
f = pd.read_csv("/home/claude/data/food/fbs.csv.gz", comment="#")
f = f[f.element == "Protein"]
animal = ["Bovine meat and products", "Eggs and products", "Fish and products", "Milk and products",
          "Mutton, goat meat and products", "Offals and products", "Other meat and products",
          "Pigmeat, fats and products", "Poultry meat, fats and products"]
f = f[f["item"].isin(animal)].copy()
f["v"] = f[["2018", "2019", "2020", "2021", "2022"]].mean(axis=1)
p = f.pivot_table(index="iso", columns="item", values="v", aggfunc="sum")
share = (p["Fish and products"] / p.sum(axis=1) * 100).rename("fish_share_animal_protein_pct")
share.index = share.index.str.upper()
df = df.merge(share, left_on="iso3", right_index=True, how="left")


def wdi(code):
    url = f"https://api.worldbank.org/v2/country/all/indicator/{code}?format=json&per_page=20000&date=2015:2025"
    data = json.load(urllib.request.urlopen(url, timeout=120))[1]
    s = pd.DataFrame([(d["countryiso3code"], int(d["date"]), d["value"]) for d in data if d["value"] is not None],
                     columns=["iso3", "year", "v"])
    return s.sort_values("year").groupby("iso3").last()["v"]


df["wash_mortality_per100k"] = df.iso3.map(wdi("SH.STA.WASH.P5"))
df["u5_mortality_per1000"] = df.iso3.map(wdi("SH.DYN.MORT"))
df.to_csv(f"{ATL}/country_table.csv", index=False)
print(df.describe().T.round(1))
print(df[df.isna().any(axis=1)][["iso3", "jmp_sewer", "fish_share_animal_protein_pct", "wash_mortality_per100k"]])
