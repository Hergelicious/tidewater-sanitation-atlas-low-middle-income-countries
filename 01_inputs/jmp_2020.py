"""JMP urban sanitation shares for 2020, to match the 2020 population baseline.
Same rules as country_table.py (urban row, national fallback), but the year closest to 2020 (2020, then 2019, 2021, ...).
Writes jmp_2020.csv."""
import numpy as np
import openpyxl
import pandas as pd

ATL = "/home/claude/atlas"
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



rec = []
for c in pd.read_csv("/home/claude/atlas/country_table.csv").iso3:
    rows = list(openpyxl.load_workbook(f"/home/claude/data/jmp/{c}.xlsx", read_only=True, data_only=True)
                ["Estimates"].iter_rows(values_only=True))
    out = {"iso3": c}
    for labels, setting in ((URBAN, "urban"), (TOTAL, "national")):
        for year in (2020, 2019, 2021, 2018, 2022, 2017, 2023, 2016, 2024, 2015):
            r = [r for r in rows[4:] if num(r[1]) == year and str(r[2]).strip() in labels]
            if r and not np.isnan(num(r[0][HDR.index("Sewer connection")])):
                out.update({f"jmp20_{k}": num(r[0][HDR.index(v)]) for k, v in KEYS.items()})
                out["jmp20_year"], out["jmp20_setting"] = year, setting
                break
        if "jmp20_year" in out:
            break
    rec.append(out)
t = pd.DataFrame(rec)
t.to_csv("/home/claude/atlas/jmp_2020.csv", index=False)
print(t.jmp20_year.value_counts().to_dict(), t.jmp20_setting.value_counts().to_dict())
