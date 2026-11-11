"""Urban sewer coverage trend per country (JMP 2015-2024 linear slope, declines set to zero)."""
import openpyxl
import numpy as np
import pandas as pd

URBAN = ("Urban", "Urbain", "Urbano", "الحضري", "Город")
eng = list(openpyxl.load_workbook("/home/claude/data/jmp/NGA.xlsx", read_only=True, data_only=True)["Estimates"].iter_rows(values_only=True))
H = [" ".join(str(h).split()) if h else "" for h in eng[3]]
JS = H.index("Sewer connection")


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


rec = []
for iso in pd.read_csv("/home/claude/atlas/country_table.csv").iso3:
    rows = list(openpyxl.load_workbook(f"/home/claude/data/jmp/{iso}.xlsx", read_only=True, data_only=True)["Estimates"].iter_rows(values_only=True))
    s = {int(num(r[1])): num(r[JS]) for r in rows[4:] if str(r[2]).strip() in URBAN and not np.isnan(num(r[1])) and 2015 <= num(r[1]) <= 2024}
    yrs = sorted(y for y, v in s.items() if not np.isnan(v))
    if len(yrs) >= 3:
        rec.append((iso, s[yrs[-1]], np.polyfit(yrs, [s[y] for y in yrs], 1)[0]))
t = pd.DataFrame(rec, columns=["iso3", "sewer_last", "slope_pp_per_yr"])
t["slope_used"] = t.slope_pp_per_yr.clip(lower=0)
t.to_csv("/home/claude/atlas/sewer_trend.csv", index=False)
print(len(t), "countries with a trend")
