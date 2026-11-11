"""Household Recode analyses (aggregates only).
1. All-household metro sanitation versus JMP (residents: households weighted by size), Supplementary Table 6 (replaces KR version).
2. Exposure gradient: residents' wet-ground share within 2 km by urban wealth quintile (hv270a) and by toilet type,
   within cities (city fixed effects), Supplementary Table 16."""
import glob, numpy as np, pandas as pd, openpyxl
import statsmodels.formula.api as smf

P = "/home/claude/dhs_private"
V = ["hv001", "hv002", "hv005", "hv009", "hv024", "hv025", "hv205", "hv270", "hv270a", "hv201", "hv007"]
hr = []
for f in sorted(glob.glob(f"{P}/hr/*/*.[dD][tT][aA]")):
    with pd.io.stata.StataReader(f) as r:
        cols = V + (["shstate"] if "shstate" in r.variable_labels() else [])
    x = pd.read_stata(f, columns=cols, convert_categoricals=False)
    if "shstate" in cols:
        rr = pd.io.stata.StataReader(f)
        lab = rr.value_labels()
        sl = lab[rr._lbllist[list(rr.variable_labels().keys()).index("shstate")]]
        code = [int(k) for k, v in sl.items() if "lagos" in str(v).lower()][0]
        x["lagos"] = x.shstate == code
    hr.append(x.assign(survey=f.split("/")[-2], cc=f.split("/")[-2][:2]))
hr = pd.concat(hr, ignore_index=True)
G = {11: "sewer", 51: "sewer", 12: "septic", 52: "septic", 16: "septic", 13: "pit", 20: "pit", 21: "pit", 22: "pit", 23: "pit",
     24: "pit", 30: "none", 31: "none", 14: "other", 15: "other", 41: "other", 42: "other", 43: "other", 53: "other", 54: "other", 96: "other"}
hr["toilet"] = hr.hv205.map(G)
hr["w"] = hr.hv005 / 1e6 * hr.hv009
u = hr[(hr.hv025 == 1) & hr.toilet.notna()]
METRO = {"BD": ("Chattogram division", lambda h: h.hv024 == 2), "CI": ("Abidjan", lambda h: h.hv024 == 1),
         "GH": ("Greater Accra", lambda h: h.hv024 == 3), "HT": ("Aire métropolitaine", lambda h: h.hv024 == 0),
         "MZ": ("Cidade de Maputo", lambda h: h.hv024 == 11), "NG": ("Lagos State", lambda h: h.lagos.fillna(False).astype(bool)),
         "PH": ("National Capital Region", lambda h: h.hv024 == 13), "TZ": ("Dar es Salaam", lambda h: h.hv024 == 7)}
ISO = {"BD": "BGD", "CI": "CIV", "GH": "GHA", "HT": "HTI", "MZ": "MOZ", "NG": "NGA", "PH": "PHL", "TZ": "TZA"}
CATS = ["sewer", "septic", "pit", "other", "none"]
eng = list(openpyxl.load_workbook("/home/claude/data/jmp/NGA.xlsx", read_only=True, data_only=True)["Estimates"].iter_rows(values_only=True))
hd = [" ".join(str(h).split()) if h else "" for h in eng[3]]
rows = []
for cc, (name, sel) in METRO.items():
    h = u[u.cc == cc]
    year = int(h.hv007.mode().iloc[0])
    for lab, sub in ((f"DHS urban, {name}", h[sel(h)]), ("DHS urban, national", h)):
        s = sub.groupby("toilet").w.sum().reindex(CATS, fill_value=0)
        s = 100 * s / s.sum()
        rows.append(dict(iso3=ISO[cc], survey_year=year, source=lab, households=len(sub), **s.round(1).to_dict(), nonsewer=round(100 - s.sewer, 1)))
    rws = list(openpyxl.load_workbook(f"/home/claude/data/jmp/{ISO[cc]}.xlsx", read_only=True, data_only=True)["Estimates"].iter_rows(values_only=True))
    r = [x for x in rws[4:] if str(x[1]).strip().split(".")[0] == str(year) and str(x[2]).strip() in ("Urban", "Urbain", "Urbano")][0]
    js = float(r[hd.index("Sewer connection")])
    rows.append(dict(iso3=ISO[cc], survey_year=year, source="JMP urban, national, survey year", sewer=js,
                     septic=float(r[hd.index("Septic tanks")]), nonsewer=round(100 - js, 1)))
t = pd.DataFrame(rows)
pd.set_option("display.width", 220)
print(t.to_string(index=False))
t.to_csv("/home/claude/atlas/supplementary/Table_S6_dhs_metro_sanitation.csv", index=False)

# 2. exposure gradient within cities
cw = pd.read_csv(f"{P}/cluster_wet.csv")
cw["cc"] = cw.survey.str[:2]
cw = cw[cw.in_zone]
e = u.merge(cw, left_on=["cc", "hv001"], right_on=["cc", "DHSCLUST"], how="inner")
e["clu"] = e.cc + "_" + e.hv001.astype(str)
e["city"] = e.ID_UC_G0.astype(str)
e["q"] = e.hv270a.fillna(e.hv270)
print("households", len(e), "clusters", e.clu.nunique(), "cities", e.city.nunique())
out = []
for var, ref in (("q", 5), ("toilet", "sewer")):
    m = smf.wls(f"wet_2km ~ C({var}, Treatment({ref!r} if isinstance({ref!r}, str) else {ref})) + C(city)".replace(f"Treatment({ref!r} if isinstance({ref!r}, str) else {ref})", f"Treatment({ref!r})"),
                data=e, weights=e.w).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(e.clu)[0]})
    raw = e.groupby(var).apply(lambda x: np.average(x.wet_2km, weights=x.w))
    for lvl in raw.index:
        key = [k for k in m.params.index if k.startswith(f"C({var}") and k.endswith(f"[T.{lvl}]")]
        b = m.params[key[0]] if key else 0.0
        se = m.bse[key[0]] if key else 0.0
        out.append(dict(group=var.replace("q", "urban wealth quintile"), level=lvl, raw_wet_share_pct=100 * raw[lvl],
                        within_city_diff_vs_ref_pp=100 * b, ci_lo=100 * (b - 1.96 * se), ci_hi=100 * (b + 1.96 * se), reference=str(ref)))
o = pd.DataFrame(out).round(1)
print(o.to_string(index=False))
o.to_csv("/home/claude/atlas/supplementary/Table_S16_dhs_exposure_gradient.csv", index=False)
