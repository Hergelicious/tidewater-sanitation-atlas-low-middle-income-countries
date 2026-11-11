"""Pre-specified within-city test: does wet ground around a cluster go with more child diarrhoea in households using
pit latrines or no facility than in sewered or septic households?
Linear probability model, survey weights, city fixed effects, standard errors clustered by DHS cluster.
Writes aggregated coefficients only: /home/claude/atlas/supplementary/Table_S15_dhs_health_models.csv"""
import glob, numpy as np, pandas as pd
import statsmodels.formula.api as smf

P = "/home/claude/dhs_private"
V = ["v001", "v002", "v005", "v006", "v024", "v025", "v106", "v113", "v116", "v190", "v190a", "b4", "b5", "b19", "h11"]
k = pd.concat([pd.read_stata(f, columns=V, convert_categoricals=False).assign(survey=f.split("/")[-2])
               for f in glob.glob(f"{P}/kr/**/*.[dD][tT][aA]", recursive=True)])
k["cc"] = k.survey.str[:2]
cw = pd.read_csv(f"{P}/cluster_wet.csv")
cw["cc"] = cw.survey.str[:2]
cw = cw[cw.in_zone]
d = k.merge(cw, left_on=["cc", "v001"], right_on=["cc", "DHSCLUST"], how="inner", suffixes=("", "_ge"))
d = d[(d.b5 == 1) & d.h11.isin([0, 1, 2]) & (d.v025 == 1)].copy()
d["diar"] = d.h11.isin([1, 2]).astype(float)
G = {11: "sewer", 51: "sewer", 12: "septic", 52: "septic", 16: "septic", 13: "pit", 20: "pit", 21: "pit", 22: "pit", 23: "pit",
     24: "pit", 30: "none", 31: "none"}
d["toilet"] = d.v116.map(G).fillna("other")
d["soil"] = d.toilet.isin(["pit", "none"]).astype(float)       # containment relies on soil or none
d["contained"] = d.toilet.isin(["sewer", "septic"])
d["age"] = pd.cut(d.b19, [-1, 5, 11, 23, 35, 59], labels=False)
d["water"] = np.select([d.v113.between(10, 14), d.v113.isin([20, 21, 30, 31, 32]), d.v113.isin([71, 72])], ["piped", "ground", "packaged"], "other")
d["w"] = d.v005 / 1e6
d["clu"] = d.cc + "_" + d.v001.astype(str)
d["city"] = d.ID_UC_G0.astype(str)
d["month"] = d.cc + "_" + d.v006.astype(int).astype(str)
d["v190a"] = d.v190a.fillna(d.v190)
print("children", len(d), "clusters", d.clu.nunique(), "cities", d.city.nunique(), d.cc.value_counts().to_dict())
print(d.groupby("toilet").apply(lambda x: np.average(x.diar, weights=x.w)).round(3).to_dict())
print("wet_2km by toilet:", d.groupby("toilet").wet_2km.mean().round(2).to_dict())

import os
base = "C(age) + C(b4) + C(v106) + C(v190a) + C(water) + C(city)" + (" + C(month)" if os.environ.get("MONTH", "1") == "1" else "")
rows = []


def fit(name, expo, data, extra_terms=""):
    dd = data.dropna(subset=[expo]).copy()
    dd = dd[dd.toilet != "other"]
    dd["E"] = dd[expo]
    m = smf.wls(f"diar ~ E * soil + C(toilet) + {base} {extra_terms}", data=dd, weights=dd.w).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(dd.clu)[0]})
    b, V = m.params, m.cov_params()
    e_cont = b["E"]; se_cont = np.sqrt(V.loc["E", "E"])
    e_soil = b["E"] + b["E:soil"]
    se_soil = np.sqrt(V.loc["E", "E"] + V.loc["E:soil", "E:soil"] + 2 * V.loc["E", "E:soil"])
    i, se_i = b["E:soil"], np.sqrt(V.loc["E:soil", "E:soil"])
    rows.append(dict(model=name, exposure=expo, children=len(dd), clusters=dd.clu.nunique(), cities=dd.city.nunique(),
                     wet_effect_sewer_septic_pp=100 * e_cont, ci_lo=100 * (e_cont - 1.96 * se_cont), ci_hi=100 * (e_cont + 1.96 * se_cont),
                     wet_effect_pit_none_pp=100 * e_soil, ci_lo_pit=100 * (e_soil - 1.96 * se_soil), ci_hi_pit=100 * (e_soil + 1.96 * se_soil),
                     interaction_pp=100 * i, ci_lo_int=100 * (i - 1.96 * se_i), ci_hi_int=100 * (i + 1.96 * se_i), p_interaction=m.pvalues["E:soil"]))


fit("Primary: wet ground share within 2 km", "wet_2km", d)
fit("Buffer 1 km", "wet_1km", d)
fit("Buffer 5 km", "wet_5km", d)
fit("Groundwater contact share within 2 km", "gw_2km", d)
d["wet2_bin"] = (d.wet_2km >= 0.5).astype(float)
fit("Wet ground share at least half (binary)", "wet2_bin", d)
fit("Excluding the Philippines", "wet_2km", d[d.cc != "PH"])
base = base.replace(" + C(month)", "")
fit("Excluding Bangladesh (no month effects)", "wet_2km", d[d.cc != "BD"])
fit("Without interview-month effects", "wet_2km", d)
m0 = smf.wls("diar ~ C(toilet) + C(age) + C(b4) + C(v106) + C(v190a) + C(water) + C(city)", data=d[d.toilet != "other"],
             weights=d.loc[d.toilet != "other", "w"]).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.loc[d.toilet != "other", "clu"])[0]})
print("Toilet differences vs none (pp):", (100 * m0.params.filter(like="C(toilet)")).round(1).to_dict(),
      (100 * 1.96 * m0.bse.filter(like="C(toilet)")).round(1).to_dict())
t = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(t.round(2).to_string(index=False))
t.round(3).to_csv("/home/claude/atlas/supplementary/Table_S15_dhs_health_models.csv", index=False)
d[["survey", "v001", "w", "diar", "toilet", "wet_2km"]].to_parquet(f"{P}/analysis_sample.parquet")
