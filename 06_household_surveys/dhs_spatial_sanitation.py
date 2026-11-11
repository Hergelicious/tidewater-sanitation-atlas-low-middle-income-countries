"""Does sanitation access differ between wetter and drier parts of the same cities? (aggregates only)
1. Linear probability models with city fixed effects: sewer, septic, pit, no facility ~ wet-ground share within 2 km.
2. Bias of the atlas assumption of uniform within-city sanitation shares: ratio of the exposure-weighted non-sewered
   share (households weighted by wet-ground share) to the city-average share, pooled over cities; cluster bootstrap.
Writes /home/claude/atlas/supplementary/Table_S35_dhs_spatial_sanitation.csv"""
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
src = open("/home/claude/atlas_dhs/hr_analyses.py").read()
exec(src[:src.index("METRO = ")])
P = "/home/claude/dhs_private"
cw = pd.read_csv(f"{P}/cluster_wet.csv"); cw["cc"] = cw.survey.str[:2]; cw = cw[cw.in_zone]
e = u.merge(cw, left_on=["cc", "hv001"], right_on=["cc", "DHSCLUST"], how="inner")
e["clu"] = e.cc + "_" + e.hv001.astype(str); e["city"] = e.ID_UC_G0.astype(str)
for k in ("sewer", "septic", "pit", "none"):
    e[k] = (e.toilet == k).astype(float)
e["nonsewer"] = 1 - e.sewer
e["pit_none"] = e.pit + e.none
rows = []
for y in ("sewer", "nonsewer", "septic", "pit", "none", "pit_none"):
    m = smf.wls(f"{y} ~ wet_2km + C(city)", data=e, weights=e.w).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(e.clu)[0]})
    b, ci = m.params["wet_2km"], m.conf_int().loc["wet_2km"]
    rows.append(dict(measure=f"{y}: change in share from no to all wet ground within 2 km, same city (percentage points)",
                     estimate=round(100 * b, 1), lo=round(100 * ci[0], 1), hi=round(100 * ci[1], 1)))


def ratio(d, col):
    g = d.groupby("city")
    num = (d.w * d.wet_2km * d[col]).groupby(d.city).sum()
    cm = (d.w * d[col]).groupby(d.city).sum() / d.w.groupby(d.city).sum()
    den = ((d.w * d.wet_2km).groupby(d.city).sum() * cm)
    return num.sum() / den.sum()


rng = np.random.default_rng(3)
clus = e.clu.unique(); idx = {c: np.where(e.clu.values == c)[0] for c in clus}
for col, lab in (("nonsewer", "non-sewered"), ("pit_none", "pit latrine or no facility"), ("septic", "septic tank")):
    r0 = ratio(e, col)
    bs = []
    for _ in range(500):
        ii = np.concatenate([idx[c] for c in rng.choice(clus, clus.size)])
        bs.append(ratio(e.iloc[ii], col))
    rows.append(dict(measure=f"Ratio of exposure-weighted {lab} share to the city-average share (pooled, residents)",
                     estimate=round(r0, 3), lo=round(np.percentile(bs, 2.5), 3), hi=round(np.percentile(bs, 97.5), 3)))
rows.append(dict(measure="households / clusters / cities", estimate=f"{len(e)} / {e.clu.nunique()} / {e.city.nunique()}", lo=np.nan, hi=np.nan))
t = pd.DataFrame(rows)
t.to_csv("/home/claude/atlas/supplementary/Table_S35_dhs_spatial_sanitation.csv", index=False)
pd.set_option("display.width", 220, "display.max_colwidth", 120)
print(t.to_string(index=False))
