"""Supplementary Table 10: children under five, 2050 decomposition, subsidence and population disaggregation tests."""
import pandas as pd
d = pd.read_csv("atlas_cities.csv").merge(pd.read_csv("city_exposure_extra.csv"), on="ID_UC_G0").merge(
    pd.read_csv("city_children.csv"), on="ID_UC_G0")
f, M = d.nonsewer_frac, 1e6
S = lambda c: round((d[c] * f).sum() / M, 1)  # noqa: E731
T = lambda c: round(d[c].sum() / M, 1)  # noqa: E731
u5 = lambda c: round((d[c] * d.u5_share_pct / 100).sum() / M, 2)  # noqa: E731
rows = [("Children under five, central", u5("ns_wet10_now")), ("Children under five, low", u5("ns_wet10_low")),
        ("Children under five, high", u5("ns_wet10_high")),
        ("2020", T("ns_wet10_now")), ("2050 sea level, 2020 population", T("ns_wet10_85")),
        ("2020 sea level, 2050 population", S("pop50_wet10_now_mask")), ("2050 sea level and population", T("ns50_wet10_85")),
        ("2050, plus land subsidence at present rates", S("pop50_wet10_85_subs")),
        ("2050, growth spread evenly within each city", S("pop50u_wet10_85")),
        ("2050, population on cells empty in 2020", S("pop50_wet10_85_newcells")),
        ("2050, sewer trend continued", T("ns50t_wet10_85"))]
t = pd.DataFrame(rows, columns=["quantity", "nonsewered_residents_millions"])
t.to_csv("supplementary/Table_S10_children_2050_decomposition.csv", index=False)
print(t.to_string(index=False))
