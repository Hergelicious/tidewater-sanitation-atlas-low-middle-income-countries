"""Soil-dependent sanitation users on wet ground: within the groundwater-only group, leave-one-country-out sensitivity of
the septic discharge share, and city rankings for soil-dependent users versus all non-sewered residents.
Soil-dependent = latrines and other improved + unimproved + septic users whose tank discharges to soil (SFD share).
Writes supplementary/Table_S33_soil_dependent.csv, Table_S34_city_ranking_soil.csv"""
import numpy as np
import pandas as pd

a = pd.read_csv("atlas_cities.csv").set_index("ID_UC_G0")
cats = ["jmp20_septic", "jmp20_latrine_other", "jmp20_unimproved", "jmp20_open_def"]
tot = a[cats].sum(axis=1)
fr = {}
for c in cats:
    x = (a[c] / tot).where(tot > 0)
    fr[c] = x.fillna(x.groupby(a.GC_DEV_USR_2025).transform("median")).fillna(x.median())
sep, od = fr["jmp20_septic"], fr["jmp20_open_def"]
sf = pd.read_csv("sfd_city_psoil.csv")
rk = pd.read_csv("/home/claude/data/sfd/atlas_cities_for_sfd.csv").set_index("rank").ID_UC_G0
sf["ID"] = sf.atlas_rank.map(rk)
own = sf.groupby("ID").p_soil.mean()


def soil_factor(pbar, own_ok=True):
    p = pd.Series(pbar, index=a.index)
    if own_ok:
        p.loc[own.index.intersection(a.index)] = own
    return 1 - od - sep * (1 - p)


M = 1e6
D, B = a.ns_wet10_now, a.ns_cr_now
G = (D - B).clip(lower=0)
pbar = sf.groupby("iso3").p_soil.mean().mean()
f = soil_factor(pbar)
rows = [("Mean of country means of septic soil share (SFD)", round(pbar, 3)),
        ("All non-sewered residents on wet ground", round(D.sum() / M, 1)),
        ("  soil-dependent", round((D * f).sum() / M, 1)),
        ("  septic discharging to drains, water bodies or sealed", round((D * (1 - f - od)).sum() / M, 1)),
        ("  no facility", round((D * od).sum() / M, 1)),
        ("Groundwater contact only (not reached by surface flooding)", round(G.sum() / M, 1)),
        ("  soil-dependent", round((G * f).sum() / M, 1)),
        ("Surface flooding (coastal or river)", round(B.sum() / M, 1)),
        ("  soil-dependent", round((B * f).sum() / M, 1))]
# leave one country out of the SFD sample (transfer uncertainty)
cm = sf.groupby("iso3").p_soil.mean()
loo = []
for c in cm.index:
    pb = cm.drop(c).mean()
    loo.append((c, pb, (D * soil_factor(pb)).sum() / M, (G * soil_factor(pb)).sum() / M))
loo = pd.DataFrame(loo, columns=["omitted", "p_soil", "soil_all", "soil_gw_only"])
rows += [("Leave one country out: septic soil share, range", f"{loo.p_soil.min():.2f} to {loo.p_soil.max():.2f}"),
         ("Leave one country out: soil-dependent on wet ground, range (millions)", f"{loo.soil_all.min():.1f} to {loo.soil_all.max():.1f}"),
         ("Leave one country out: soil-dependent, groundwater only, range (millions)", f"{loo.soil_gw_only.min():.1f} to {loo.soil_gw_only.max():.1f}"),
         ("Bangladesh omitted: septic soil share", round(float(loo.set_index("omitted").loc["BGD", "p_soil"]), 2)),
         ("Bangladesh omitted: soil-dependent on wet ground (millions)", round(float(loo.set_index("omitted").loc["BGD", "soil_all"]), 1)),
         ("City median instead of country means: soil-dependent (millions)", round((D * soil_factor(sf.p_soil.median())).sum() / M, 1)),
         ("No city-specific SFD shares (country mean everywhere): soil-dependent (millions)", round((D * soil_factor(pbar, False)).sum() / M, 1)),
         ("Bounds: no septic soil-dependent / all septic soil-dependent (millions)", f"{(D * soil_factor(0, False)).sum() / M:.1f} to {(D * soil_factor(1, False)).sum() / M:.1f}")]
t = pd.DataFrame(rows, columns=["quantity", "value"])
t.to_csv("supplementary/Table_S33_soil_dependent.csv", index=False)
loo.round(3).to_csv("supplementary/Table_S33b_soil_loo.csv", index=False)
print(t.to_string(index=False)); print(loo.round(2).to_string(index=False))
# rankings
S = D * f
r = pd.DataFrame({"city": a.GC_UCN_MAI_2025, "iso3": a.iso3, "nonsewered_wet_M": D / M, "soil_dependent_M": S / M,
                  "soil_share_pct": 100 * f})
r["rank_nonsewered"] = r.nonsewered_wet_M.rank(ascending=False, method="min").astype(int)
r["rank_soil"] = r.soil_dependent_M.rank(ascending=False, method="min").astype(int)
r = r.sort_values("rank_soil")
o20 = len(set(r.nsmallest(20, "rank_nonsewered").index) & set(r.nsmallest(20, "rank_soil").index))
o50 = len(set(r.nsmallest(50, "rank_nonsewered").index) & set(r.nsmallest(50, "rank_soil").index))
rho = r.nonsewered_wet_M.rank().corr(r.soil_dependent_M.rank())
print(f"top20 overlap {o20}, top50 overlap {o50}, spearman {rho:.3f}")
r.head(50).round(3).to_csv("supplementary/Table_S34_city_ranking_soil.csv", index=False)
print(r.head(25).round(2).to_string())
mv = r.assign(d=r.rank_nonsewered - r.rank_soil).query("rank_soil <= 20 or rank_nonsewered <= 20")
print(mv.sort_values("d")[["city", "iso3", "rank_nonsewered", "rank_soil", "soil_share_pct"]].head(8).to_string())
print(mv.sort_values("d")[["city", "iso3", "rank_nonsewered", "rank_soil", "soil_share_pct"]].tail(8).to_string())
open("soil_rank_summary.txt", "w").write(f"{o20},{o50},{rho:.3f}")
