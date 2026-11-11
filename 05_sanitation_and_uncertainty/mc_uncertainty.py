"""Joint (Monte Carlo) uncertainty of the exposure estimate.
Per draw: a global vertical datum bias, a city datum error and a city elevation bias (land too high or low), the
groundwater threshold (pit depth plus separation), the JRC tide percentile (shared across cities), the city non-sewered
share, and the share of septic-tank users whose tank discharges to soil (from SFD reports, see sfd_summary.py).
City responses are interpolated from precomputed layers: wet ground as a function of the effective threshold above HAT
(land shifts -1.2 to +1.2 m and thresholds HAT + 1 m and + 5 m), surface flooding as a function of the land shift.
Writes supplementary/Table_S31_joint_uncertainty.csv"""
import json
import numpy as np
import pandas as pd

import os
rng = np.random.default_rng(2026)
VAR = os.environ.get("MCVAR", "base")   # base | corr (country-correlated datum errors) | fixedp (no transfer uncertainty in septic share)
N = 4000
a = pd.read_csv("atlas_cities.csv").set_index("ID_UC_G0")
v = pd.read_csv("city_exposure_v07.csv").set_index("ID_UC_G0").reindex(a.index)
esl = pd.read_csv("city_esl.csv").set_index("ID_UC_G0").reindex(a.index)
sf = pd.read_csv("sfd_city_psoil.csv")
rk = pd.read_csv("/home/claude/data/sfd/atlas_cities_for_sfd.csv").set_index("rank").ID_UC_G0
sf["ID"] = sf.atlas_rank.map(rk)
cm = sf.groupby("iso3").p_soil.mean().values                        # country means (bootstrapped per draw)
SD = float(sf.p_soil.std())                                          # between-city spread
own = pd.Series(sf.p_soil.values, index=sf.ID).groupby(level=0).mean().reindex(a.index).values  # cities with an SFD

tag = lambda s: f"s{s:+.1f}".replace(".", "")
shifts = np.array([1.2, 0.8, 0.4, 0.0, -0.4, -0.8, -1.2])           # land raised (+) or lowered (-)
eD = np.r_[1.0, 3 - shifts, 5.0]                                    # effective threshold above HAT: 1, 1.8, ..., 4.2, 5
Dg = np.c_[a.pop_wet10_gw1.values, np.stack([v[f"pop_D_{tag(s)}"].values for s in shifts], 1), a.pop_wet10_gw5.values]
order = np.argsort(eD); eD, Dg = eD[order], Dg[:, order]
Dg = np.maximum.accumulate(Dg, axis=1)                              # enforce monotone response
sB = shifts[::-1]
Bg = np.stack([v[f"pop_B_{tag(s)}"].values for s in sB], 1)
Bg = np.minimum.accumulate(Bg, axis=1)                              # higher land, less surface flooding
ns = a.nonsewer_frac.values
cats = ["jmp20_septic", "jmp20_latrine_other", "jmp20_unimproved", "jmp20_open_def"]
tot = a[cats].sum(axis=1)
sep = (a.jmp20_septic / tot).where(tot > 0)
sep = sep.fillna(sep.groupby(a.GC_DEV_USR_2025).transform("median")).fillna(sep.median()).values
od = (a.jmp20_open_def / tot).where(tot > 0)
od = od.fillna(od.groupby(a.GC_DEV_USR_2025).transform("median")).fillna(0).values
hsd = ((esl.hat_base_p95 - esl.hat_base_p5) / 3.29).values
n = len(a)


def interp_rows(x, grid, vals):
    """Row-wise linear interpolation of vals (n x k) on grid (k,) at x (n,)."""
    x = np.clip(x, grid[0], grid[-1])
    j = np.clip(np.searchsorted(grid, x) - 1, 0, len(grid) - 2)
    w = (x - grid[j]) / (grid[j + 1] - grid[j])
    r = np.arange(len(x))
    return vals[r, j] * (1 - w) + vals[r, j + 1] * w


out = {k: np.empty(N) for k in ("exposure", "surface", "groundwater_only", "soil_dependent", "soil_dependent_gw_only", "septic_soil_share")}
iso = a.iso3.values; uiso, iidx = np.unique(iso, return_inverse=True)
for i in range(N):
    if VAR == "corr":   # datum error shared within each country (sd 0.3) plus city (0.265): same total variance
        s = rng.normal(0, 0.2) + rng.normal(0, 0.3, uiso.size)[iidx] + rng.normal(0, 0.265, n) + rng.normal(0, 0.25, n)
    else:
        s = rng.normal(0, 0.2) + rng.normal(0, 0.4, n) + rng.normal(0, 0.25, n)   # global datum, city datum, city DEM bias
    t = rng.triangular(2, 3, 4)
    dh = rng.normal() * hsd
    fns = np.clip(ns * np.exp(rng.normal(0, 0.12, n)), 0, 1)
    mu = float(np.clip((cm.mean() if VAR == "fixedp" else rng.choice(cm, cm.size).mean()), 0.02, 0.98))
    var = min(SD ** 2, 0.95 * mu * (1 - mu)); k = mu * (1 - mu) / var - 1
    p = rng.beta(mu * k, (1 - mu) * k, n)
    p = np.where(np.isfinite(own), own, p)
    ps = float(np.average(p, weights=a.ns_wet10_now.values * sep))
    D = interp_rows(t - s + dh, eD, Dg) * fns
    B = np.minimum(interp_rows(s - dh, sB, Bg) * fns, D)
    out["exposure"][i] = D.sum() / 1e6
    out["surface"][i] = B.sum() / 1e6
    out["groundwater_only"][i] = (D - B).sum() / 1e6
    out["soil_dependent"][i] = (D * (1 - od - sep * (1 - p))).sum() / 1e6
    out["soil_dependent_gw_only"][i] = ((D - B) * (1 - od - sep * (1 - p))).sum() / 1e6
    out["septic_soil_share"][i] = ps
rows = [dict(quantity=k, median=round(np.median(x), 1 if k != "septic_soil_share" else 2),
             p05=round(np.percentile(x, 5), 1 if k != "septic_soil_share" else 2),
             p95=round(np.percentile(x, 95), 1 if k != "septic_soil_share" else 2)) for k, x in out.items()]
rows.append(dict(quantity="groundwater_only / surface (%)", median=round(100 * np.median(out["groundwater_only"] / out["surface"]), 0),
                 p05=round(100 * np.percentile(out["groundwater_only"] / out["surface"], 5), 0),
                 p95=round(100 * np.percentile(out["groundwater_only"] / out["surface"], 95), 0)))
rows.append(dict(quantity="P(groundwater_only > 25 million)", median=round(float((out["groundwater_only"] > 25).mean()), 3), p05=np.nan, p95=np.nan))
t = pd.DataFrame(rows)
sfx = "" if VAR == "base" else "_" + VAR
t.to_csv(f"supplementary/Table_S31{sfx}_joint_uncertainty.csv", index=False)
pd.DataFrame(out).to_csv(f"mc_draws{sfx}.csv", index=False)
print(t.to_string(index=False))
