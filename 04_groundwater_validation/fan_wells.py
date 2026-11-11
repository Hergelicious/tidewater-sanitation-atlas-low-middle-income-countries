"""Skill of the Fan et al. water table model and of the tidal rule at the same coastal wells.
Writes supplementary/Table_S11_fan_well_skill.csv."""
import glob
import numpy as np
import pandas as pd
import xarray as xr

ig = pd.read_csv("/home/claude/data/igrac/lmic_wells_validation.csv").assign(set="Low and middle income (IGRAC)")
ja = pd.read_csv("/home/claude/data/gw/wells_validation.csv")
ja = ja.assign(set=np.where(ja.region == "United States", "United States", "Other high income"))
w = pd.concat([ig[["lon", "lat", "dtw", "e_rel", "set", "iso3"]], ja[["lon", "lat", "dtw", "e_rel", "set", "iso3"]]])
w = pd.read_csv("/home/claude/data/gw/wells_all_msl.csv")
w["e_rel"] = w.e_msl  # mean-sea-level referenced elevation (central)
w["fan"] = np.nan
for f in sorted(glob.glob("/home/claude/data/fanwtd/*_WTD_annualmean.nc")):
    d = xr.open_dataset(f)
    la, lo = d.lat.values.astype("float64"), d.lon.values.astype("float64")
    ch = d.WTD.encoding["chunksizes"]
    sel = np.where(w.fan.isna() & w.lat.between(la[0], la[-1]) & w.lon.between(lo[0], lo[-1]))[0]
    if sel.size == 0:
        continue
    i = np.abs(np.subtract.outer(w.lat.values[sel], la)).argmin(1) if sel.size < 3000 else \
        np.clip(np.round((w.lat.values[sel] - la[0]) / (la[1] - la[0])).astype(int), 0, la.size - 1)
    j = np.clip(np.round((w.lon.values[sel] - lo[0]) / (lo[1] - lo[0])).astype(int), 0, lo.size - 1)
    key = pd.DataFrame({"k": sel, "i": i, "j": j, "ci": i // ch[1], "cj": j // ch[2]})
    for (ci, cj), g in key.groupby(["ci", "cj"]):
        r0, c0 = ci * ch[1], cj * ch[2]
        blk = d.WTD.isel(time=0, lat=slice(r0, r0 + ch[1]), lon=slice(c0, c0 + ch[2])).values
        msk = d["mask"].isel(lat=slice(r0, r0 + ch[1]), lon=slice(c0, c0 + ch[2])).values
        v = blk[g.i - r0, g.j - c0]
        v = np.where(msk[g.i - r0, g.j - c0] == 1, -v, np.nan)
        w.loc[g.k.values, "fan"] = v
w = w[w.fan.notna()]


def skill(pred, obs):
    tp, fp = (pred & obs).sum(), (pred & ~obs).sum()
    fn, tn = (~pred & obs).sum(), (~pred & ~obs).sum()
    return dict(n=len(obs), flagged_pct=100 * pred.mean(), obs_shallow_pct=100 * obs.mean(),
                PPV=tp / max(tp + fp, 1), sensitivity=tp / max(tp + fn, 1), specificity=tn / max(tn + fp, 1))


rows = []
for s, x in w.groupby("set"):
    for thr in (3, 5):
        obs = x.dtw.values <= thr
        rows.append(dict(set=s, obs_threshold_m=thr, method="Tidal rule (HAT + 3 m)", **skill(x.e_rel.values <= 3, obs)))
        rows.append(dict(set=s, obs_threshold_m=thr, method=f"Fan model, depth <= {thr} m", **skill(x.fan.values <= thr, obs)))
        rows.append(dict(set=s, obs_threshold_m=thr, method="Both agree", **skill((x.e_rel.values <= 3) & (x.fan.values <= 3), obs)))
t = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(t.round(2).to_string(index=False))
print(w.groupby("set").apply(lambda x: pd.Series({"median_fan_depth_at_flagged": x.loc[x.e_rel <= 3, "fan"].median(),
                                                   "median_obs_depth_at_flagged": x.loc[x.e_rel <= 3, "dtw"].median(),
                                                   "corr_spearman": x[["fan", "dtw"]].corr("spearman").iloc[0, 1]})).round(2))
t.round(3).to_csv("supplementary/Table_S11_fan_well_skill.csv", index=False)
