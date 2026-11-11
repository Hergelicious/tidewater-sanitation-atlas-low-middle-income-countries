"""Merge city exposure with sanitation, food and health covariates and derive all reported columns.

Non-sewered population = population x (1 - urban sewer share / 100), JMP 2020 to match the 2020 population
(national total where a country file has no urban figure; regional median where no figure exists, flagged).
The latest JMP year (2024 for most countries) is kept as a sensitivity (ns24_*).
ns_*   : non-sewered, 2020 population
ns50_* : non-sewered, 2050 population, sewer share held at its latest value
ns50t_*: non-sewered, 2050 population, sewer share extended along its 2015-2024 trend (sewer_trend.py)
"""
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree

ATL = "/home/claude/atlas"
ex = pd.read_csv(f"{ATL}/city_exposure_raw.csv")
c = gpd.read_file(f"{ATL}/cities.gpkg").drop(columns="geometry")
iso = pd.read_csv(f"{ATL}/country_iso3.csv")
ct = pd.read_csv(f"{ATL}/country_table.csv")
esl = pd.read_csv(f"{ATL}/city_esl.csv")
trend = pd.read_csv(f"{ATL}/sewer_trend.csv")[["iso3", "slope_used"]]
j20 = pd.read_csv(f"{ATL}/jmp_2020.csv")

c = c.merge(iso, left_on="GC_CNT_GAD_2025", right_on="country", how="left")
df = c.merge(ex, on="ID_UC_G0").merge(ct.drop(columns="country"), on="iso3", how="left").merge(esl, on="ID_UC_G0")

df = df.merge(j20, on="iso3", how="left")
df["sewer_imputed"] = df.jmp20_sewer.isna()
regs = c[["iso3", "GC_DEV_USR_2025"]].drop_duplicates("iso3")
reg_med = ct.merge(regs, on="iso3").groupby("GC_DEV_USR_2025").jmp_sewer.median()
reg_med20 = j20.merge(regs, on="iso3").groupby("GC_DEV_USR_2025").jmp20_sewer.median()
df["sewer_pct"] = df.jmp20_sewer.fillna(df.GC_DEV_USR_2025.map(reg_med20))
df["sewer_pct_2024"] = df.jmp_sewer.fillna(df.GC_DEV_USR_2025.map(reg_med))
df["sewer_source"] = np.where(df.sewer_imputed, "regional median", df.jmp20_setting.fillna("urban"))
df["nonsewer_frac"] = 1 - df.sewer_pct / 100
df["unsafe_frac"] = (df.jmp20_unimproved.fillna(0) + df.jmp20_open_def.fillna(0)) / 100
df = df.merge(trend, on="iso3", how="left")
df["sewer_2050_trend"] = np.minimum(100, df.sewer_pct + df.slope_used.fillna(0) * (2050 - 2020))

new = {}
for col in ex.columns:
    if col.startswith("pop_") and col not in ("pop_grid", "pop_on_sea_pixels_zone"):
        new["ns_" + col[4:]] = df[col] * df.nonsewer_frac
        if col in ("pop_wet10_now", "pop_within10km", "pop_gw3_now", "pop_vous_10"):
            new["ns24_" + col[4:]] = df[col] * (1 - df.sewer_pct_2024 / 100)
    if col.startswith("pop50_") and col != "pop50_grid":
        new["ns50_" + col[6:]] = df[col] * df.nonsewer_frac
        new["ns50t_" + col[6:]] = df[col] * (1 - df.sewer_2050_trend / 100)
new["share_wet_zone_now"] = (df.pop_wet10_now / df.pop_within10km.where(df.pop_within10km > 0)).fillna(0)

# estuarine flag: distance from the coastal-zone population centre to the generalised 1:110m coastline,
# which closes estuary mouths (the 1:10m coastline used for the zones follows large estuaries inland)
ok = df.zone_lat.notna()
coarse = gpd.read_file("/home/claude/data/ne/ne_110m_coastline.shp").to_crs(4326)
pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(df.zone_lon.fillna(0), df.zone_lat.fillna(0)), crs=4326)
dist = np.full(len(df), np.nan)
for i in np.where(ok.values)[0]:
    p = pts.geometry.iloc[i]
    utm = 32600 + int((p.x + 180) // 6) + 1 if p.y >= 0 else 32700 + int((p.x + 180) // 6) + 1
    near = coarse.iloc[coarse.sindex.query(p.buffer(3))]
    if len(near) == 0:
        continue
    dist[i] = near.to_crs(utm).distance(gpd.GeoSeries([p], crs=4326).to_crs(utm).iloc[0]).min() / 1000
new["zone_dist_to_open_coast_km"] = dist
df = pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)
df["estuarine"] = df.zone_dist_to_open_coast_km > 40
df.to_csv(f"{ATL}/atlas_cities.csv", index=False)

M = lambda col, x=df: round(x[col].sum() / 1e6, 1)  # noqa: E731
print("cities", len(df), "countries", df.iso3.nunique(), "| sewer source:", df.sewer_source.value_counts().to_dict())
for k in ["within10km", "wet10_now", "wet10_low", "wet10_high", "tidal_now", "tidal_only_low", "tidal_only_high",
          "river_10", "vous_10", "gw3_now", "coast_now", "def_now", "wet10_datum03", "wet10_gw1", "wet10_gw5",
          "wet5km", "wet20km", "wet10_surgemodel", "wet10_85", "wet10_egm", "gw3_egm"]:
    print(f"{k:18s} ns {M('ns_' + k):6}  all {M('pop_' + k):6}")
for k in ["wet10_85", "wet10_85_low", "wet10_85_high"]:
    print(f"2050 {k:14s} const {M('ns50_' + k):6}  trend {M('ns50t_' + k):6}")
print("estuarine cities", int(df.estuarine.sum()), "ns_wet", M("ns_wet10_now", df[df.estuarine]),
      "| without them ns_wet", M("ns_wet10_now", df[~df.estuarine]), "surge", M("ns_vous_10", df[~df.estuarine]))
print("2024 sanitation sensitivity", M("ns24_wet10_now"), "zone", M("ns24_within10km"))
print("excl imputed", M("ns_wet10_now", df[~df.sewer_imputed]))
r = df.groupby("GC_DEV_USR_2025")[["ns_wet10_now", "ns_tidal_now", "ns_vous_10", "ns50_wet10_85", "ns50t_wet10_85"]].sum() / 1e6
print(r.round(1))
s = df.ns_wet10_now.sort_values(ascending=False)
print("top20 share", round(s.head(20).sum() / s.sum() * 100, 1), "top50", round(s.head(50).sum() / s.sum() * 100, 1))
f = df.fish_share_animal_protein_pct
print("fish>=30:", df.loc[f >= 30, "iso3"].nunique(), M("ns_wet10_now", df[f >= 30]),
      round(df.loc[f >= 30, "ns_wet10_now"].sum() / df.ns_wet10_now.sum() * 100, 1))
print("unsafe on wet", round((df.pop_wet10_now * df.unsafe_frac).sum() / 1e6, 1))
