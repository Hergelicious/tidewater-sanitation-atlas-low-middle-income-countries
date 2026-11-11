"""Select monitoring wells within 10 km of the Natural Earth 1:10 m coastline from the Jasechko et al. (2024)
compilation (annual values since 2000; station medians), and plan the DeltaDTM tiles needed (gw_sample.py)."""
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely

GW = "/home/claude/data/gw"
d = pd.read_csv(f"{GW}/AnnualDepthToGroundwater.csv")
e = pd.read_csv(f"{GW}/AnnualGroundwaterElevation.csv")
d, e = d[d.IntegerYear >= 2000], e[e.IntegerYear >= 2000]
m = d.merge(e, on=["StnID", "IntegerYear", "Lat", "Lon"], how="outer")
st = m.groupby("StnID").agg(lat=("Lat", "first"), lon=("Lon", "first"), dtw=("DepthToWater_m", "median"),
                            gwe=("GroundwaterElevation_masl", "median"), n=("IntegerYear", "nunique")).reset_index()
both = m.dropna(subset=["DepthToWater_m", "GroundwaterElevation_masl"])
both = both.assign(gl=both.DepthToWater_m + both.GroundwaterElevation_masl)
st = st.merge(both.groupby("StnID").gl.median().rename("ground_masl"), on="StnID", how="left")
g = gpd.GeoDataFrame(st, geometry=gpd.points_from_xy(st.lon, st.lat), crs=4326)
coast = gpd.read_file("/home/claude/data/ne/ne_10m_coastline.shp")
cu = shapely.unary_union(coast.geometry.values)
g["dist_deg"] = g.geometry.distance(cu)
c = g[g.dist_deg < 0.15].copy()
cl = gpd.GeoSeries([cu], crs=4326).to_crs(3857).iloc[0]
c["dist_km"] = c.to_crs(3857).geometry.distance(cl) / 1000 * np.cos(np.radians(c.lat))
c = c[c.dist_km <= 10]
c.drop(columns="geometry").to_csv(f"{GW}/coastal_wells.csv", index=False)

t = gpd.read_file("/home/claude/data/deltadtm/tiles.gpkg")
c = c[c.dtw.between(-2, 200)].copy()
c["tlon"], c["tlat"] = np.floor(c.lon).astype(int), np.floor(c.lat).astype(int)
cnt = c.groupby(["tlat", "tlon"]).size().rename("n").reset_index()
cnt = cnt.merge(t[["longitude", "latitude", "tile", "zipfile"]], left_on=["tlon", "tlat"],
                right_on=["longitude", "latitude"], how="left")
cnt["region"] = pd.cut(cnt.tlon, [-181, -100, -30, 60, 181], labels=["NAmW", "Americas_E", "EurAfr", "Asia_Oce"])
sel = pd.concat([cnt[(cnt.tile.notna()) & (cnt.region != "NAmW")],
                 cnt[(cnt.tile.notna()) & (cnt.region == "NAmW")].nlargest(40, "n")])
sel.to_csv(f"{GW}/gw_tiles.csv", index=False)
print(len(c), "coastal wells;", len(sel), "tiles")
