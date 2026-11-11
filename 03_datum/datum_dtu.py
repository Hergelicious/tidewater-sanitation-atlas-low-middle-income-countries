"""Spatially varying datum offset from the DTU22MDT geodetic mean dynamic topography (Knudsen et al.; XGM2019e geoid,
mean-tide system), converted to the tide-free EGM2008 frame of DeltaDTM with the permanent-tide term
-0.198 (1 + k) P2(sin lat), k = 0.3. Mean sea level above the EGM2008 geoid = MDT + tide term; DeltaDTM is lowered by it.
Offset per city: median of non-fill MDT cells within 25 km of the coastal-zone population centre (50 km if none).
Writes datum_offset.csv and cities/<ID>/dtmmsl.npy. offset_at(lat, lon) is reused for wells."""
import numpy as np
import pandas as pd

MDT = np.load("/home/claude/data/dtumdt/dtu22mdt.npy")  # 1/8 degree, rows 90N..80S, cols 0..360E
RES = 0.125


def tide(lat):
    return -0.198 * 1.3 * (1.5 * np.sin(np.radians(lat)) ** 2 - 0.5)


def offset_at(lat, lon, radius_km=25):
    lat, lon = np.atleast_1d(lat).astype(float), np.atleast_1d(lon).astype(float)
    out = np.full(lat.size, np.nan)
    for k, (a, b) in enumerate(zip(lat, lon)):
        for rad in (radius_km, 2 * radius_km, 4 * radius_km):
            dr = int(np.ceil(rad / 111.32 / RES)) + 1
            r0 = int(round((90 - a) / RES)); c0 = int(round((b % 360) / RES))
            rr = np.arange(r0 - dr, r0 + dr + 1).clip(0, MDT.shape[0] - 1)
            cc = np.arange(c0 - dr, c0 + dr + 1) % (MDT.shape[1] - 1)
            sub = MDT[np.ix_(rr, cc)]
            la = 90 - rr * RES; lo = cc * RES
            dist = np.sqrt(((la[:, None] - a) * 111.32) ** 2 + ((((lo[None, :] - b % 360) + 180) % 360 - 180) * 111.32 * np.cos(np.radians(a))) ** 2)
            v = sub[(dist <= rad) & (sub != 0)]
            if v.size:
                out[k] = np.median(v) + tide(a)
                break
    return out


if __name__ == "__main__":
    from pipeline import CDIR
    import geopandas as gpd
    atl = pd.read_csv("atlas_cities.csv")[["ID_UC_G0", "zone_lat", "zone_lon"]]
    c = gpd.read_file("cities.gpkg").to_crs(3857)
    cen = c.geometry.centroid.to_crs(4326)
    cc = pd.DataFrame({"ID_UC_G0": c.ID_UC_G0, "clat": cen.y, "clon": cen.x})
    atl = atl.merge(cc, on="ID_UC_G0")
    la = atl.zone_lat.fillna(atl.clat).values; lo = atl.zone_lon.fillna(atl.clon).values
    atl["datum_offset_m"] = offset_at(la, lo)
    old = pd.read_csv("datum_offset.csv").rename(columns={"datum_offset_m": "offset_gll_smoothed"})
    out = atl[["ID_UC_G0", "datum_offset_m"]].merge(old[["ID_UC_G0", "offset_gll_smoothed"]], on="ID_UC_G0", how="left")
    out.to_csv("datum_offset.csv", index=False)
    for u, off in zip(out.ID_UC_G0, out.datum_offset_m):
        d = np.load(f"{CDIR}/{u}/dtm.npy")
        np.save(f"{CDIR}/{u}/dtmmsl.npy", (d - (off if np.isfinite(off) else 0)).astype("float32"))
    print(out.datum_offset_m.describe().round(2), out.datum_offset_m.isna().sum())
    print("corr with interim:", round(out[["datum_offset_m", "offset_gll_smoothed"]].corr().iloc[0, 1], 2))
