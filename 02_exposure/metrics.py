"""City-level wet-ground exposure for the Tidewater Sanitation Atlas.

Wet ground (any of):
  C  coastal flood: land hydraulically connected to the sea with bare-earth elevation <= 100-yr extreme sea level
  G  shallow groundwater: elevation <= highest astronomical tide (+ sea-level rise) + pit-depth threshold (3 m main; 1, 5 m sensitivity)
  R  river flood: GloFAS 100-yr depth > 0
Present day and 2050 (RCP4.5, RCP8.5; median projections).
"""
import json
import sys
import os
import numpy as np
DTMNAME = os.environ.get("DTMNAME", "dtmmsl")  # DeltaDTM referenced to local mean sea level (central); 'dtm' = EGM2008
import pandas as pd
import geopandas as gpd
from rasterio.features import geometry_mask
from scipy import ndimage
from multiprocessing import Pool

sys.path.insert(0, "/home/claude/atlas")
from pipeline import load_grids, grid_transform, CDIR  # noqa: E402

ATL = "/home/claude/atlas"
cities = gpd.read_file(f"{ATL}/cities.gpkg").to_crs(4326).set_index("ID_UC_G0")
esl = pd.read_csv(f"{ATL}/city_esl.csv").set_index("ID_UC_G0")
grids = load_grids()
STRUCT = np.ones((3, 3), bool)
LAND = gpd.read_file("/home/claude/data/ne/ne_10m_land.shp").to_crs(4326)
COAST = gpd.read_file("/home/claude/data/ne/ne_10m_coastline.shp").to_crs(4326)
LAND_SIDX, COAST_SIDX = LAND.sindex, COAST.sindex


def coast_masks(uid, g, shape):
    """Masks of pixels within 5/10/20 km of the Natural Earth 1:10 m coastline (which follows large estuaries), and of ocean (non-land) pixels."""
    from shapely.geometry import box
    from rasterio.features import rasterize
    tr = grid_transform(g)
    x0, y1 = tr.c, tr.f
    x1, y0 = x0 + g["w"] * tr.a, y1 + g["h"] * tr.e
    big = box(x0 - 0.3, y0 - 0.3, x1 + 0.3, y1 + 0.3)
    lat = (y0 + y1) / 2
    lon = (x0 + x1) / 2
    utm = 32600 + int((lon + 180) // 6) + 1 if lat >= 0 else 32700 + int((lon + 180) // 6) + 1
    cl = COAST.iloc[COAST_SIDX.query(big)].clip(big)
    out = {}
    for dkm in (5, 10, 20):
        if len(cl) == 0:
            out[dkm] = np.zeros(shape, bool)
            continue
        buf = cl.to_crs(utm).buffer(dkm * 1000).to_crs(4326)
        out[dkm] = rasterize([(geom, 1) for geom in buf if not geom.is_empty], out_shape=shape,
                             transform=tr, fill=0, dtype="uint8").astype(bool)
    ld = LAND.iloc[LAND_SIDX.query(big)].clip(big)
    landm = rasterize([(geom, 1) for geom in ld.geometry if not geom.is_empty], out_shape=shape, transform=tr,
                      fill=0, dtype="uint8").astype(bool) if len(ld) else np.zeros(shape, bool)
    return out, ~landm


def ocean_seeds(sea, ocean):
    lab, n = ndimage.label(sea, structure=STRUCT)
    if n == 0:
        return np.zeros_like(sea)
    keep = np.zeros(n + 1, bool)
    keep[np.unique(lab[sea & ocean])] = True
    keep[0] = False
    return keep[lab]


def bathtub(dtm, sea, level, near=None):
    wet = sea | (dtm <= level)
    if near is not None:
        wet &= near | sea
    lab, n = ndimage.label(wet, structure=STRUCT)
    if n == 0:
        return np.zeros_like(sea)
    keep = np.zeros(n + 1, bool)
    keep[np.unique(lab[sea])] = True
    keep[0] = False
    return keep[lab] & ~sea


def load(uid, name, shape):
    try:
        return np.load(f"{CDIR}/{uid}/{name}.npy")
    except FileNotFoundError:
        return np.full(shape, np.nan, "float32")


def city(uid):
    g = grids[uid]
    shape = (g["h"], g["w"])
    inside = ~geometry_mask([cities.loc[uid, "geometry"]], out_shape=shape, transform=grid_transform(g))
    pop = np.nan_to_num(load(uid, "pop", shape)) * inside
    dtm = load(uid, DTMNAME, shape)
    seaf = load(uid, "sea", shape)
    riv = np.nan_to_num(load(uid, "riv", shape))
    vous = np.nan_to_num(load(uid, "vous", shape))
    sea = np.nan_to_num(seaf) >= 0.5
    has_dtm = np.isfinite(dtm) | sea
    d = np.where(np.isfinite(dtm), dtm, 9999.0)  # no tile or >30 m: treated as high ground
    e = esl.loc[uid]
    near, ocean = coast_masks(uid, g, shape)
    seeds = ocean_seeds(sea, ocean)
    land = ~sea
    n10 = near[10]

    def bt(level, nr=n10):
        return bathtub(d, seeds, level, nr) & land

    def gw(level, nr=n10):
        return land & nr & (d <= level)
    lay = {
        "coast_now": bt(e.esl_base), "coast_45": bt(e.esl_45_2050), "coast_85": bt(e.esl_85_2050),
        "coast_85_p95": bt(e.esl_85_2050_p95),
        "coast_now_5km": bt(e.esl_base, near[5]), "coast_now_20km": bt(e.esl_base, near[20]),
        "coast_now_nolimit": bathtub(d, seeds, e.esl_base) & land,
        "gw1_now": gw(e.hat_base + 1), "gw3_now": gw(e.hat_base + 3), "gw5_now": gw(e.hat_base + 5),
        "gw3_now_5km": gw(e.hat_base + 3, near[5]), "gw3_now_20km": gw(e.hat_base + 3, near[20]),
        "gw3_45": gw(e.hat_base + e.rslr_45_2050 + 3),
        "gw3_85": gw(e.hat_base + e.rslr_85_2050 + 3),
        "river": land & (riv > 0) & (riv < 100),
        "vous_now": land & (vous > 0) & (vous < 100),
        "low2m": land & (d <= 2), "low5m": land & (d <= 5),
    }
    lay["wet_now"] = lay["coast_now"] | lay["gw3_now"] | lay["river"]
    lay["wet_45"] = lay["coast_45"] | lay["gw3_45"] | lay["river"]
    lay["wet_85"] = lay["coast_85"] | lay["gw3_85"] | lay["river"]
    lay["wet_now_gw1"] = lay["coast_now"] | lay["gw1_now"] | lay["river"]
    lay["wet_now_gw5"] = lay["coast_now"] | lay["gw5_now"] | lay["river"]
    lay["wet_now_vous"] = lay["vous_now"] | lay["gw3_now"] | lay["river"]
    lay["wet_now_5km"] = lay["coast_now_5km"] | lay["gw3_now_5km"] | lay["river"]
    lay["wet_now_20km"] = lay["coast_now_20km"] | lay["gw3_now_20km"] | lay["river"]
    lay["river_10"] = lay["river"] & n10
    lay["vous_10"] = lay["vous_now"] & n10
    lay["wet10_now"] = lay["coast_now"] | lay["gw3_now"] | lay["river_10"]
    lay["wet10_85"] = lay["coast_85"] | lay["gw3_85"] | lay["river_10"]
    # uncertainty bounds: elevation error (DeltaDTM MAE 0.45 m), sea-level percentiles, pit depth 2-4 m
    dlo, dhi = d + 0.45, d - 0.45
    rv = lay["river_10"]
    lay["wet10_low"] = (bathtub(dlo, seeds, e.esl_base_p5, n10) & land) | (land & n10 & (dlo <= e.hat_base_p5 + 2)) | rv
    lay["wet10_high"] = (bathtub(dhi, seeds, e.esl_base_p95, n10) & land) | (land & n10 & (dhi <= e.hat_base_p95 + 4)) | rv
    lay["wet10_85_low"] = (bathtub(dlo, seeds, e.esl_85_2050_p5, n10) & land) | \
        (land & n10 & (dlo <= e.hat_base_p5 + e.rslr_85_2050_p5 + 2)) | rv
    lay["wet10_85_high"] = (bathtub(dhi, seeds, e.esl_85_2050_p95, n10) & land) | \
        (land & n10 & (dhi <= e.hat_base_p95 + e.rslr_85_2050_p95 + 4)) | rv
    # vertical datum: DeltaDTM is on EGM2008, sea levels on mean sea level; test land 0.3 m lower
    dd = d - 0.3
    lay["wet10_datum03"] = (bathtub(dd, seeds, e.esl_base, n10) & land) | (land & n10 & (dd <= e.hat_base + 3)) | rv
    lay["tidal_only_low"] = (bathtub(dlo, seeds, e.esl_base_p5, n10) & land) | (land & n10 & (dlo <= e.hat_base_p5 + 2))
    lay["tidal_only_high"] = (bathtub(dhi, seeds, e.esl_base_p95, n10) & land) | (land & n10 & (dhi <= e.hat_base_p95 + 4))
    lay["wet10_gw1"] = lay["coast_now"] | lay["gw1_now"] | rv
    lay["wet10_gw5"] = lay["coast_now"] | lay["gw5_now"] | rv
    lay["wet10_surgemodel"] = lay["vous_10"] | lay["gw3_now"] | rv
    lay["wet5km"] = lay["coast_now_5km"] | lay["gw3_now_5km"] | (lay["river"] & near[5])
    lay["wet20km"] = lay["coast_now_20km"] | lay["gw3_now_20km"] | (lay["river"] & near[20])
    # flood defences: coastal flooding removed, groundwater and river flooding kept
    lay["def_now"] = lay["gw3_now"] | rv
    lay["def_85"] = lay["gw3_85"] | rv
    # comparison: elevation left on the EGM2008 geoid (no sea-level referencing)
    de = np.where(np.isfinite(load(uid, "dtm", shape)), load(uid, "dtm", shape), 9999.0)
    lay["wet10_egm"] = (bathtub(de, seeds, e.esl_base, n10) & land) | (land & n10 & (de <= e.hat_base + 3)) | rv
    lay["gw3_egm"] = land & n10 & (de <= e.hat_base + 3)
    # sea level shifted from the 1980-2014 baseline to 2020 (+0.08 m global mean rise)
    lay["wet10_sl08"] = (bathtub(d, seeds, e.esl_base + 0.08, n10) & land) | (land & n10 & (d <= e.hat_base + 0.08 + 3)) | rv
    # datum uncertainty: land 0.4 m higher or lower relative to mean sea level
    for nm_, sh in (("dp04", 0.4), ("dm04", -0.4)):
        dd_ = d + sh
        lay["wet10_" + nm_] = (bathtub(dd_, seeds, e.esl_base, n10) & land) | (land & n10 & (dd_ <= e.hat_base + 3)) | rv
    # like-for-like comparison and overlap with the storm-surge map
    lay["cr_now"] = lay["coast_now"] | rv
    lay["vous_in_wet"] = lay["vous_10"] & lay["wet10_now"]
    # distance from the sea (km), bands
    from scipy import ndimage as _nd
    _lat = abs(grid_transform(g).f) if False else abs(g.get("lat0", 0))
    _t = grid_transform(g); _la = np.radians(_t.f - _t.a * shape[0] / 2)
    dist = _nd.distance_transform_edt(~sea, sampling=(0.0926, 0.0926 * np.cos(_la)))
    for lo_, hi_ in ((0, 1), (1, 3), (3, 5), (5, 10)):
        band = (dist > lo_) & (dist <= hi_) if lo_ > 0 else (dist <= hi_)
        lay[f"wet10_d{lo_}_{hi_}"] = lay["wet10_now"] & band
        lay[f"gw3_d{lo_}_{hi_}"] = lay["gw3_now"] & band
        lay[f"zone_d{lo_}_{hi_}"] = land & n10 & band
    lay["tidal_now"] = lay["coast_now"] | lay["gw3_now"]
    lay["tidal_85"] = lay["coast_85"] | lay["gw3_85"]
    zp = pop * n10
    if zp.sum() > 0:
        rr, cc = np.indices(shape)
        r_ = (rr * zp).sum() / zp.sum()
        c_ = (cc * zp).sum() / zp.sum()
        zlon, zlat = grid_transform(g) * (c_ + 0.5, r_ + 0.5)
    else:
        zlon = zlat = np.nan
    out = {"ID_UC_G0": uid, "pop_grid": float(pop.sum()), "pop_within10km": float(pop[n10].sum()),
           "zone_lon": float(zlon), "zone_lat": float(zlat), "pop_on_sea_pixels_zone": float(pop[sea & n10].sum()),
           "dtm_cover": float((has_dtm & inside).sum() / max(inside.sum(), 1))}
    pop50 = np.nan_to_num(load(uid, "pop50", shape)) * inside
    out["pop50_grid"] = float(pop50.sum())
    out["pop50_within10km"] = float(pop50[n10].sum())
    for k, m in lay.items():
        out[f"pop_{k}"] = float(pop[m].sum())
        if k in ("wet10_now", "wet10_85", "wet10_85_low", "wet10_85_high", "tidal_85", "coast_85", "gw3_85",
                 "river_10", "def_85", "vous_10"):
            out[f"pop50_{k}"] = float(pop50[m].sum())
    return out


if __name__ == "__main__":
    ids = list(grids)
    with Pool(2) as p:
        res = p.map(city, ids, chunksize=8)
    df = pd.DataFrame(res)
    df.to_csv(f"{ATL}/city_exposure_raw.csv", index=False)
    print(df.describe().T.round(0).to_string())
