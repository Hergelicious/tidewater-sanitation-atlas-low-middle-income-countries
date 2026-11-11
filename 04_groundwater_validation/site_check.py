"""Compare the wet-ground classification with published site observations of sanitation-related groundwater problems
(Dakar Thiaroye, Dar es Salaam, Cotonou, Jakarta). For each site: wet-ground and groundwater-contact shares of
residents within 1 km, and median ground elevation above the highest astronomical tide.
Writes supplementary/Table_S29_site_evidence.csv"""
import numpy as np
import pandas as pd
from rasterio.features import geometry_mask
import metrics as M
from pipeline import grid_transform

SITES = [  # city uid, site, lat, lon, evidence, source
    (39, "Pikine", 14.752, -17.396, "Nitrate above 50 mg/L in 82% of sources; E. coli in about half; septic tanks dominant", "Diaw 2020"),
    (39, "Guediawaye", 14.777, -17.390, "As above", "Diaw 2020"),
    (39, "Golf Sud", 14.772, -17.414, "As above (validation grid)", "Diaw 2020"),
    (39, "Medina Gounass", 14.772, -17.382, "As above (validation grid)", "Diaw 2020"),
    (39, "Yeumbeul", 14.776, -17.357, "Water table < 2 m in dry season; nitrate 69 to 599 mg/L; leaking septic tanks", "Pouye 2022"),
    (39, "Keur Massar", 14.782, -17.311, "As above", "Pouye 2022"),
    (39, "Malika", 14.801, -17.338, "As above", "Pouye 2022"),
    (3570, "Kawe", -6.737, 39.222, "Nitrate 435 mg/L in shallow well; water table < 1 m near coast", "Mtoni 2013"),
    (3570, "Mbagala", -6.900, 39.264, "Nitrate up to 421 mg/L", "Mtoni 2013 (citing earlier work)"),
    (3570, "Buguruni", -6.838, 39.244, "Nitrate up to 151 mg/L", "Mtoni 2013 (citing earlier work)"),
    (3570, "Ilala", -6.829, 39.263, "Nitrate up to 151 mg/L", "Mtoni 2013 (citing earlier work)"),
    (3570, "Keko", -6.838, 39.276, "Nitrate 74 to 108 mg/L", "Mtoni 2013 (citing earlier work)"),
    (1588, "Ladji", 6.391, 2.431, "Groundwater < 1 m; all wells positive for faecal coliforms; wells < 5 m from septic tanks", "Totin 2008"),
    (1588, "Vossa", 6.389, 2.416, "As above", "Totin 2008"),
    (1588, "Ahouansori", 6.388, 2.421, "As above", "Totin 2008"),
    (1588, "Djidje", 6.383, 2.431, "As above", "Totin 2008"),
    (1588, "Hinde", 6.379, 2.433, "As above", "Totin 2008"),
    (1588, "Yenawa", 6.390, 2.461, "As above", "Totin 2008"),
    (1588, "Avotrou", 6.379, 2.472, "As above", "Totin 2008"),
    (1588, "Enagnon", 6.358, 2.450, "As above", "Totin 2008"),
    (1588, "Placodji", 6.352, 2.433, "As above", "Totin 2008"),
    (5472, "Jakarta Utara (coastal north)", -6.135, 106.893, "Total coliforms above standard in 84% of wells, E. coli 55%", "Khairunnisa 2025"),
    (5472, "Jakarta Selatan (inland south, contrast)", -6.284, 106.805, "Total coliforms 53%, E. coli 24%", "Khairunnisa 2025"),
]


def masks(uid):
    g = M.grids[uid]
    shape = (g["h"], g["w"])
    tr = grid_transform(g)
    inside = ~geometry_mask([M.cities.loc[uid, "geometry"]], out_shape=shape, transform=tr)
    pop = np.nan_to_num(M.load(uid, "pop", shape))
    dtm = M.load(uid, M.DTMNAME, shape)
    sea = np.nan_to_num(M.load(uid, "sea", shape)) >= 0.5
    riv = np.nan_to_num(M.load(uid, "riv", shape))
    d = np.where(np.isfinite(dtm), dtm, 9999.0)
    e = M.esl.loc[uid]
    near, ocean = M.coast_masks(uid, g, shape)
    seeds = M.ocean_seeds(sea, ocean)
    land, n10 = ~sea, near[10]
    C = M.bathtub(d, seeds, e.esl_base, n10) & land
    G = land & n10 & (d <= e.hat_base + 3)
    R = land & (riv > 0) & (riv < 100) & n10
    return dict(tr=tr, pop=pop, D=C | G | R, G=G, land=land, n10=n10, eh=d - e.hat_base, shape=shape, inside=inside)


cache, rows = {}, []
for uid, site, lat, lon, ev, src in SITES:
    if uid not in cache:
        cache[uid] = masks(uid)
    m = cache[uid]
    rr, cc = np.indices(m["shape"])
    xs, ys = m["tr"] * (cc + 0.5, rr + 0.5)
    dist = np.hypot((xs - lon) * 111.32 * np.cos(np.radians(lat)), (ys - lat) * 110.57)
    w = (dist <= 1.0) & m["land"]
    p = m["pop"] * w
    share = lambda k: 100 * (p * m[k]).sum() / p.sum() if p.sum() > 0 else np.nan
    eh = m["eh"][w & (m["eh"] < 9000)]
    rows.append(dict(city=M.cities.loc[uid, "GC_UCN_MAI_2025"], site=site, lat=lat, lon=lon, evidence=ev, source=src,
                     in_coastal_zone=bool(m["n10"][w].mean() > 0.5), wet_ground_pct=round(share("D"), 0),
                     groundwater_contact_pct=round(share("G"), 0),
                     median_elev_above_HAT_m=round(float(np.median(eh)), 1) if eh.size else np.nan))
t = pd.DataFrame(rows)
# city context: share of coastal-zone residents on wet ground
for uid in cache:
    m = cache[uid]
    z = m["pop"] * m["n10"] * m["inside"] * m["land"]
    print(M.cities.loc[uid, "GC_UCN_MAI_2025"], "city zone wet share %", round(100 * (z * m["D"]).sum() / z.sum(), 1))
pd.set_option("display.width", 250)
print(t.drop(columns=["evidence", "source"]).to_string(index=False))
t.to_csv("supplementary/Table_S29_site_evidence.csv", index=False)
