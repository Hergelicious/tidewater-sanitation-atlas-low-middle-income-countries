"""Extreme sea level, tide and sea-level rise per city: mean of the 3 JRC projection points nearest the
city's geographic centroid (median, 5th and 95th percentiles)."""
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree

E = "/home/claude/data/esl/"
FILES = {"esl_base": "ESL_100RP_Baseline", "esl_45_2050": "ESL_100RP_RCP45-2050", "esl_85_2050": "ESL_100RP_RCP85-2050",
         "hat_base": "max_Astronomical_Tide_Baseline", "rslr_45_2050": "RSLR_RCP45-2050", "rslr_85_2050": "RSLR_RCP85-2050"}


def xyz(la, lo):
    la, lo = np.radians(la), np.radians(lo)
    return np.c_[np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]


c = gpd.read_file("/home/claude/atlas/cities.gpkg")
cent = c.to_crs(3857).geometry.centroid.to_crs(4326)
out = pd.DataFrame({"ID_UC_G0": c.ID_UC_G0})
for k, f in FILES.items():
    d = pd.read_csv(E + f + ".csv").dropna(subset=["valueMedian"])
    dist, i = cKDTree(xyz(d.latitude.values, d.longitude.values)).query(xyz(cent.y.values, cent.x.values), k=3)
    out[k] = d.valueMedian.values[i].mean(axis=1)
    out[k + "_p95"] = d.value95th.values[i].mean(axis=1)
    out[k + "_p5"] = d.value5th.values[i].mean(axis=1)
    out["esl_dist_km"] = dist[:, 0] * 6371
out.to_csv("/home/claude/atlas/city_esl.csv", index=False)
