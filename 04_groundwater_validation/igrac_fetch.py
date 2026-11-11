"""Water levels for coastal monitoring wells in low- and middle-income countries from IGRAC GGIS (GGMN).
Per well: median depth to groundwater below ground surface since 2000, from 'Water depth [from the ground
surface]' records or, where only elevations are given, ground surface elevation minus water level elevation."""
import gzip
import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd

D = "/home/claude/data/igrac"
w = pd.read_csv(f"{D}/lmic_coastal_wells.csv", low_memory=False)
os.makedirs(f"{D}/levels", exist_ok=True)
T2000 = 946684800


def fetch(wid):
    p = f"{D}/levels/{wid}.json"
    if os.path.exists(p):
        return
    url = f"https://ggis.un-igrac.org/groundwater/record/{wid}/measurements/WellLevelMeasurement/data"
    for _ in range(3):
        try:
            b = urllib.request.urlopen(url, timeout=60).read()
            try:
                b = gzip.decompress(b)
            except OSError:
                pass
            open(p, "wb").write(b)
            return
        except Exception:
            continue


with ThreadPoolExecutor(4) as ex:
    list(ex.map(fetch, w.id.tolist()))

rec = []
for r in w.itertuples():
    p = f"{D}/levels/{r.id}.json"
    if not os.path.exists(p):
        continue
    try:
        data = json.load(open(p)).get("data", [])
    except Exception:
        continue
    df = pd.DataFrame(data)
    if df.empty:
        continue
    df = df[df.dt >= T2000]
    depth = df[df.par.str.contains("ground surface", case=False, na=False)].v
    src = "depth"
    if depth.empty:
        el = df[df.par.str.contains("a.m.s.l", case=False, na=False)].v
        if el.empty or not np.isfinite(r.ground_surface_elevation):
            continue
        depth, src = r.ground_surface_elevation - el, "elevation"
    rec.append({"id": r.id, "lon": r.lon, "lat": r.lat, "iso3": r.iso3, "income": r.income, "country": r.country,
                "dtw": float(np.median(depth)), "n_obs": int(depth.size), "source": src})
out = pd.DataFrame(rec)
out.to_csv(f"{D}/lmic_wells_dtw.csv", index=False)
print(len(out), "wells with depth since 2000;", out.country.value_counts().to_dict(), out.source.value_counts().to_dict())
print(out.dtw.describe().round(2))
