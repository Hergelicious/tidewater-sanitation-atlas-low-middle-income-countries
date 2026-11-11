import numpy as np
import geopandas as gpd

def vertices(path, step_km=0.2):
    g = gpd.read_file(path).to_crs(4326)
    pts = []
    for geom in g.geometry.explode(index_parts=False):
        x, y = np.asarray(geom.coords.xy)
        for i in range(len(x) - 1):
            n = max(1, int(np.hypot((x[i + 1] - x[i]) * 111 * np.cos(np.radians(y[i])), (y[i + 1] - y[i]) * 111) / step_km))
            t = np.linspace(0, 1, n, endpoint=False)
            pts.append(np.c_[y[i] + t * (y[i + 1] - y[i]), x[i] + t * (x[i + 1] - x[i])])
    return np.vstack(pts)



_V = {}

def v110():
    if "v" not in _V:
        _V["v"] = vertices("/home/claude/data/ne/ne_110m_coastline.shp", 1.0)
    return _V["v"]
