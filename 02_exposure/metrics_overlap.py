"""Population in each combination of the three wet-ground mechanisms (coastal flood C, groundwater contact G, river flood R).
Writes city_overlap.csv."""
import numpy as np, pandas as pd
from multiprocessing import Pool
import metrics as M
from metrics import bathtub, coast_masks, ocean_seeds, load, grids, cities, esl, grid_transform, geometry_mask


def city(uid):
    g = grids[uid]; shape = (g["h"], g["w"])
    inside = ~geometry_mask([cities.loc[uid, "geometry"]], out_shape=shape, transform=grid_transform(g))
    pop = np.nan_to_num(load(uid, "pop", shape)) * inside
    dtm = load(uid, M.DTMNAME, shape); sea = np.nan_to_num(load(uid, "sea", shape)) >= 0.5
    riv = np.nan_to_num(load(uid, "riv", shape)); d = np.where(np.isfinite(dtm), dtm, 9999.0)
    e = esl.loc[uid]; near, ocean = coast_masks(uid, g, shape); seeds = ocean_seeds(sea, ocean)
    land, n10 = ~sea, near[10]
    C = bathtub(d, seeds, e.esl_base, n10) & land
    G = land & n10 & (d <= e.hat_base + 3)
    R = land & (riv > 0) & (riv < 100) & n10
    o = {"ID_UC_G0": uid}
    for name, m in {"C_only": C & ~G & ~R, "G_only": G & ~C & ~R, "R_only": R & ~C & ~G, "CG": C & G & ~R,
                    "GR": G & R & ~C, "CR": C & R & ~G, "CGR": C & G & R}.items():
        o["pop_" + name] = float(pop[m].sum())
    return o


if __name__ == "__main__":
    with Pool(2) as p:
        out = pd.DataFrame(p.map(city, list(grids), chunksize=8))
    out.to_csv("city_overlap.csv", index=False)
    a = pd.read_csv("atlas_cities.csv")[["ID_UC_G0", "nonsewer_frac", "ns_wet10_now"]].merge(out, on="ID_UC_G0")
    r = {k[4:]: round((a[k] * a.nonsewer_frac).sum() / 1e6, 2) for k in out.columns if k.startswith("pop_")}
    print(r, "sum", round(sum(r.values()), 2), "central", round(a.ns_wet10_now.sum() / 1e6, 2))
    pd.Series(r, name="nonsewered_millions").to_csv("supplementary/Table_S18_mechanism_overlap.csv")
