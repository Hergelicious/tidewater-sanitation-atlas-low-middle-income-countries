"""Supplementary tables S1 (countries), S2 (scenarios) and the open city-level atlas table."""
import numpy as np
import pandas as pd
import geopandas as gpd

A = "/home/claude/atlas/"
d = pd.read_csv(A + "atlas_cities.csv")
M = lambda col, x=d: round(x[col].sum() / 1e6, 1)  # noqa: E731

g = d.groupby(["iso3", "GC_CNT_GAD_2025"]).agg(
    cities=("ID_UC_G0", "size"), coastal_zone_pop_2020=("pop_within10km", "sum"), urban_sewer_pct=("sewer_pct", "first"),
    sewer_source=("sewer_source", "first"), nonsewered_wet_now=("ns_wet10_now", "sum"),
    nonsewered_wet_now_low=("ns_wet10_low", "sum"), nonsewered_wet_now_high=("ns_wet10_high", "sum"),
    nonsewered_tidal_now=("ns_tidal_now", "sum"), nonsewered_river_now=("ns_river_10", "sum"),
    nonsewered_surge_map=("ns_vous_10", "sum"), nonsewered_wet_2050=("ns50_wet10_85", "sum"),
    nonsewered_wet_2050_sewer_trend=("ns50t_wet10_85", "sum"),
    fish_share_animal_protein_pct=("fish_share_animal_protein_pct", "first")).reset_index()
g = g.rename(columns={"GC_CNT_GAD_2025": "country"})
g["country"] = [c.encode("latin-1").decode("utf-8") if "Ã" in c else c for c in g.country]
for c in g.columns:
    if c.startswith(("nonsewered", "coastal")):
        g[c] = (g[c] / 1e3).round(1)
g = g.rename(columns={c: c + "_thousands" for c in g.columns if c.startswith(("nonsewered", "coastal"))})
g["fish_share_animal_protein_pct"] = g.fish_share_animal_protein_pct.round(1)
g.sort_values("nonsewered_wet_now_thousands", ascending=False).to_csv(A + "supplementary/Table_S1_countries.csv", index=False)

rows = [
    ("Central estimate, 2020 (elevation referenced to mean sea level, 2020 sanitation)", M("ns_wet10_now")),
    ("Assumption range, low (elevation +0.45 m, sea level 5th percentile, threshold HAT + 2 m)", M("ns_wet10_low")),
    ("Assumption range, high (elevation -0.45 m, sea level 95th percentile, threshold HAT + 4 m)", M("ns_wet10_high")),
    ("Tidal only (coastal flood or groundwater contact), central", M("ns_tidal_now")),
    ("Tidal only, low bound", M("ns_tidal_only_low")), ("Tidal only, high bound", M("ns_tidal_only_high")),
    ("River flood within coastal zone", M("ns_river_10")),
    ("Elevation left on the EGM2008 geoid (no sea-level referencing)", M("ns_wet10_egm")),
    ("Land a further 0.3 m lower", M("ns_wet10_datum03")),
    ("Latest JMP sanitation (2024 for most countries)", M("ns24_wet10_now")),
    ("Groundwater contact threshold HAT + 1 m", M("ns_wet10_gw1")),
    ("Potentially inadequate separation, threshold HAT + 5 m", M("ns_wet10_gw5")),
    ("Potential groundwater contact only (potentially persistent)", M("ns_gw3_now")),
    ("100-year flooding only (episodic)", round((d.ns_wet10_now - d.ns_gw3_now).clip(lower=0).sum() / 1e6, 1)),
    ("Coastal zone 5 km", M("ns_wet5km")), ("Coastal zone 20 km", M("ns_wet20km")),
    ("Hydrodynamic surge map as coastal flood layer", M("ns_wet10_surgemodel")),
    ("127 estuary and delta cities removed", M("ns_wet10_now", d[~d.estuarine])),
    ("Countries with imputed sewer share removed", M("ns_wet10_now", d[~d.sewer_imputed])),
    ("Coastal flood layer removed (overlap check)", M("ns_def_now")),
    ("Storm-surge map only", M("ns_vous_10")),
    ("Storm-surge map only, estuary and delta cities removed", M("ns_vous_10", d[~d.estuarine])),
    ("2050 RCP8.5, 2020 population", M("ns_wet10_85")),
    ("2050 RCP8.5, 2050 population, constant sewer share", M("ns50_wet10_85")),
    ("2050 RCP8.5, 2050 population, low bound", M("ns50_wet10_85_low")),
    ("2050 RCP8.5, 2050 population, high bound", M("ns50_wet10_85_high")),
    ("2050 RCP8.5, 2050 population, sewer trend continued", M("ns50t_wet10_85")),
    ("2050 RCP8.5, sewer trend continued, low bound", M("ns50t_wet10_85_low")),
    ("2050 RCP8.5, sewer trend continued, high bound", M("ns50t_wet10_85_high")),
]
s = pd.DataFrame(rows, columns=["scenario", "nonsewered_residents_on_wet_ground_millions"])
s.to_csv(A + "supplementary/Table_S2_sensitivity.csv", index=False)
print(s.to_string(index=False))

c = gpd.read_file(A + "cities.gpkg").to_crs(3857)
cent = c.geometry.centroid.to_crs(4326)
d = d.merge(pd.DataFrame({"ID_UC_G0": c.ID_UC_G0, "lon": cent.x.round(4), "lat": cent.y.round(4)}), on="ID_UC_G0")
cols = {"ID_UC_G0": "ghsl_id", "GC_UCN_MAI_2025": "city", "GC_CNT_GAD_2025": "country", "iso3": "iso3",
        "GC_DEV_USR_2025": "sdg_region", "GC_DEV_WIG_2025": "income_group", "lon": "lon", "lat": "lat",
        "estuarine": "estuary_or_delta_city", "pop_grid": "population_2020", "pop_within10km": "coastal_zone_population_2020",
        "sewer_pct": "urban_sewer_connection_pct", "sewer_source": "sewer_share_source",
        "ns_wet10_now": "nonsewered_wet_ground_now", "ns_wet10_low": "nonsewered_wet_ground_now_low",
        "ns_wet10_high": "nonsewered_wet_ground_now_high", "ns_coast_now": "nonsewered_coastal_flood_now",
        "ns_gw3_now": "nonsewered_groundwater_contact_now", "ns_river_10": "nonsewered_river_flood_now",
        "ns_vous_10": "nonsewered_surge_map_now", "pop50_grid": "population_2050",
        "ns50_wet10_85": "nonsewered_wet_ground_2050", "ns50_wet10_85_low": "nonsewered_wet_ground_2050_low",
        "ns50_wet10_85_high": "nonsewered_wet_ground_2050_high", "ns50t_wet10_85": "nonsewered_wet_ground_2050_sewer_trend",
        "fish_share_animal_protein_pct": "country_fish_share_animal_protein_pct"}
t = d[list(cols)].rename(columns=cols)
t["nonsewered_flood_only_now"] = (t.nonsewered_wet_ground_now - t.nonsewered_groundwater_contact_now).clip(lower=0)
t["share_of_nonsewered_on_wet_ground"] = (d.ns_wet10_now / d.ns_within10km.where(d.ns_within10km > 0)).round(3)
t["city"] = [x.encode("latin-1").decode("utf-8") if isinstance(x, str) and "Ã" in x else x for x in t.city]
t["country"] = [x.encode("latin-1").decode("utf-8") if isinstance(x, str) and "Ã" in x else x for x in t.country]
for k in t.columns:
    if k.startswith(("nonsewered", "population", "coastal_zone")) and k != "share_of_nonsewered_on_wet_ground":
        t[k] = t[k].round(0).astype(int)
t["urban_sewer_connection_pct"] = t.urban_sewer_connection_pct.round(1)
t["country_fish_share_animal_protein_pct"] = t.country_fish_share_animal_protein_pct.round(1)
t.sort_values("nonsewered_wet_ground_now", ascending=False).to_csv(A + "Tidewater_Sanitation_Atlas_cities_v0.csv", index=False)
print(t.shape)
