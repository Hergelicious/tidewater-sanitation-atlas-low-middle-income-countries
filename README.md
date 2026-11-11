# Tidewater Sanitation Atlas

Code and city-level data for *Shallow groundwater reveals overlooked coastal sanitation vulnerability in low- and middle-income countries* (Hassan, 2026).

The atlas maps "wet ground" in the 10 km coastal zones of 1,225 urban centres in 88 low- and middle-income countries: cells reached by the 100-year coastal flood, by the 100-year river flood, or lying within 3 m of the highest astronomical tide (potential groundwater contact with a typical pit), and counts the residents without a sewer connection who live there.

## Contents

| Folder | What it does |
|---|---|
| `01_inputs/` | Selects the urban centres and their coastal zones, extracts every input layer onto the GHS-POP 3 arc-second grid (`pipeline.py`), and prepares the per-city sea-level, tide and sanitation inputs. |
| `02_exposure/` | Computes the wet-ground layers and all exposure counts per city (`metrics.py` and its extensions), merges them with covariates (`assemble.py`), and writes the main and supplementary exposure tables. |
| `03_datum/` | References DeltaDTM to local mean sea level with the DTU22MDT mean dynamic topography and the permanent-tide correction, and compares the result with the Seeger and Minderhoud tiles. |
| `04_groundwater_validation/` | Tests the tidal rule at coastal monitoring wells (Jasechko et al., IGRAC GGMN, CGWL), by country, distance from the coast and season; the gradient-boosting comparison; the Fan et al. water-table comparison; and the published-site checks. |
| `05_sanitation_and_uncertainty/` | Sanitation technology split, septic-tank discharge from Shit Flow Diagram reports, soil-dependent counts and rankings, one-at-a-time and joint (Monte Carlo) uncertainty, and the adaptation targeting comparison. |
| `06_household_surveys/` | Links DHS clusters to wet ground and fits the exposure-gradient, health and within-city sanitation models. These scripts need DHS microdata, which are not included (see below). |
| `data/Tidewater_Sanitation_Atlas_cities_v0.csv` | The released atlas: 1,225 urban centres, 28 variables (data dictionary below). |
| `data/supplementary_tables/` | All 35 Supplementary Tables as CSV, numbered as in the paper (`00_table_mapping.csv` lists the scripts' internal numbers). |
| `data/sfd_containment.csv`, `data/sfd_city_psoil.csv` | Containment shares extracted from 29 city Shit Flow Diagram reports, with the report URLs, and the derived septic-to-soil shares for the 21 cities that report them. |

## Running the pipeline

Order: `01_inputs/pipeline.py` (downloads and tiles the inputs; several hundred GB of rasters, hours to days), `01_inputs/city_esl.py`, `01_inputs/jmp_2020.py`, `03_datum/datum_dtu.py`, then `02_exposure/metrics.py`, `metrics_extra.py`, `metrics_overlap.py`, `metrics_v07.py`, `02_exposure/assemble.py`, and the analysis scripts in any order. Each script states in its docstring what it reads and writes.

The scripts were run with Python 3.13 and the package versions in `requirements.txt`. Paths are absolute under `/home/claude/` (inputs in `data/`, outputs in `atlas/`); run `bash set_root.sh /your/root` to rewrite them for another machine. The environment variable `DTMNAME` selects the elevation layer (`dtmmsl`, the default, is DeltaDTM referenced to mean sea level; `dtm` leaves it on the EGM2008 geoid). `MCVAR` selects the variant of the joint uncertainty analysis (`base`, `corr`, `fixedp`).

## Input data

All inputs are public and are documented, with licences, in the paper's Data availability statement: GHSL (UCDB R2024A, GHS-POP R2023A, GHS-WUP R2025A), DeltaDTM v1.1, DTU22MDT, JRC extreme sea-level projections and coastal flood maps, GloFAS river flood maps v2.1.2, JMP household estimates, Jasechko et al. (2024) groundwater levels, IGRAC GGMN, the CGWL compilation (CC BY-NC 4.0; derived statistics only are published here), the Fan et al. (2017) water table, Oelsmann et al. (2026) vertical land motion, World Bank WDI and income classification, Natural Earth, geoBoundaries and OpenStreetMap place names. Well records and DHS microdata are not redistributed.

DHS files (Household Recode, Children's Recode and GPS files for Bangladesh 2022, Côte d'Ivoire 2021, Ghana 2022, Haiti 2016–17, Mozambique 2022–23, Nigeria 2023–24, Philippines 2022 and Tanzania 2022) must be requested from the DHS Program and placed under `dhs_private/{hr,kr,ge}/<survey>/`. The scripts in `06_household_surveys/` write aggregated results only.

## Data dictionary: `Tidewater_Sanitation_Atlas_cities_v0.csv`

Counts are residents in 2020 unless stated; "non-sewered" applies the national urban sewer share (JMP 2020).

| Column | Meaning |
|---|---|
| `ghsl_id` | GHSL Urban Centre Database R2024A identifier |
| `city`, `country`, `iso3` | Main city name, country, ISO3 code |
| `sdg_region` | SDG region |
| `income_group` | World Bank income group (FY2027) |
| `lon`, `lat` | Population-weighted centre of the coastal zone |
| `estuary_or_delta_city` | 1 if the coastal-zone centre lies more than 40 km from the generalised 1:110 m coastline |
| `population_2020` | Residents of the urban centre |
| `coastal_zone_population_2020` | Residents within 10 km of the coastline |
| `urban_sewer_connection_pct`, `sewer_share_source` | JMP urban sewer share used and its source (urban, national or regional median) |
| `nonsewered_wet_ground_now` | Non-sewered residents on wet ground (central estimate) |
| `nonsewered_wet_ground_now_low`, `_high` | Assumption range (elevation error, sea-level percentiles, groundwater threshold) |
| `nonsewered_coastal_flood_now` | Reached by the 100-year coastal flood (bathtub, connected) |
| `nonsewered_groundwater_contact_now` | On ground within 3 m of the highest astronomical tide |
| `nonsewered_river_flood_now` | Reached by the GloFAS 100-year river flood |
| `nonsewered_flood_only_now` | Reached by coastal or river flooding but not by groundwater contact |
| `nonsewered_surge_map_now` | On the published 90 m storm-surge map (Vousdoukas) |
| `share_of_nonsewered_on_wet_ground` | Non-sewered residents on wet ground divided by non-sewered coastal-zone residents |
| `population_2050` | Coastal-zone residents in 2050 (GHS-WUP) |
| `nonsewered_wet_ground_2050`, `_low`, `_high` | 2050, RCP8.5 sea-level rise, 2050 population, 2020 sewer share |
| `nonsewered_wet_ground_2050_sewer_trend` | 2050 with the 2015 to 2024 sewer trend continued |
| `country_fish_share_animal_protein_pct` | Fish share of animal protein supply (gcamfaostat), country level |

## Licence and citation

Code: MIT. The atlas table and supplementary tables: CC BY 4.0. Please cite the paper and, for the inputs, the original datasets listed in its Data availability statement.
