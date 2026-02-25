# I/O Utilities

Source: `/Users/isaac/src/meems/meems-from-scratch/src/multispecies_resistance/io.py`

## PEDIC FEEMS Inputs

Expected files per species:
- `<name>_feems_coords.txt` (lon lat per sample)
- `<name>_feems_genos.npy` (genotype matrix)

## `list_pedic_species(root)`
Scans a directory and returns species names where both coordinate and genotype files exist.

## `load_pedic_species(...)`
Loads multiple species from PEDIC FEEMS-style files.

Behavior:
- Converts sample coordinates from `lon,lat` to `lat,lon`.
- Builds pseudo-sites via `build_pseudosites(...)`.
- Optionally samples raster covariates for pseudo-sites.
- Returns `(species_list, env_names)`.

Raster options:
- pass explicit `raster_paths`, or
- pass `raster_root` + `raster_pattern` + `raster_recursive`.
