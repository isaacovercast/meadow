# I/O Utilities

`multispecies_resistance.io` provides dataset loading helpers for PEDIC FEEMS-style files and optional raster covariate sampling. It is the main entrypoint for constructing `SpeciesData` objects directly from files on disk.

## PEDIC File Convention
Each species should include:
- `<name>_feems_coords.txt`: sample coordinates in `lon lat` order.
- `<name>_feems_genos.npy`: sample-by-marker genotype matrix.

## `list_pedic_species(root)`
Finds species names with both required PEDIC files present.

Parameters:
- `root`: directory containing PEDIC files.

Returns:
- `names`: sorted species names with complete file pairs.

## `load_pedic_species(...)`
Loads one or more species, builds pseudo-sites, and optionally samples site-level raster covariates.

Parameters:
- `root`: directory containing PEDIC files.
- `species_names`: explicit species list (uses all if omitted).
- `spacing_km` / `spacing_deg`: pseudo-site spacing passed to `build_pseudosites`.
- `raster_paths`: explicit rasters to sample for site covariates.
- `raster_root`: raster directory/glob root when `raster_paths` is omitted.
- `raster_pattern`: glob pattern for raster discovery.
- `raster_recursive`: recursive raster search toggle.
- `coords_crs`: CRS of coordinates for raster sampling.
- `raster_fill_method`: nodata fill mode (`"nan"`, `"mean"`, `"nearest"`).
- `mmap_mode`: optional NumPy memory-map mode for genotype loading.

Returns:
- `species_list`: list of `SpeciesData` entries.
- `env_names`: environmental feature names from sampled rasters.
