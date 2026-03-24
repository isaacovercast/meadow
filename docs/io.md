# I/O Utilities

`multispecies_resistance.io` provides sample-level loading for PEDIC FEEMS-style files.

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

## `load_pedic_species(root, species_names=None, mmap_mode=None)`
Loads one or more PEDIC species as sample-level `SpeciesData` records.

Behavior:
- reads sample coordinates and genotypes,
- converts coordinates from `lon,lat` to `lat,lon`,
- returns `SpeciesData(name, genotypes, sample_coords)` objects.

Parameters:

- `root`: directory containing PEDIC files.
- `species_names`: optional subset of species names.
- `mmap_mode`: optional NumPy memory-map mode for genotype loading.

Returns:

- `species_list`: list of sample-level `SpeciesData` records.
