# Training Utilities

`multispecies_resistance.train` converts sample-level `SpeciesData` inputs into graph training records, including sample-to-node assignment, optional raster covariate sampling at graph nodes, and model optimization.

## Graph Output Type
`build_species_graphs(...)` returns `SpeciesGraph` objects (defined in `multispecies_resistance.graph`), each carrying:
- `name`: species identifier.
- `node_coords`: graph node coordinates.
- `sample_coords`: original observed sample coordinates.
- `edge_index` and `edge_features`: edge topology and edge covariates.
- `edge_nbr_i` and `edge_nbr_j`: neighboring-edge pairs used for optional smoothing penalties.
- `pair_i`, `pair_j`, `pair_dist`: pairwise training targets.
- optional validation pairs (`val_pair_i`, `val_pair_j`, `val_pair_dist`).

## `build_species_graphs(...)`
Builds graph training data from sample-level species records.

Key behavior:

- `input_graph=None`: builds a shared dense mesh and assigns each sample to the nearest mesh node.
- `input_graph=...`: uses provided global graph nodes and assigns each sample to the nearest graph node.
- `mesh_grid_type`: controls the layout of the default shared mesh.
- `mesh_spacing_km=None`: automatically picks a spacing from nearest-neighbor sample distances.

Environmental extraction:

- pass `raster_paths` or `raster_root` to sample node covariates during graph construction.
- rasters are sampled at the exact node coordinates used for training.

Returns:

- `graphs`: list of `SpeciesGraph`.
- `stats`: `{mean, std}` if edge standardization is enabled, else `None`.

## `choose_mesh_spacing_km(...)`
Chooses a default mesh spacing from per-species nearest-neighbor sample distances.

Behavior:

- projects sample coordinates into a planar CRS,
- computes nearest-neighbor distances within each species,
- pools those distances,
- returns a robust spacing estimate in kilometers.

## `prepare_pairs(distance_matrix)`
Converts a full pairwise distance matrix into unique upper-triangle pair arrays.

## `split_pairs(...)`
Splits pair targets into training/validation sets using site holdout or random pair holdout.

## `train_model(...)`
Trains `MultiSpeciesResistanceModel` with optional validation splits and early stopping.

Key parameters:

- `l2_shared`, `l2_species`: quadratic penalties on shared and species-specific edge logits.
- `edge_smoothing`: value in `[0, 1]` that increasingly penalizes differences between logits on edges that share a node.
- `val_fraction`, `val_strategy`, `patience`, `min_delta`, `restore_best`: validation-split and early-stopping controls.
