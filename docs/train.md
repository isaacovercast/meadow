# Training Utilities

`multispecies_resistance.train` converts per-species inputs into graph training records, prepares pairwise targets, and optimizes the multi-species resistance model with optional validation and early stopping.

## `SpeciesGraphData`
Dataclass that stores one species' graph features and pairwise training targets.

Fields:
- `name`: species name.
- `edge_index`: `E x 2` edge list.
- `edge_features`: `E x F` edge feature matrix.
- `node_coords`: `N x 2` node coordinates.
- `pair_i`, `pair_j`: node index arrays for target pairs.
- `pair_dist`: target genetic distances for pairs.
- `num_nodes`: node count.
- `val_pair_i`, `val_pair_j`, `val_pair_dist`: optional validation pair targets.

## `build_species_graphs(...)`
Builds `SpeciesGraphData` objects from species data using either per-species graphs or a shared mesh/global graph.

Parameters:
- `species_list`: list of species input records.
- `graph_type`: `"delaunay"`, `"knn"`, or `"dense_mesh"`.
- `k`: neighbor count for per-species knn graphs.
- `project_to`: optional projection CRS.
- `coord_order`: coordinate order.
- `coords_crs`: coordinate CRS.
- `standardize`: whether to standardize edge features across all species.
- `mesh_spacing_km` / `mesh_spacing_deg`: shared mesh spacing.
- `mesh_grid_type`: mesh node layout.
- `mesh_graph_type`: mesh edge builder (`"delaunay"` or `"knn"`).
- `mesh_k`: neighbor count for mesh knn.
- `mesh_coords`: optional precomputed mesh coordinates.
- `mesh_env`: optional environmental covariates on mesh nodes.
- `buffer_km`: mesh extent buffer.
- `bbox`: mesh clipping mode.
- `bbox_file`: polygon clipping file.
- `input_graph`: optional global GML graph path.

Returns:
- `graphs`: list of `SpeciesGraphData`.
- `stats`: `{mean, std}` if standardized, otherwise `None`.

## `prepare_pairs(distance_matrix)`
Converts a full pairwise distance matrix to unique upper-triangle pairs.

Parameters:
- `distance_matrix`: `N x N` symmetric matrix.

Returns:
- `pair_i`: row indices for unique pairs.
- `pair_j`: column indices for unique pairs.
- `pair_dist`: distances for those pairs.

## `split_pairs(pair_i, pair_j, pair_dist, num_nodes, val_fraction=0.2, strategy="site", seed=0, min_val_pairs=50)`
Splits pair targets into train/validation sets.

Parameters:
- `pair_i`, `pair_j`: pair index arrays.
- `pair_dist`: pair target distances.
- `num_nodes`: node count used by indices.
- `val_fraction`: validation fraction.
- `strategy`: `"site"` holdout or `"pair"` random holdout.
- `seed`: random seed.
- `min_val_pairs`: minimum pair count before accepting site-holdout split.

Returns:
- `train_pair_i`, `train_pair_j`, `train_pair_dist`.
- `val_pair_i`, `val_pair_j`, `val_pair_dist`.

## `train_model(...)`
Trains `MultiSpeciesResistanceModel` by minimizing squared error between observed genetic distances and modeled effective resistance distances.

Parameters:
- `species_graphs`: list of `SpeciesGraphData` training records.
- `hidden_dim`: model hidden width.
- `lr`: learning rate.
- `epochs`: maximum epochs.
- `l2_shared`: shared-logit regularization weight.
- `l2_species`: species-logit regularization weight.
- `log_every`: logging interval.
- `val_fraction`: validation split fraction if not already provided.
- `val_strategy`: validation split mode (`"site"` or `"pair"`).
- `min_val_pairs`: minimum validation pair count for site strategy.
- `patience`: early-stopping patience.
- `min_delta`: minimum validation improvement.
- `restore_best`: restore best validation weights on stop.
- `seed`: random seed for generated splits.

Returns:
- `model`: trained `MultiSpeciesResistanceModel`.
