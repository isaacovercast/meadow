# Graph Utilities

`meadow.graph` builds spatial graph topology and edge feature matrices from site coordinates and environmental covariates. It supports per-species graph construction as well as multiple shared mesh backends across species.

## `SpeciesGraph`
Dataclass that stores one species graph and its pairwise training targets.

Fields:

- `name`: species identifier.
- `edge_index`: `E x 2` edge list.
- `edge_features`: `E x F` edge feature matrix.
- `node_coords`: `N x 2` graph node coordinates.
- `sample_coords`: original observed sample coordinates.
- `pair_i`, `pair_j`, `pair_dist`: pairwise training targets.
- `num_nodes`: node count.
- `edge_nbr_i`, `edge_nbr_j`: precomputed neighboring-edge pairs for edge smoothing penalties.
- `edge_support_weight`: optional per-edge attenuation weight derived from distance to occupied nodes.
- `val_pair_i`, `val_pair_j`, `val_pair_dist`: optional validation targets.

## `SpeciesGraph.plot(edge_feature_idx=None, ...)`
Plots graph edges and sample locations, with optional edge coloring by a selected `edge_features` column index.

Parameters:

- `edge_feature_idx`: feature-column index for edge coloring, or `None` for a constant edge color.
- `basemap`: `True` for CartoDB Positron, `False` for no basemap, or a contextily provider object.
- `basemap_crs`: projected CRS used for basemap rendering.
- `coord_order`: coordinate order (`"latlon"` or `"lonlat"`).
- `coords_crs`: CRS of stored coordinates.
- style args: `sample_size`, `edge_width`, `edge_cmap`, `sample_color`, `sample_alpha`, `edge_alpha`, `edge_color`, `add_colorbar`, `title`.

Returns:

- `(ax, gdf_edges)`: Matplotlib axis and GeoDataFrame of edge geometries.

## `haversine_km(a, b)`
Computes great-circle distances in kilometers between paired geographic coordinates.

Parameters:

- `a`: array with shape `(..., 2)` in `lat, lon`.
- `b`: array with shape `(..., 2)` in `lat, lon`.

Returns:

- `distance`: array with shape `...`.

## `project_coords(coords, coord_order="latlon", coords_crs="EPSG:4326", target_crs="EPSG:3857")`
Projects coordinates from an input CRS to a target CRS.

Parameters:

- `coords`: `N x 2` coordinate array.
- `coord_order`: input order (`"latlon"` or `"lonlat"`).
- `coords_crs`: CRS of `coords`.
- `target_crs`: CRS for projected output.

Returns:

- `projected`: `N x 2` projected coordinates in `x, y`.

## `classify_land_points(coords, coord_order="latlon", coords_crs="EPSG:4326")`
Classifies point coordinates as terrestrial or non-terrestrial using a global land polygon dataset.

Parameters:

- `coords`: `N x 2` coordinate array.
- `coord_order`: input order (`"latlon"` or `"lonlat"`).
- `coords_crs`: CRS of `coords`.

Returns:

- `is_land`: length-`N` boolean array where `True` means the point falls on land.

## `apply_coastline_mask(node_coords, edge_index, mask_coastline="terrestrial", ...)`
Filters a graph to land-only or marine-only nodes and reindexes the surviving edges.

Parameters:

- `node_coords`: `N x 2` graph node coordinates.
- `edge_index`: `E x 2` graph edges.
- `mask_coastline`: either `"terrestrial"` or `"marine"`.
- `coord_order`: input coordinate order for `node_coords`.
- `coords_crs`: CRS of `node_coords`.

Returns:

- `node_coords`: filtered node coordinates in `lat, lon`.
- `edge_index`: reindexed edge list over the filtered nodes.

## `build_edge_neighbor_pairs(edge_index, num_nodes)`
Builds the neighboring-edge index pairs used when smoothing predicted edge logits during training.

Parameters:

- `edge_index`: `E x 2` edge list over graph nodes.
- `num_nodes`: total number of graph nodes referenced by `edge_index`.

Returns:

- `(edge_nbr_i, edge_nbr_j)`: parallel arrays of edge indices where each pair shares a node.

## `compute_edge_support_weight(node_coords, edge_index, occupied_nodes, support_decay_km, support_floor=0.01)`
Computes optional per-edge support weights by measuring graph distance from each node to the nearest occupied node and converting those distances into a smooth decay.

Parameters:

- `node_coords`: `N x 2` graph node coordinates in `lat, lon`.
- `edge_index`: `E x 2` edge list over graph nodes.
- `occupied_nodes`: node ids with observed samples.
- `support_decay_km`: positive decay scale in kilometers.
- `support_floor`: lower bound retained on very distant edges.

Returns:

- `edge_support_weight`: length-`E` attenuation values in `[support_floor, 1]`.

## `build_geodesic_mesh_graph(...)`
Constructs a shared geodesic triangular mesh from an icosphere and clips it to the study region while preserving native mesh adjacency.

Parameters:

- `coords_list`: list of per-species coordinate arrays.
- `spacing_km` / `spacing_deg`: target mesh spacing; `spacing_deg` is converted to an approximate kilometer spacing for compatibility.
- `grid_type`: accepted for compatibility; only `"triangular"` is supported.
- `project_to`: optional projected CRS used for clipping geometry.
- `coord_order`: coordinate order in `coords_list`.
- `coords_crs`: CRS of coordinates.
- `buffer_km`: extra geographic buffer before clipping.
- `bbox`: clipping mode (`"square"`, `"convex_hull"`, `"polygon"`, or `None`).
- `bbox_file`: polygon file path for `bbox="polygon"`.

Returns:

- `mesh_coords`: `M x 2` geodesic mesh node coordinates in `lat, lon`.
- `edge_index`: `E x 2` native geodesic edge list derived from triangle faces.

## `build_dggrid_mesh_graph(...)`
Constructs a shared triangular mesh by calling an external DGGRID-style command-line tool and converting its output cell polygons into a vertex-edge graph of triangle sides.

Parameters:

- `coords_list`: list of per-species coordinate arrays.
- `spacing_km` / `spacing_deg`: target mesh spacing; `spacing_deg` is converted to an approximate kilometer spacing for compatibility.
- `grid_type`: accepted for compatibility; only `"triangular"` is supported.
- `project_to`: optional projected CRS used for clipping geometry.
- `coord_order`: coordinate order in `coords_list`.
- `coords_crs`: CRS of coordinates.
- `buffer_km`: extra geographic buffer before clipping.
- `bbox`: clipping mode (`"square"`, `"convex_hull"`, `"polygon"`, or `None`).
- `bbox_file`: polygon file path for `bbox="polygon"`.

Returns:

- `mesh_coords`: `M x 2` DGGRID vertex coordinates in `lat, lon`.
- `edge_index`: `E x 2` edges running along triangle sides.

## `edge_features(site_coords, site_env, edge_index)`
Builds edge-level feature vectors from spatial distance and environmental differences.

Parameters:

- `site_coords`: `S x 2` site coordinates in `lat, lon`.
- `site_env`: `S x K` site environmental matrix (or empty).
- `edge_index`: `E x 2` edge list.

Returns:

- `features`: `E x (1 + K)` matrix where column 0 is distance (km).

## `standardize_features(x)`
Standardizes feature columns to zero mean and unit variance.

Parameters:

- `x`: `N x K` feature matrix.

Returns:

- `x_std`: standardized features.
- `mean`: per-column mean.
- `std`: per-column standard deviation (zeros replaced with ones).
