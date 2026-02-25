# Graph Utilities

`multispecies_resistance.graph` builds spatial graph topology and edge feature matrices from site coordinates and environmental covariates. It supports per-species graph construction as well as shared dense mesh generation across species.

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

## `build_knn_graph(site_coords, k=6, project_to=None, coord_order="latlon", coords_crs="EPSG:4326")`
Builds an undirected k-nearest-neighbor graph.

Parameters:
- `site_coords`: `S x 2` site coordinates.
- `k`: neighbors per node.
- `project_to`: optional CRS for distance computation.
- `coord_order`: coordinate order of `site_coords`.
- `coords_crs`: CRS of `site_coords`.

Returns:
- `edge_index`: `E x 2` undirected edges with sorted node indices.

## `build_delaunay_graph(site_coords, project_to=None, coord_order="latlon", coords_crs="EPSG:4326")`
Builds an undirected graph from Delaunay triangulation edges.

Parameters:
- `site_coords`: `S x 2` site coordinates.
- `project_to`: optional CRS for triangulation.
- `coord_order`: coordinate order of `site_coords`.
- `coords_crs`: CRS of `site_coords`.

Returns:
- `edge_index`: `E x 2` undirected edges with sorted node indices.

## `build_dense_mesh_graph(...)`
Constructs a shared mesh covering all species and returns mesh nodes plus graph edges.

Parameters:
- `coords_list`: list of per-species coordinate arrays.
- `spacing_km` / `spacing_deg`: mesh node spacing (provide one).
- `grid_type`: `"triangular"` or `"rect"` node placement.
- `mesh_graph_type`: `"delaunay"` or `"knn"` edge construction.
- `k`: neighbor count when using `"knn"`.
- `project_to`: optional CRS for graph construction.
- `coord_order`: coordinate order in `coords_list`.
- `coords_crs`: CRS of coordinates.
- `buffer_km`: extra geographic buffer before meshing.
- `bbox`: clipping mode (`"square"`, `"convex_hull"`, `"polygon"`, or `None`).
- `bbox_file`: polygon file path for `bbox="polygon"`.

Returns:
- `mesh_coords`: `M x 2` mesh node coordinates.
- `edge_index`: `E x 2` mesh edge list.

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
