# Plan: Implement `build_geodesic_mesh_graph(...)` with `trimesh`

## 1. Goal
Implement a new `build_geodesic_mesh_graph(...)` function in `src/multispecies_resistance/graph.py` using `trimesh` so it can act as a drop-in replacement for `build_dense_mesh_graph(...)`.

This plan is explicitly focused on compatibility with the current codebase. The new function should preserve the downstream expectations already relied on by:
- `build_species_graphs(...)`
- edge feature construction
- sample-to-node assignment
- plotting and graph export
- CV fold construction on shared graphs

No implementation is performed in this plan. This file specifies the exact changes to make.

## 2. Compatibility Target
The new function should match the current dense-mesh builder in the ways that matter to the rest of the package.

### 2.1 Function contract
`build_geodesic_mesh_graph(...)` should return exactly:
- `mesh_coords`: `N x 2` array in `lat, lon`
- `edge_index`: `E x 2` undirected integer edge list with node indices into `mesh_coords`

### 2.2 Input compatibility
The function signature should be compatible with current `build_dense_mesh_graph(...)` usage, meaning it should accept:
- `coords_list`
- `spacing_km`
- `spacing_deg`
- `grid_type`
- `project_to`
- `coord_order`
- `coords_crs`
- `buffer_km`
- `bbox`
- `bbox_file`

Not all of these need to affect the geodesic mesh in the same way they do now, but the function should accept them so it can be swapped into existing call sites with minimal disruption.

### 2.3 Behavioral compatibility
The new function should preserve these expectations:
- all returned node coordinates are in `lat, lon`
- all graph edges are spatially local and undirected
- all node/edge arrays are NumPy arrays with stable dtypes
- clipping to `square`, `convex_hull`, or `polygon` remains supported
- graph construction works from the union of all species sample coordinates
- downstream code does not need to know whether the mesh came from a local lattice or a geodesic mesh

## 3. Design Summary
The current dense-mesh builder:
1. creates local lattice nodes over a bounding box,
2. clips nodes,
3. reconstructs adjacency with Delaunay,
4. filters long edges.

The new geodesic builder should instead:
1. construct a global or sufficiently dense geodesic triangular mesh from `trimesh.creation.icosphere(...)`,
2. derive native adjacency directly from triangle faces,
3. clip nodes to the study region,
4. retain only native edges whose endpoints survive clipping,
5. optionally keep the largest connected component.

This avoids Delaunay entirely and preserves native geodesic mesh topology.

## 4. Detailed Implementation Plan

## 4.1 Add `trimesh` dependency
Files to update:
- `environment.yml`
- any packaging/dependency metadata if present

Planned change:
- add `trimesh` as a direct dependency

Reason:
- the new mesh builder will rely on `trimesh.creation.icosphere(...)` as the source of vertices and triangle faces

Validation:
- importing `trimesh` should succeed in the project environment

## 4.2 Add new helper functions to `src/multispecies_resistance/graph.py`
The new mesh builder should be composed from small helpers so its behavior is easy to test and reason about.

### A) `_cartesian_to_latlon(vertices)`
Purpose:
- convert unit-sphere Cartesian coordinates from the icosphere to `lat, lon`

Inputs:
- `vertices`: `N x 3`

Outputs:
- `coords_latlon`: `N x 2`

Implementation notes:
- `lat = degrees(arcsin(z / r))`
- `lon = degrees(arctan2(y, x))`
- for unit sphere, `r = 1`, but computing `r` explicitly is safer

### B) `_edge_index_from_faces(faces)`
Purpose:
- build unique undirected edges from triangular faces

Inputs:
- `faces`: `F x 3`

Outputs:
- `edge_index`: `E x 2`

Implementation notes:
- each face contributes `(a, b)`, `(b, c)`, `(a, c)`
- sort endpoint indices within each edge
- deduplicate globally
- return sorted integer array

### C) `_choose_icosphere_subdivision_for_spacing(spacing_km)`
Purpose:
- map requested `spacing_km` to an icosphere subdivision level

Inputs:
- `spacing_km`

Outputs:
- integer subdivision level
- optionally also actual median edge length at that level

Implementation notes:
- try a fixed set of subdivision levels, e.g. `0..7`
- for each level:
  - build temporary icosphere
  - compute edge list from faces
  - convert vertices to `lat, lon`
  - compute median great-circle edge length using `haversine_km(...)`
- pick the level whose median edge length is closest to `spacing_km`
- bias toward the coarsest acceptable level if two are equally close, to control graph size

Important compatibility note:
- if `spacing_km` is currently the user's main resolution control, the new builder should continue to honor it rather than forcing callers to think in terms of subdivision level

### D) `_study_region_geometry(...)`
Purpose:
- build the clipping geometry used to subset the geodesic mesh, with behavior matching current `bbox` handling

Inputs:
- `all_coords_latlon`
- `buffer_km`
- `bbox`
- `bbox_file`
- `coords_crs`

Outputs:
- region geometry in projected coordinates
- possibly also projected CRS used for clipping

Implementation notes:
- retain the current behaviors:
  - `bbox="square"`: rectangular sample bounding box plus buffer
  - `bbox="convex_hull"`: convex hull of sample locations plus buffer
  - `bbox="polygon"`: polygon read from `bbox_file`
- use projected planar geometry for clipping, as the current implementation already does
- preserve the current error behavior for malformed polygon files

### E) `_clip_graph_to_region(mesh_coords, edge_index, region, coords_crs)`
Purpose:
- subset nodes and edges to the requested study region without changing mesh topology

Inputs:
- `mesh_coords`: `N x 2` in `lat, lon`
- `edge_index`: `E x 2`
- `region`: clipping geometry
- `coords_crs`

Outputs:
- clipped `mesh_coords`
- clipped and reindexed `edge_index`

Implementation notes:
- project nodes to the region CRS
- keep nodes within or touching the region
- build old-to-new node index map
- keep only edges whose two endpoints survive
- remap surviving edge indices into compact node numbering

Important design point:
- do not add any new edges during clipping
- do not call Delaunay after clipping

### F) `_largest_connected_component(mesh_coords, edge_index)`
Purpose:
- remove small disconnected graph fragments caused by clipping

Inputs:
- `mesh_coords`
- `edge_index`

Outputs:
- component-filtered `mesh_coords`
- component-filtered `edge_index`

Implementation notes:
- compute connected components on the node-edge graph
- keep the largest component only
- reindex nodes after component filtering

This should likely be enabled by default because polygon or hull clipping may leave tiny isolated slivers that are not useful downstream.

## 4.3 Implement `build_geodesic_mesh_graph(...)` in `src/multispecies_resistance/graph.py`
Planned public signature:

```python
def build_geodesic_mesh_graph(
    coords_list: list[np.ndarray],
    spacing_km: float | None = 50.0,
    spacing_deg: float | None = None,
    grid_type: str = "triangular",
    project_to: str | CRS | None = None,
    coord_order: str = "latlon",
    coords_crs: str | CRS | None = "EPSG:4326",
    buffer_km: float = 0.0,
    bbox: str | None = "square",
    bbox_file: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    ...
```

### A) Preserve unused compatibility parameters where necessary
- `grid_type` should be accepted even if only triangular geodesic meshes are supported
- for simplicity and correctness, reject unsupported values explicitly:
  - allow `grid_type="triangular"`
  - raise for `grid_type="rect"`

Reason:
- this keeps the signature compatible while making the geodesic semantics explicit

### B) Preserve `spacing_deg` in the signature
- geodesic meshes should fundamentally be driven by `spacing_km`, not `spacing_deg`
- for drop-in compatibility, accept `spacing_deg` but handle it explicitly

Planned behavior:
- if `spacing_km` is provided, use it
- if only `spacing_deg` is provided, convert it to an approximate kilometer spacing using the sample mean latitude, mirroring the current approximation style
- if both are provided, raise just as current code does
- if neither is provided, use the default `spacing_km`

### C) Use all species coordinates to define the region
- stack all `coords_list` inputs
- normalize into `lat, lon` based on `coord_order`
- use these coordinates only for:
  - choosing the clipping region
  - choosing effective mesh resolution

### D) Build the raw geodesic mesh
- determine subdivision level from requested spacing
- call `trimesh.creation.icosphere(...)`
- convert vertices to `lat, lon`
- derive native undirected edges from faces

### E) Clip the raw mesh to the study region
- build region geometry from current `bbox` logic
- clip nodes and inherited edges
- if resulting graph is empty, raise a clear error analogous to the current dense-mesh builder

### F) Keep the largest connected component
- after clipping, reduce to the largest connected component
- raise if no valid component remains

### G) Return only `mesh_coords` and `edge_index`
- do not return faces or extra metadata
- keep the function contract identical to `build_dense_mesh_graph(...)`

## 4.4 Keep `build_dense_mesh_graph(...)` intact initially
To minimize disruption, the first implementation should not immediately replace current behavior everywhere.

Planned approach:
- add `build_geodesic_mesh_graph(...)` as a new function first
- leave `build_dense_mesh_graph(...)` unchanged during the first implementation pass
- once validated, optionally make `build_dense_mesh_graph(...)` delegate to the geodesic builder or add a mesh-mode selector in a later pass

Reason:
- this allows direct testing of the new builder without destabilizing the current pipeline immediately
- it is the safest path toward a later drop-in swap

## 4.5 Optional second-stage compatibility swap
After validating the new function, there are two possible compatibility paths.

### Option A) Internal replacement
- modify `build_dense_mesh_graph(...)` so it simply calls `build_geodesic_mesh_graph(...)`
- preserve the same public signature
- downstream code remains unchanged

### Option B) Explicit selector during transition
- temporarily add a mesh construction selector internal to `graph.py` or `train.py`
- use geodesic builder only when explicitly chosen

Recommendation:
- implement Option A only after the new function is validated against current examples and plotting behavior
- for the first implementation, keep the new builder separate

## 5. Specific Compatibility Constraints
These constraints should guide implementation choices.

### 5.1 Coordinate conventions
- all returned coordinates must remain `lat, lon`
- do not return projected coordinates from the geodesic builder
- all clipping/projection work should stay internal

### 5.2 Edge-index conventions
- `edge_index` must remain undirected
- each edge must appear exactly once
- endpoints must be sorted within each row or otherwise follow the current normalized convention
- dtype should remain integer, ideally `np.int64`

### 5.3 Error behavior
The new function should keep current error style where possible:
- empty `coords_list` -> clear `ValueError`
- both `spacing_km` and `spacing_deg` provided -> clear `ValueError`
- invalid `bbox` value -> clear `ValueError`
- invalid polygon file -> clear `ValueError`
- clipping that produces zero nodes -> clear `ValueError`

### 5.4 Downstream graph assumptions
The new graph must remain compatible with:
- `edge_features(...)`
- `build_edge_neighbor_pairs(...)`
- `SpeciesGraph.plot(...)`
- `build_species_graphs(...)`
- `build_graph_cv_folds(...)`

That means:
- no multigraph edges
- no duplicate nodes
- no empty graphs
- no disconnected slivers if avoidable

## 6. Validation Plan

## 6.1 Unit-level validation
For the new builder alone:
1. returns non-empty `mesh_coords` and `edge_index` on a simple coordinate set
2. `mesh_coords.shape[1] == 2`
3. `edge_index.shape[1] == 2`
4. all `edge_index` values are in `[0, len(mesh_coords))`
5. no duplicate undirected edges
6. no self-edges
7. output graph is connected after largest-component filtering

## 6.2 Geometric validation
1. median edge length should be close to requested `spacing_km`
2. edge-length distribution should be substantially tighter than the current local-lattice-plus-Delaunay builder
3. clipping with `convex_hull` should not create long perimeter chords, because no Delaunay reconstruction occurs after clipping
4. plotting should show isotropic triangular mesh geometry without row-based horizontal banding from grid construction itself

## 6.3 Compatibility validation
1. swap geodesic mesh output into `build_species_graphs(...)` without changing downstream code
2. confirm sample-to-node assignment still works
3. confirm raster sampling at node coordinates still works
4. confirm `SpeciesGraph.plot(...)` still renders edges and samples normally
5. confirm `build_edge_neighbor_pairs(...)` still produces valid neighboring-edge pairs
6. confirm CV fold construction in `cv.py` still runs on the geodesic graph

## 6.4 Example-level validation
Test on at least:
1. one synthetic small example
2. one real notebook workflow already present in the repository
3. one case using `bbox="convex_hull"`
4. one case using a polygon file

## 7. Documentation Plan
Files to update once implementation begins:
- `src/multispecies_resistance/graph.py` docstrings
- `README.md`
- `docs/graph.md`
- `docs/train.md` if the new builder is wired into `build_species_graphs(...)`
- notebooks/examples if the geodesic builder becomes the default or recommended path

Documentation points to include:
- the geodesic builder preserves native icosphere adjacency instead of using Delaunay
- `spacing_km` selects the nearest available geodesic resolution
- `grid_type` is accepted for compatibility but only triangular meshes are supported
- clipping does not add edges or retriangulate

## 8. Open Design Choices To Resolve During Implementation
These do not block planning, but they should be decided explicitly during coding.

1. Should `build_geodesic_mesh_graph(...)` keep the full global icosphere and clip it, or should it first restrict candidate vertices by a coarse geographic window before clipping?
- Recommendation: start with full icosphere at moderate subdivision levels; optimize later only if performance becomes a problem.

2. Should subdivision selection bias toward slightly coarser or slightly finer edge lengths when no exact spacing match exists?
- Recommendation: bias slightly coarse to control graph size and reduce overfitting risk.

3. Should the largest connected component be mandatory or optional?
- Recommendation: mandatory by default for compatibility and robustness.

4. Should `build_dense_mesh_graph(...)` eventually delegate internally to `build_geodesic_mesh_graph(...)`?
- Recommendation: yes, but only after validating that downstream plotting and training behavior remain stable.

## 9. Implementation Order
1. Add `trimesh` dependency.
2. Add helper functions for face-to-edge conversion, vertex conversion, spacing-to-subdivision mapping, clipping, and connected-component cleanup.
3. Implement `build_geodesic_mesh_graph(...)` with the same external contract as `build_dense_mesh_graph(...)`.
4. Validate standalone outputs for shape, connectivity, and edge-length regularity.
5. Compare graph geometry visually against current dense mesh on the same sample set.
6. Only after validation, decide whether to wire it in as the default dense-mesh implementation.
