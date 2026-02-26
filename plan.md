# Plan: Move `SpeciesGraphData` to `SpeciesGraph` and Add Built-In Plotting

## 1. Scope and Intent
This plan covers a focused refactor:
1. Move graph result container class from `train.py` into `graph.py`.
2. Rename `SpeciesGraphData` to `SpeciesGraph`.
3. Add `sample_coords` to retain original sampling coordinates for each species graph.
4. Add `SpeciesGraph.plot()` to render graph edges on a basemap, optionally color edges by selected edge-feature column, and overlay original sample points.

No backward compatibility layer will be added unless explicitly requested later.

## 2. Target API

### 2.1 New class location and name
- Class lives in: `src/multispecies_resistance/graph.py`
- Class name: `SpeciesGraph`

### 2.2 Proposed `SpeciesGraph` fields
- `name: str`
- `edge_index: np.ndarray` (`E x 2`, int)
- `edge_features: np.ndarray` (`E x F`, float)
- `node_coords: np.ndarray` (`N x 2`, lat/lon by project convention)
- `sample_coords: np.ndarray` (`S x 2`, original observed sample coords)
- `pair_i: np.ndarray`
- `pair_j: np.ndarray`
- `pair_dist: np.ndarray`
- `num_nodes: int`
- `val_pair_i: np.ndarray | None = None`
- `val_pair_j: np.ndarray | None = None`
- `val_pair_dist: np.ndarray | None = None`

### 2.3 `SpeciesGraph.plot()` behavior
`plot()` should:
1. Draw edges as line segments between `node_coords`.
2. Optionally color edges by an edge-feature column index.
3. Overlay `sample_coords` as points.
4. Optionally add a basemap using projected coordinates.
5. Return plotting artifacts for downstream use.

## 3. Detailed File-by-File Plan

## 3.1 `src/multispecies_resistance/graph.py`

### Add class definition
- Introduce `@dataclass class SpeciesGraph` with the fields listed above.
- Keep type hints strict and explicit.

### Add member method `plot(...)`
Proposed signature:
- `def plot(self, edge_feature_idx: int | None = None, ax=None, basemap: bool | object = True, basemap_crs: str = "EPSG:3857", coord_order: str = "latlon", coords_crs: str = "EPSG:4326", sample_size: float = 12.0, edge_width: float = 2.0, edge_cmap="viridis", sample_color: str = "black", sample_alpha: float = 0.8, edge_alpha: float = 0.9, add_colorbar: bool = True, title: str | None = None):`

Method behavior:
1. Validate `edge_index`, `node_coords`, `edge_features`, `sample_coords` shapes.
2. If `edge_feature_idx is None`, draw edges in a constant color.
3. Else validate index in `[0, F-1]` and map `edge_features[:, edge_feature_idx]` to edge colors.
4. Build edge segments from `node_coords[edge_index[:, 0/1]]`.
5. Transform coordinates when `basemap` is enabled (using existing projection helper).
6. Draw edges with `LineCollection`.
7. Draw sample points from `sample_coords` over the edges.
8. Add optional colorbar and title.
9. Return `(ax, gdf_edges)` or `(ax, gdf_edges, mappable)` depending on implementation preference (choose one and document it).

### Internal helper reuse
- Reuse existing projection logic in `graph.py` (or minimal local helper inside method).
- Avoid duplicating projection code already available.

## 3.2 `src/multispecies_resistance/train.py`

### Remove local class
- Delete `SpeciesGraphData` dataclass from this file.

### Update imports
- Import `SpeciesGraph` from `multispecies_resistance.graph`.

### Update `build_species_graphs` outputs
- Change return annotation from `List[SpeciesGraphData]` to `List[SpeciesGraph]`.
- Construct `SpeciesGraph(...)` instances instead of `SpeciesGraphData(...)`.
- Populate new field:
  - `sample_coords=sp.sample_coords` for each species graph.

### Preserve current training behavior
- Keep pair construction/splitting/training logic unchanged except for class name/type updates.

## 3.3 `src/multispecies_resistance/__init__.py`

### Export surface update
- Export `SpeciesGraph` from `.graph`.
- Remove export of `SpeciesGraphData` from `.train`.
- Ensure top-level import paths remain consistent for users.

## 3.4 `src/multispecies_resistance/viz.py`

### Integrate with class method (minimal-change strategy)
- Keep existing standalone plotting utilities for now.
- Optionally refactor internals to call `SpeciesGraph.plot()` where natural, but do not force a broad rewrite in this pass.
- Ensure no duplicated behavior diverges (especially coordinate projection and basemap handling).

## 3.5 Notebooks and examples

### Update class naming references
- Replace any mention of `SpeciesGraphData` with `SpeciesGraph` semantics.

### Demonstrate new method usage
- Update plotting cells to call:
  - `graphs[idx].plot(edge_feature_idx=<k>, basemap=True)`
- Keep existing figure outputs semantically equivalent.

## 3.6 Documentation (`docs/` + README)

### Document new class
- Add/refresh module docs describing `SpeciesGraph` fields including `sample_coords`.
- Document `plot()` parameters and return value.

### Update training docs
- `build_species_graphs` now returns `List[SpeciesGraph]`.
- Explain that each graph object is self-contained for mapping and diagnostics.

## 4. Design Decisions and Constraints

### 4.1 Coordinate conventions
- Keep canonical in-memory coordinates as `lat, lon`.
- Convert to plotting order (`lon, lat`) and projected XY only at render time.

### 4.2 Edge coloring
- Coloring by edge feature uses raw column values by default.
- Do not auto-standardize in `plot()`; plotting should reflect stored values.

### 4.3 Basemap dependency behavior
- If basemap libs are unavailable, fail with clear error or gracefully fall back to non-basemap plotting (pick one and document).

### 4.4 Performance
- Use vectorized segment creation and a `LineCollection` for scale.
- Avoid per-edge Python draw calls.

## 5. Validation Plan

## 5.1 Static checks
1. Imports succeed:
   - `from multispecies_resistance.graph import SpeciesGraph`
2. No stale references to `SpeciesGraphData`.

## 5.2 Behavioral checks
1. `build_species_graphs(...)` returns objects with populated `sample_coords`.
2. `graphs[i].plot()` works with:
   - `edge_feature_idx=None`
   - valid `edge_feature_idx`
   - invalid `edge_feature_idx` (expected error path)
3. Sample points visibly overlay edge layer.

## 5.3 Notebook checks
1. All notebooks run with updated class name and plotting calls.
2. No notebook references to removed class name.

## 6. Implementation Sequence
1. Add `SpeciesGraph` + `plot()` in `graph.py`.
2. Update `train.py` to construct and return `SpeciesGraph`.
3. Update package exports in `__init__.py`.
4. Update notebooks/examples to use `graphs[i].plot(...)`.
5. Update docs and README.
6. Run compile/tests/notebook JSON validation.

## 7. Expected End State
1. Graph outputs are represented by a single canonical class `SpeciesGraph` in `graph.py`.
2. Every graph object retains original observed sample coordinates via `sample_coords`.
3. Mapping a species graph becomes one call (`graph.plot(...)`) with optional edge-feature coloring.
4. The pipeline is cleaner: graph construction in `train.py`, graph representation and graph-level visualization in `graph.py`.
