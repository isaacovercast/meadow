# Plan: Simplify Pipeline to Sample-Level Inputs Only (Breaking Refactor)

## 1. Refactor Goals (Strict)
1. `SpeciesData` should contain only sample-level genotype and coordinate data, plus species identity (`name`).
2. Remove pseudo-site preprocessing from the default pipeline.
3. Move all environmental extraction into `build_species_graphs`.
4. Remove compatibility code and legacy members/functions that are no longer needed.
5. Prefer deletion over deprecation.

This plan intentionally favors simplification over backward compatibility.

## 2. Final Target Architecture

### 2.1 Canonical in-memory species unit
`SpeciesData` will represent only observed samples:
- `name: str`
- `genotypes: N x M`
- `sample_coords: N x 2` (`lat, lon`)

No site-level storage in `SpeciesData`.

### 2.2 Where node definitions happen
All node definitions happen in `build_species_graphs`:
- per-sample nodes (`graph_type="delaunay"|"knn"` default)
- shared mesh nodes (`graph_type="dense_mesh"`)
- provided global nodes (`input_graph`)

### 2.3 Where environmental extraction happens
Environmental covariates are sampled in `build_species_graphs` at the node coordinates actually used by edges and pairwise training targets.

## 3. File-by-File Change Plan

## 3.1 `src/multispecies_resistance/data.py`

### Keep
- `SpeciesData` (redefined)
- `aggregate_site_genotypes`
- `pairwise_site_distance`

### Remove
- `remap_sites`
- `_ensure_latlon`
- `grid_nodes_from_bbox` (move to `graph.py`, see below)
- `build_pseudosites`

### Redefine `SpeciesData`
- Replace fields with only:
  - `name: str`
  - `genotypes: np.ndarray`
  - `sample_coords: np.ndarray`
- Remove `sample_sites`, `site_coords`, `site_env`, `num_sites()`.

Rationale:
- Enforce a single sample-level contract.
- Prevent persistent pseudo-site/site state from leaking across stages.

## 3.2 `src/multispecies_resistance/graph.py`

### Keep
- `haversine_km`
- projection helpers
- graph builders (`build_knn_graph`, `build_delaunay_graph`, `build_dense_mesh_graph`)
- `edge_features`, `standardize_features`

### Move in
- Move `grid_nodes_from_bbox` from `data.py` into `graph.py` because it is mesh infrastructure.

### Update
- Update imports accordingly (`train.py` should get mesh helpers from `graph.py`).

## 3.3 `src/multispecies_resistance/io.py`

### `load_pedic_species` becomes sample-only loader
Current behavior to remove:
- pseudo-site construction
- raster stack creation/sampling
- site-level assignments

New behavior:
1. Load sample `coords` and `genotypes`.
2. Convert PEDIC `lon,lat` to `lat,lon` once.
3. Return `SpeciesData(name=..., genotypes=..., sample_coords=...)` entries only.

### Signature simplification
- Remove from signature:
  - `spacing_km`, `spacing_deg`
  - `raster_paths`, `raster_root`, `raster_pattern`, `raster_recursive`
  - `coords_crs`, `raster_fill_method`
- Keep only:
  - `root`, `species_names`, `mmap_mode`

### Return simplification
- Return only `List[SpeciesData]`.
- Remove `env_names` from return type.

## 3.4 `src/multispecies_resistance/train.py`

This becomes the main orchestration layer for node definition + sample-to-node assignment + env sampling + graph features.

### `build_species_graphs` signature changes
Add raster/env inputs:
- `raster_paths: Iterable[str | Path] | None = None`
- `raster_root: str | Path | None = None`
- `raster_pattern: str = "*.tif"`
- `raster_recursive: bool = True`
- `raster_fill_method: str = "nan"`
- `raster_coord_order: str = "latlon"`
- `raster_coords_crs: str = "EPSG:4326"`

### Remove all dependence on `sp.site_coords`, `sp.sample_sites`, `sp.site_env`

### Explicit sample-to-node assignment algorithm (inside `build_species_graphs`)
For each species `sp` with `N` samples:

1. Define `node_coords` by graph mode:
- `delaunay` / `knn`: `node_coords = sp.sample_coords`.
- `dense_mesh`: `node_coords = mesh_coords` (shared across species).
- `input_graph`: `node_coords = global graph node coordinates`.

2. Build `sample_sites` (length `N`) mapping sample index -> node index:
- `delaunay` / `knn`: identity map
  - `sample_sites = np.arange(N, dtype=np.int64)`.
- `dense_mesh`: nearest mesh node
  - build KDTree on `mesh_coords`.
  - `sample_sites = tree.query(sp.sample_coords, k=1)[1].astype(np.int64)`.
- `input_graph`: nearest provided graph node
  - build KDTree on global node coords.
  - `sample_sites = tree.query(sp.sample_coords, k=1)[1].astype(np.int64)`.

3. Aggregate sample genotypes by `sample_sites`:
- `site_genos, site_counts = aggregate_site_genotypes(sp.genotypes, sample_sites, num_sites=node_coords.shape[0], allow_empty=True)`.
- Use only occupied nodes (`site_counts > 0`) when building pairwise targets.

4. Build edges on full `node_coords`, and compute `edge_features` using env sampled at `node_coords`.

This yields exactly one remap in shared-node modes (sample -> shared node), with no pseudo-site intermediate.

### Environmental extraction in `build_species_graphs`
1. If raster inputs are provided:
- open one raster stack.
- sample at `node_coords` for each graph mode (once for shared-node modes).
2. Else if `mesh_env` provided in shared-node modes, use it.
3. Else use zero-width env features.

## 3.5 `src/multispecies_resistance/viz.py`

Since `SpeciesData` no longer has site-level fields:
- replace usages that assume `sp.site_coords` with:
  - `g.node_coords` for node overlays, or
  - explicit `sample_coords_list` for sample overlays.
- for `plot_shared_resistance(..., show_sites=True)`, accept explicit sample coords to overlay sample points.

Goal: visualization consumes graph outputs + optional sample overlays, not inferred site-level state.

## 3.6 `src/multispecies_resistance/__init__.py`

Update exports to match simplified API:
- remove `build_pseudosites` export.
- remove exports for deleted helpers.
- keep graph/train/raster/climate/model/viz exports that remain.

## 3.7 `src/multispecies_resistance/raster.py`

No algorithmic changes required.

## 3.8 `src/multispecies_resistance/climate.py`

No algorithmic changes required.

Optional follow-up:
- let `build_species_graphs` accept climate-source args and call `download_climate_layers` internally.

## 3.9 `src/multispecies_resistance/model.py`

No changes required.

## 4. Deletion List (Intentional)

Delete these functions/members as part of simplification:
1. `build_pseudosites` (`data.py`)
2. `remap_sites` (`data.py`)
3. `SpeciesData.sample_sites`
4. `SpeciesData.site_coords`
5. `SpeciesData.site_env`
6. `SpeciesData.num_sites()`
7. loader-time raster options in `load_pedic_species`
8. loader return `env_names`

## 5. API Break Summary

Deliberate breaking changes:
1. `SpeciesData` schema becomes sample-only plus `name`.
2. `load_pedic_species` signature and return type change.
3. `build_species_graphs` gains raster inputs and owns sample-to-node assignment.
4. pseudo-site helpers/pathway removed.

No compatibility layer will be provided.

## 6. Test Plan (Aligned to Simplification)

### 6.1 Unit tests
1. `load_pedic_species` returns `SpeciesData(name, genotypes, sample_coords)` only.
2. `build_species_graphs` per-sample mode uses identity sample mapping.
3. `build_species_graphs` dense mesh mode maps samples directly to mesh nodes.
4. `build_species_graphs` input graph mode maps samples directly to provided nodes.

### 6.2 Env extraction tests
1. Rasters sampled at `node_coords` used by each graph mode.
2. Shared-node modes sample env once and reuse.
3. No raster inputs -> zero-width env features.

### 6.3 Visualization tests
1. Plot functions work with graph outputs and no site-level fields in `SpeciesData`.
2. Sample overlays require explicit sample coordinate inputs.

## 7. Implementation Order
1. Redefine `SpeciesData` and delete pseudo-site helpers in `data.py`.
2. Move `grid_nodes_from_bbox` into `graph.py`.
3. Rewrite `load_pedic_species` to sample-only loader.
4. Refactor `build_species_graphs` to own node definition, sample-to-node mapping, and env extraction.
5. Update visualization functions for new data contract.
6. Update exports and docs.
7. Add/repair tests for new API.

## 8. Expected Result
After refactor:
1. Canonical species representation is minimal: `name`, sample genotypes, sample coordinates.
2. No pseudo-site state persists in memory objects.
3. Shared graph modes perform a single nearest-neighbor mapping from samples to shared nodes.
4. Environmental extraction is tied to the exact training node geometry.
5. Code surface is smaller and easier to reason about.
