# Plan: Simplify `build_species_graphs` to Two Construction Modes

## 1. Requested Change (Scope)
Refactor `build_species_graphs` so graph construction has only two modes:
1. `input_graph` is provided: use `.gml` graph nodes/edges.
 - Note that coordinates inside gml nodes can be recorded either as x/y, lat/lon, or pos values. Don't touch the part of this code that extracts these values from the gml nodes.
2. `input_graph is None` (default): build and use the shared mesh path currently used by `graph_type="dense_mesh"`.

Required API direction from request:
- remove `graph_type` parameter,
- retain `input_graph`,
- retain `mesh_grid_type` as mesh-form option.

No implementation is performed in this plan; this file specifies exact changes only.

## 2. Target Behavior

## 2.1 Mode selection
- `input_graph is not None`:
  - load nodes/edges from `.gml`,
  - map samples to nearest graph node,
  - aggregate genotypes at graph nodes,
  - build pairwise targets from occupied nodes.
- `input_graph is None`:
  - run shared-mesh construction path (currently dense mesh),
  - map samples to nearest mesh node,
  - aggregate genotypes at mesh nodes,
  - build pairwise targets from occupied nodes.

## 2.2 Removed behavior
- remove per-species graph build modes tied to:
  - `graph_type="delaunay"`
  - `graph_type="knn"`
- remove graph-mode branching via `graph_type`.

## 3. Detailed Code Plan

## 3.1 `src/multispecies_resistance/train.py`

### A) Update function signature
Current:
- `build_species_graphs(..., graph_type: str = "delaunay", k: int = 6, ..., input_graph: str | None = None, ...)`

Planned:
- `build_species_graphs(..., input_graph: str | None = None, ..., mesh_grid_type: str = "triangular", ...)`
- remove `graph_type` argument.
- remove `k` argument if it is only used by removed per-species knn path.
 - Yes remove the `k` argument

### B) Remove old branching
- delete `graph_type = graph_type.lower()`.
- delete entire fallback branch that currently builds per-species `delaunay/knn` graphs on raw sample coordinates.
- keep only:
  - `.gml` branch (`input_graph is not None`),
  - mesh branch (`input_graph is None`).

### C) Make mesh branch default path
- convert current `elif graph_type == "dense_mesh":` block into `else:` for `input_graph is None`.
- keep current sample-to-node nearest-neighbor mapping behavior.
- keep current occupied-node filtering and pair target generation logic.

### D) Raster + mesh env behavior
- preserve existing behavior:
  - `raster_paths`/`raster_root` sampling at node coordinates,
  - `mesh_env` support in shared-node modes,
  - zero-width env fallback when neither is provided.

### E) Import cleanup
- remove imports that become unused after deleting per-species path:
  - `build_delaunay_graph`
  - `build_knn_graph`
  (unless still needed by retained mesh-edge options; see clarifications section).

### F) Error text and docstring updates
- rewrite `build_species_graphs` docstring to state two-mode behavior clearly.
- replace any `graph_type`-specific error text with mode-specific messaging.

## 3.2 `src/multispecies_resistance/__init__.py`
- no signature code here, but ensure exports remain valid after import cleanup.
- if removed imports/functions affect exposed API, adjust import list accordingly.

## 3.3 Documentation updates (`docs/` + `README.md`)

### A) Update API examples
- remove `graph_type=...` from all `build_species_graphs(...)` examples.
- present two usage patterns:
  - default mesh (no `input_graph`),
  - explicit `input_graph="...gml"`.

### B) Update behavior descriptions
- `docs/train.md`: replace three-mode description with two-mode description.
- `docs/overview.md` and `docs/overview.svg`: remove references to `delaunay`/`knn` modes.
- `README.md`: remove statements/examples mentioning `"delaunay" | "knn" | "dense_mesh"` selector.

### C) Parameter documentation
- remove `graph_type` and `k` docs.
- retain and describe `mesh_grid_type`.
- keep other mesh controls that remain supported (subject to clarifications below).

## 3.4 Notebooks and examples
- remove `graph_type=` arguments from:
  - `notebooks/pedic_example.ipynb`
  - `notebooks/meems-Dev.ipynb`
  - `notebooks/example_geotiff_pseudosites.ipynb`
  - `examples/minimal_prototype.py`
- ensure default-mesh behavior is used when `input_graph` is not provided.
- keep `.gml` usage examples (if any) aligned to new default behavior.

## 3.5 Optional cleanup of dead code paths
- search/remove any remaining string checks or comments tied to removed `graph_type` values.
- remove stale test fixtures or references expecting per-species graph modes.

## 4. Validation Plan

## 4.1 Static checks
1. `build_species_graphs` no longer accepts `graph_type`.
2. No internal references to removed `"delaunay"`/`"knn"` mode branching in `train.py`.
3. No examples/docs call `build_species_graphs(..., graph_type=...)`.

## 4.2 Behavioral checks
1. `input_graph=None` runs shared mesh path and returns valid `SpeciesGraph` objects.
2. `input_graph="...gml"` runs external-graph path and returns valid `SpeciesGraph` objects.
3. Raster sampling and feature standardization still work in both modes.

## 4.3 Notebook sanity
1. Notebook JSON remains valid after edits.
2. Code cells reflect new `build_species_graphs` call signature.

## 5. Implementation Order
1. Refactor `build_species_graphs` signature and branching logic in `train.py`.
2. Remove obsolete imports / dead per-species graph code.
3. Update examples and notebooks.
4. Update docs and README.
5. Run compile and notebook JSON checks.

## 6. Clarifications Needed
1. Should `mesh_graph_type` and `mesh_k` remain supported in default mesh mode, or should mesh edges be fixed to one method (e.g., always Delaunay) now that `graph_type` is removed?
Remove mesh_graph_type and mesh_k. I removed the 'default mesh mode' and now there should only be two ways to build a species graph, by passing in a .gml or using dense mesh.
2. Should `mesh_coords` (user-supplied node coordinates) remain supported, or should default mesh always be auto-generated when `input_graph is None`?
- Remove mesh_coords
3. For API cleanliness, should we remove `k` immediately (it is only for removed per-species knn mode unless `mesh_graph_type="knn"` remains)?
Remove k
4. Do you want a hard error if callers still pass `graph_type`, or should we temporarily accept it and raise a custom migration error with guidance?
Error
