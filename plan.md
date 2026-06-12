# Plan: Make DGGRID the Default Mesh Builder and Remove the Dense-Mesh Path

## 1. Goal
Refactor mesh construction so that:
- `build_dggrid_mesh_graph(...)` becomes the default shared mesh backend
- `build_dense_mesh_graph(...)` is removed entirely
- any helper that exists only to support `build_dense_mesh_graph(...)` is also removed
- the rest of the training pipeline continues to operate on the same `(mesh_coords, edge_index)` interface

This plan is intentionally simplification-oriented. No backward-compatibility layer should be retained unless it is still required by a surviving code path.

No implementation is performed in this plan.

## 2. Current State
The code currently has three graph-construction modes inside `build_species_graphs(...)`:
- provided graph via `input_graph`
- shared dense mesh via `build_dense_mesh_graph(...)`
- shared geodesic mesh via `build_geodesic_mesh_graph(...)`
- shared DGGRID mesh via `build_dggrid_mesh_graph(...)`

The dense mesh path is the only one that:
- creates a local regular lattice from a bounding box
- reconstructs topology with Delaunay
- filters long perimeter edges afterward

That path is now redundant if DGGRID is the preferred default.

## 3. Target End State
After the refactor:
- `build_species_graphs(...)` defaults to `mesh_builder="dggrid"`
- `build_dense_mesh_graph(...)` no longer exists
- `mesh_builder="dense"` is no longer accepted
- any dense-only helpers are deleted
- all docs/examples/notebooks refer to DGGRID or geodesic, not dense
- the package has exactly two internal shared-mesh backends:
  - `dggrid`
  - `geodesic`
- plus the separate `input_graph` path

## 4. Main Design Decisions

## 4.1 Keep backend selection explicit
Do not remove the backend selector entirely.

Recommendation:
- keep `mesh_builder: str`
- allowed values become:
  - `"dggrid"`
  - `"geodesic"`

Reason:
- the geodesic builder is still useful as a pure-Python fallback or comparison mode
- explicit mesh backend selection keeps the architecture clean

## 4.2 Make DGGRID the default
Change the default signature in `build_species_graphs(...)` from:
- `mesh_builder: str = "dense"`

to:
- `mesh_builder: str = "dggrid"`

That makes the preferred path the default without changing the overall function shape.

## 4.3 Remove dense-mode compatibility completely
Do not keep:
- `mesh_builder="dense"`
- deprecated aliasing from `dense` to `dggrid`
- warning-based compatibility behavior

If a caller still uses `mesh_builder="dense"`, it should fail with a direct `ValueError`.

## 5. Code Changes by File

## 5.1 `src/multispecies_resistance/train.py`
This is the primary public API change.

### Changes
1. Change default:
```python
mesh_builder: str = "dggrid"
```

2. Restrict validation:
```python
if mesh_builder not in {"dggrid", "geodesic"}:
    raise ValueError(...)
```

3. Remove the dense branch from backend dispatch.
Current conceptual shape:
```python
if mesh_builder == "dense":
    graph_fn = build_dense_mesh_graph
elif mesh_builder == "geodesic":
    graph_fn = build_geodesic_mesh_graph
else:
    graph_fn = build_dggrid_mesh_graph
```

Target shape:
```python
if mesh_builder == "geodesic":
    graph_fn = build_geodesic_mesh_graph
elif mesh_builder == "dggrid":
    graph_fn = build_dggrid_mesh_graph
else:
    raise ValueError(...)
```

4. Preserve all downstream logic unchanged:
- coastline masking
- raster sampling
- edge feature construction
- sample-to-node assignment
- support weighting
- smoothing-neighbor construction

### Things to verify
- no other code path assumes a `dense` default
- no examples rely on omitting `mesh_builder` and then mentally expecting the old dense mesh

## 5.2 `src/multispecies_resistance/graph.py`
This is the main cleanup site.

### Remove the dense mesh builder
Delete:
- `build_dense_mesh_graph(...)`

### Remove dense-only helpers if truly unused
Candidates to remove:
- `grid_nodes_from_bbox(...)`
- `_filter_long_mesh_edges(...)`
- `build_delaunay_graph(...)`

These should be removed only if they are no longer referenced anywhere else after the refactor.

### Keep shared helpers still used by surviving backends
Expected survivors:
- `_coords_to_latlon(...)`
- `_spacing_km_from_deg(...)`
- `_resolve_bbox_spec(...)`
- `_study_region_geometry(...)`
- `_clip_graph_to_region(...)`
- `_largest_connected_component(...)`
- geodesic helpers
- DGGRID helpers
- coastline helpers
- edge feature helpers

### Specific dead-code audit requirement
Before deleting helper functions, verify references across the repo.
The implementation should explicitly check whether each of these is still used:
- `build_delaunay_graph(...)`
- `grid_nodes_from_bbox(...)`
- `_filter_long_mesh_edges(...)`

If any of them are used outside the dense builder, they must either:
- remain, or
- be refactored into the surviving caller appropriately

The user explicitly wants unused pieces removed, so this dead-code audit is required.

## 5.3 `src/multispecies_resistance/__init__.py`
Update exports.

### Remove exports
If deleted from `graph.py`, also remove from package exports:
- `build_dense_mesh_graph`
- `build_delaunay_graph` if deleted

### Keep exports
- `build_dggrid_mesh_graph`
- `build_geodesic_mesh_graph`

This should reflect the simplified supported API.

## 5.4 `README.md`
Update the user-facing description of graph construction.

### Required edits
1. Replace references to “shared dense mesh” as the default.
2. State clearly that the default shared mesh backend is DGGRID.
3. Update any example calls or explanatory text that imply dense is the default.
4. Update any “alternative mesh builder” language so it reflects the new hierarchy:
- default: DGGRID
- alternative: geodesic
- provided graph: `input_graph`

### Suggested wording direction
- “By default, `build_species_graphs(...)` builds a shared triangular DGGRID mesh.”
- “Set `mesh_builder="geodesic"` to use the geodesic fallback mesh instead.”

## 5.5 `docs/train.md`
Update the training docs.

### Required edits
- remove mention of dense as a supported backend
- state that `mesh_builder` accepts only `"dggrid"` and `"geodesic"`
- note that DGGRID is the default
- update any references that describe the default path as a dense mesh

## 5.6 `docs/graph.md`
Update graph docs to reflect the new supported mesh builders.

### Required edits
- remove the section for `build_dense_mesh_graph(...)`
- remove documentation for deleted dense-only helpers
- keep or update documentation for any remaining reusable helper that survives the audit
- keep the DGGRID and geodesic sections
- make the DGGRID section clearly the primary shared mesh builder

## 5.7 `docs/overview.md` and other docs pages
Search and replace any statements that say:
- shared dense mesh is the default
- dense mesh is the standard path

Update diagrams or overview text if they explicitly mention “dense mesh”.

## 5.8 Notebooks and examples
Update notebooks and example scripts to match the new semantics.

### Required checks
- examples that omit `mesh_builder` should still be valid, now implying DGGRID
- examples that explicitly pass `mesh_builder="dense"` must be updated or removed
- any narrative text referring to dense mesh should be rewritten

Likely files to inspect:
- `examples/minimal_prototype.py`
- notebooks under `notebooks/`
- any docs snippets mirrored in notebooks

## 5.9 `environment.yml`
Ensure DGGRID remains listed as a dependency.

Since DGGRID becomes the default backend, this dependency is no longer optional in practice.

## 6. Behavioral Checks After Refactor

## 6.1 Public API behavior
Confirm that:
- `build_species_graphs(...)` with no `mesh_builder` argument uses DGGRID
- `build_species_graphs(..., mesh_builder="geodesic")` still works
- `build_species_graphs(..., mesh_builder="dense")` raises a hard error

## 6.2 Structural checks
Confirm that:
- shared graph output still has `mesh_coords` / `edge_index` in the same shape conventions
- coastline masking still works on DGGRID and geodesic outputs
- raster sampling still operates on the surviving node coordinates
- downstream training code is unchanged

## 6.3 Dead-code cleanup checks
Confirm that no deleted symbol is still referenced anywhere in the repo.
At minimum search for:
- `build_dense_mesh_graph`
- `grid_nodes_from_bbox`
- `_filter_long_mesh_edges`
- `build_delaunay_graph`

Any remaining reference must either be removed or justified by a surviving use.

## 7. Implementation Sequence
1. Update `build_species_graphs(...)` to make DGGRID the default and remove dense as an accepted backend.
2. Remove `build_dense_mesh_graph(...)` from `graph.py`.
3. Remove any helper that was only supporting the dense mesh path.
4. Update package exports in `__init__.py`.
5. Update docs and README.
6. Update examples and notebooks.
7. Run code search to confirm no stale references remain.
8. Run compile and notebook JSON validation.

## 8. Validation Plan
At minimum:
- `python -m py_compile src/multispecies_resistance/*.py examples/minimal_prototype.py`
- notebook JSON parse check for all notebooks
- repository-wide search for removed dense-mesh symbols

If the runtime environment supports it, also perform:
- one smoke test with default DGGRID backend
- one smoke test with `mesh_builder="geodesic"`

## 9. Non-Goals
This refactor should not:
- redesign the DGGRID builder itself
- redesign geodesic mesh generation
- change the graph feature schema
- change training logic
- preserve dense-mesh compatibility

## 10. Expected Result
After this refactor, the package has a simpler and more opinionated shared-mesh story:
- default shared mesh: DGGRID
- alternate shared mesh: geodesic
- explicit custom graph: `input_graph`

The local dense-lattice + Delaunay path is removed entirely, along with any helper code that existed only to support it.
