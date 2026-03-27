# Plan: Add Data-Support-Based Edge Down-Weighting That Is Easy To Toggle Off

## 1. Goal
Add an optional mechanism that down-weights edges far from nodes that have observed data, using graph distance to the nearest supported node.

The design goal is not just correctness. It must also be easy to disable completely without affecting the rest of the code path. The implementation should therefore be isolated, parameterized cleanly, and avoid entangling the core model logic with special-case conditionals.

No implementation is performed in this plan. This file specifies the changes only.

## 2. Desired Behavior
For each species graph:
- nodes with observed samples should have full support weight
- nodes far from any observed sample should have low support weight
- edges should inherit an attenuation factor based on the support of their endpoint nodes
- this attenuation should reduce the influence of unsupported regions on effective resistance inference

The attenuation should be optional.

When the feature is off:
- behavior should be identical to the current implementation
- no soft warnings
- no partial application
- no need for callers to change existing code

## 3. Recommended Modeling Choice
Use graph-distance-based support weights and apply them directly to edge conductance during resistance-matrix construction.

### 3.1 Why graph distance
Graph distance is the right quantity because:
- the model operates on the graph, not continuous space
- clipped meshes and irregular boundaries make Euclidean distance misleading
- shortest-path distance along the graph reflects actual connectivity support

### 3.2 Why direct conductance attenuation
There are three possible designs:
1. add support weight as an extra edge feature
2. directly attenuate conductance/resistance using support weight
3. use support weight only in a regularization term

Recommendation:
- use direct conductance attenuation

Reason:
- this guarantees unsupported edges are down-weighted
- it matches the stated goal directly
- it is simpler to reason about than adding support as a learned feature
- it is easy to toggle with a single parameter

## 4. Isolation Strategy
This feature should be isolated in three places only:
1. graph preprocessing: compute support weights
2. graph container: store support weights
3. model resistance construction: optionally apply those weights

Everything else should remain unchanged.

In particular:
- `edge_features` should not be changed
- existing CV code should not need special handling
- plotting should not need special handling
- the default training behavior should stay unchanged when support attenuation is disabled

## 5. Public Toggle Design
Add one explicit control parameter to training and graph-building code paths.

Recommended parameter name:
- `support_decay_km: float | None = None`

Interpretation:
- `None`: feature is disabled, preserve current behavior exactly
- positive float: enable support attenuation using this decay scale

Optional second parameter:
- `support_floor: float = 0.01`

Interpretation:
- minimum multiplicative support retained on very distant edges
- avoids exact zero conductance scaling and associated numerical instability

This keeps toggling simple:
- on: set `support_decay_km`
- off: leave it as `None`

## 6. Data Structures To Add

## 6.1 `SpeciesGraph` additions
File:
- `src/multispecies_resistance/graph.py`

Add optional fields:
- `edge_support_weight: np.ndarray | None = None`

Do not add extra fields unless they are needed for debugging.

Rationale:
- the downstream model only needs edge-level attenuation
- storing intermediate node distances and node weights is unnecessary for the first implementation
- keeping only `edge_support_weight` makes the feature smaller and easier to disable/remove later

If debugging support is needed later, it can be added in a separate pass.

## 7. Graph-Side Implementation Plan

## 7.1 Add graph-distance helper
File:
- `src/multispecies_resistance/graph.py`

Add helper:
- `compute_edge_support_weight(...)`

Suggested signature:

```python
def compute_edge_support_weight(
    node_coords: np.ndarray,
    edge_index: np.ndarray,
    occupied_nodes: np.ndarray,
    support_decay_km: float,
    support_floor: float = 0.01,
) -> np.ndarray:
    ...
```

Inputs:
- `node_coords`: `N x 2` in `lat, lon`
- `edge_index`: `E x 2`
- `occupied_nodes`: integer node ids with observed data for the species
- `support_decay_km`: positive decay scale
- `support_floor`: lower bound on support weight

Output:
- `edge_support_weight`: length `E`, values in `(0, 1]`

### Behavior:
1. compute edge lengths in km using `haversine_km(...)`
2. build adjacency with those edge lengths as graph weights
3. run multi-source Dijkstra from all occupied nodes
4. obtain `dist_to_supported[node]`
5. convert node distances into node support weights
6. convert node support weights into edge support weights

## 7.2 Multi-source shortest path
Implementation approach:
- use `scipy.sparse.csgraph.dijkstra(...)` if convenient
- alternatively implement a small heap-based multi-source Dijkstra in pure Python/NumPy

Recommendation:
- prefer `scipy.sparse.csgraph.dijkstra` because SciPy is already a dependency and the implementation will be shorter and clearer

Planned steps:
1. create sparse weighted adjacency matrix from `edge_index` and edge lengths
2. pass `indices=occupied_nodes`
3. request minimum distance to any source
4. collapse to a single distance vector if the API returns per-source distances

## 7.3 Distance-to-weight transform
Recommended node-weight function:

```python
node_weight = support_floor + (1.0 - support_floor) * exp(-dist_to_supported / support_decay_km)
```

Properties:
- occupied nodes: distance `0`, weight `1`
- far nodes: asymptote to `support_floor`
- no exact zero values

This should be implemented in one small helper or directly inside `compute_edge_support_weight(...)`.

## 7.4 Node-to-edge aggregation rule
Recommended edge-weight rule:

```python
edge_support_weight = minimum(node_weight[u], node_weight[v])
```

Reason:
- strict and easy to interpret
- if either endpoint is in weakly supported territory, the edge is weakly supported

Alternative rules like mean or geometric mean should not be implemented initially.

## 8. Training-Data Construction Changes
File:
- `src/multispecies_resistance/train.py`

## 8.1 Add optional parameter to `build_species_graphs(...)`
Add:
- `support_decay_km: float | None = None`
- `support_floor: float = 0.01`

Behavior:
- if `support_decay_km is None`, do not compute support weights
- if provided, compute per-species `edge_support_weight` after sample-to-node assignment and occupied-node identification

## 8.2 Where to compute support weight
For both graph-construction branches in `build_species_graphs(...)`:
1. assign samples to nodes
2. compute `site_counts`
3. determine occupied nodes via `valid = np.where(site_counts > 0)[0]`
4. if support attenuation is enabled:
   - call `compute_edge_support_weight(...)`
   - store result in `SpeciesGraph(edge_support_weight=...)`
5. if disabled:
   - store `edge_support_weight=None`

This keeps the feature local to graph construction and avoids recomputing support distances during training.

## 8.3 Validation rules
Add explicit validation:
- `support_decay_km` must be `None` or `> 0`
- `support_floor` must satisfy `0 <= support_floor <= 1`

If invalid, raise hard errors.

## 9. Model-Side Implementation Plan
File:
- `src/multispecies_resistance/model.py`

## 9.1 Keep current API largely intact
The simplest isolated change is to modify `resistance_matrix(...)` to accept an optional edge attenuation vector.

Suggested change:

```python
def resistance_matrix(
    self,
    species_idx: int,
    edge_index: torch.Tensor,
    edge_feat: torch.Tensor,
    num_nodes: int,
    edge_support_weight: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ...
```

Behavior:
- if `edge_support_weight is None`, preserve current conductance construction exactly
- if provided, attenuate conductance before building the Laplacian

## 9.2 Exact attenuation point
Current logic:

```python
resistance = softplus(shared + species) + 1e-4
conductance = 1.0 / resistance
```

Planned logic:

```python
resistance = softplus(shared + species) + 1e-4
conductance = 1.0 / resistance
if edge_support_weight is not None:
    conductance = conductance * edge_support_weight
```

This is the cleanest insertion point.

Important point:
- do not modify logits
- do not modify `edge_features`
- do not modify loss definitions
- only attenuate conductance when constructing the Laplacian

This keeps the feature localized and easy to disable.

## 10. Training Loop Changes
File:
- `src/multispecies_resistance/train.py`

## 10.1 Pass optional support weights into the model
Inside `train_model(...)`, when looping over `species_graphs`:
- if `g.edge_support_weight is not None`, convert it to torch and pass it to `model.resistance_matrix(...)`
- otherwise pass `None`

This is the only training-loop change needed.

## 10.2 Do not add new training hyperparameters here
The toggle should live in graph construction, not in `train_model(...)`.

Reason:
- support attenuation is a property of the graph/data support geometry
- it should be baked into the graph object and then consumed uniformly during training
- this makes it much easier to reason about and easier to turn off by rebuilding graphs without support attenuation

## 11. Optional Debug/Inspection Support
This is optional and should not block the first implementation.

Possible additions later:
- helper to plot `edge_support_weight` on a graph
- helper to inspect node distance-to-support statistics

These should not be part of the first pass unless needed for debugging.

## 12. Documentation Plan
Files to update when implementing:
- `src/multispecies_resistance/graph.py` docstrings
- `src/multispecies_resistance/train.py` docstrings
- `src/multispecies_resistance/model.py` docstrings
- `README.md`
- `docs/graph.md`
- `docs/train.md`

Key documentation points:
- support attenuation is optional and disabled by default
- it down-weights conductance on edges far from occupied nodes
- it uses graph distance, not Euclidean distance
- it is controlled by `support_decay_km`
- setting `support_decay_km=None` fully disables it

## 13. Validation Plan

## 13.1 Static validation
1. with `support_decay_km=None`, all existing call paths still work
2. `SpeciesGraph` can still be constructed without `edge_support_weight`
3. `train_model(...)` still works on graphs without support weights

## 13.2 Behavioral validation when enabled
1. edges adjacent to occupied nodes have support weight near `1`
2. edges deep in unsupported regions have support weight near `support_floor`
3. support weights are monotone with graph distance from occupied nodes
4. no NaNs or infs in support weights

## 13.3 End-to-end validation
1. build graphs with and without support attenuation on the same dataset
2. verify identical outputs when attenuation is off
3. verify that unsupported remote regions contribute less when attenuation is on
4. inspect inferred edge patterns to confirm remote unsupported regions are suppressed

## 13.4 Numerical validation
1. Laplacian remains well-defined when support weights are near `support_floor`
2. effective resistance computation still succeeds
3. no disconnected-graph numerical failures are introduced solely by attenuation

## 14. Reasons This Design Is Easy To Toggle Off
This design is intentionally easy to disable because:
- the public toggle is a single parameter: `support_decay_km=None`
- all graph-preprocessing code is behind one conditional in `build_species_graphs(...)`
- all model behavior changes are behind one optional argument in `resistance_matrix(...)`
- when disabled, `edge_support_weight=None` flows through the system and current behavior is preserved exactly
- no edge features, losses, or graph topology are modified when disabled

## 15. Implementation Order
1. add `edge_support_weight` field to `SpeciesGraph`
2. add `compute_edge_support_weight(...)` helper to `graph.py`
3. add optional `support_decay_km` and `support_floor` to `build_species_graphs(...)`
4. compute/store support weights during graph construction when enabled
5. extend `model.resistance_matrix(...)` with optional `edge_support_weight`
6. pass support weights through `train_model(...)`
7. update docs and examples
8. run compile and basic graph-building checks

## 16. Explicit Non-Goals For The First Pass
To keep the feature isolated and simple, do not do any of the following in the first implementation:
- do not change `edge_features`
- do not add support weight as a learned feature column
- do not prune edges or nodes from the graph
- do not introduce species-specific support-decay models beyond the per-graph computation
- do not modify CV logic
- do not add plotting/UI features unless needed for debugging
