# Visualization Utilities

## `plot_multi_edge_resistance`

When `overlay=False`:
- Plots one panel per species.

When `overlay=True`:
- Requires identical `edge_index` and `node_coords` across all species graphs.
- Computes species edge resistances separately from `model.edge_resistance(species_idx, edge_features)`.
- Aggregates per-edge values across species into one vector using `overlay_stat`:
  - `"mean"`: edge-wise mean across species
  - `"std"` or `"sd"`: edge-wise standard deviation across species
- Plots a single shared graph with the aggregated edge values.

Key parameters:
- `overlay`: enable single-map aggregation mode
- `overlay_stat`: aggregation statistic in overlay mode (`"mean"` or `"std"`)
- `show_sites`: optional plotting of sample/site points
- `sample_coords_list`: optional per-species sample coordinates for `show_sites`
