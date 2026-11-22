# Visualization Utilities

`meadow.viz` provides static matplotlib plots and optional interactive folium views for site distributions, species-specific edge resistance, multi-species summaries, and shared resistance surfaces.

## Colormap
The module defines a default resistance colormap (`edge_cmap`) derived from EEMS-style colors and uses it in edge/surface plotting functions.

## `plot_sites(site_coords, ax=None, title=None, alpha=0.75)`
Plots site coordinates as a scatter map.

Parameters:

- `site_coords`: `N x 2` site coordinates in `lat, lon`.
- `ax`: optional matplotlib axis.
- `title`: optional plot title.
- `alpha`: marker transparency.

Returns:

- `ax`: matplotlib axis with plotted sites.

## `plot_samples(graphs, snap_to_graph=False, sample_size=12, **kwargs)`
Plots the shared graph once with samples from every taxon. All graphs must share node coordinates and edges. The original graphs remain unchanged.

- `snap_to_graph=False`: show each sample at its observed coordinates.
- `snap_to_graph=True`: show one marker at each assigned vertex, with marker area proportional to the total number of samples across taxa at that vertex. Assignment matches the nearest-node rule used by `build_species_graphs`.
- `sample_size`: marker area for one sample, in points squared (default `12`). At a vertex containing `n` samples, the marker area is `n * sample_size`.
- `**kwargs`: existing `graph.plot()` options, including `figsize`, `basemap`, `sample_color`, `sample_alpha`, `edge_color`, `edge_width`, `map_padding`, `edge_feature_idx`, and `add_colorbar`. Feature-colored edges use the first graph's features. All samples use the requested marker color.

When support decay is enabled, existing support weights are combined by taking the maximum across taxa; an explicit `support_decay_km` recomputes support from all assigned sample vertices.

```python
from meadow.viz import plot_samples

# Observed sample coordinates.
ax, edges = plot_samples(graphs, figsize=(10, 12), edge_color="gray")

# Graph vertices; marker area represents sample count.
ax, edges = plot_samples(
    graphs, snap_to_graph=True, sample_size=15,
    figsize=(10, 12), edge_color="gray",
)
```

Returns `(ax, gdf_edges)`, matching `graph.plot()`. Titles, padding, equal geographic aspect, and hidden coordinate ticks follow that method.

## `plot_species_resistance(...)`
Plots one graph's edge resistances as line segments, with equal geographic aspect, padded bounds, and no axis titles.

Parameters:

- `site_coords`: `S x 2` node coordinates.
- `edge_index`: `E x 2` edge list.
- `edge_values`: optional edge values. If omitted, provide `model` and `edge_features`.
- `ax`: optional matplotlib axis.
- `cmap`: edge colormap.
- `basemap`: background basemap setting (`True`, provider object, `False`, or `None`).
- `basemap_crs`: CRS used when plotting with basemap.
- `coord_order`: coordinate order.
- `coords_crs`: CRS of input coordinates.
- `explore`: return folium map when `True`.
- `explore_kwargs`: extra options for `GeoDataFrame.explore`.
- `model`: trained model used to compute edge values.
- `edge_features`: edge features used with `model`.
- `species_idx`: species index used with `model`.
- `show_sites`: overlay points on top of edges.
- `sample_coords`: alternate point coordinates for overlays.
- `value_label`: colorbar label for the plotted edge values.
- `alpha`: edge opacity (default `0.8`).
- `linewidth`: edge width in points (default `0.8`).
- `map_padding`: minimum fraction of each coordinate span added per side (default `0.1`); bounds also expand to fit the panel without stretching geography. Sample coordinates contribute to the bounds even when hidden.
- `extent`: optional explicit `(xmin, xmax, ymin, ymax)` limits in plotting coordinates, overriding padding.
- `show_colorbar`: show the static colorbar (default `True`).
- `show_ticks`: show coordinate ticks and labels (default `True` for this single-species helper).

Returns:

- `(ax, gdf)` or `(ax, gdf, folium_map)` when `explore=True`.

## `plot_multi_edge_resistance(...)`
Plots multiple species either as per-species subplots or as one aggregated overlay.

Parameters:

- `species_list`: species records.
- `graphs`: per-species graph objects.
- `model`: trained model.
- `cmap`: colormap.
- `basemap`, `basemap_crs`, `coord_order`, `coords_crs`: spatial plotting controls.
- `explore`: return interactive maps.
- `explore_kwargs`: options for interactive rendering.
- `overlay`: if `True`, aggregate species into one edge map.
- `overlay_stat`: aggregation statistic (`"mean"` or `"std"`).
- `combine_with_shared`: if `True`, plot the full combined species resistance; if `False`, plot only the species-specific component.
- `show_sites`: overlay point coordinates.
- `sample_coords_list`: optional per-species point coordinate overrides.
- `ncols`: subplot columns when `overlay=False`.
- `figsize`: size **per panel** in inches (default `(8, 10)`); total figure size in overlay mode.
- `alpha`: edge opacity (default `0.8`).
- `linewidth`: edge width in points (default `0.8`).
- `map_padding`: minimum fraction of each coordinate span added per side (default `0.1`); bounds also expand to fit the panel without stretching geography. All facets use the same padded extent, including all sample coordinates whether shown or hidden.
- `show_colorbar`: show static colorbars in either facet or overlay mode (default `True`). Interactive legends are controlled separately through `explore_kwargs`.

Facets use constrained layout and narrow colorbars. Facets and overlays hide axis titles and coordinate ticks. For example:

```python
axes, gdfs = plot_multi_edge_resistance(
    species_list, graphs, model,
    show_sites=True,
    show_colorbar=False,
    alpha=0.8,
    linewidth=0.8,
    map_padding=0.1,
    figsize=(8, 10),
)
```

Returns:

- Overlay mode: `(ax, gdf)` (plus map when `explore=True`).
- Facet mode: `(axes, gdfs)` (plus maps when `explore=True`).

## `plot_shared_resistance(...)`
Plots the shared model component as colored edges or a rasterized interpolated surface, without axis titles or coordinate ticks.

Parameters:

- `species_list`, `graphs`, `model`: required plotting inputs.
- `graph_index`: graph/species index used for geometry.
- `cmap`, `basemap`, `basemap_crs`, `coord_order`, `coords_crs`: spatial rendering controls.
- `explore`, `explore_kwargs`: interactive map controls.
- `rasterize`: if `True`, interpolate to a raster surface.
- `grid_size`: raster resolution.
- `interpolation`: `"midpoint"`, `"rbf"`, or `"kriging"`.
- `interp_method`: SciPy method for midpoint interpolation.
- `fill_method`: fill behavior for midpoint interpolation.
- `rbf_function`, `rbf_kwargs`: RBF interpolation settings.
- `kriging_kwargs`: kriging interpolation settings.
- `show_sites`: overlay site points.
- `alpha`: edge opacity (default `0.8`).
- `linewidth`, `map_padding`: edge width and map padding when `rasterize=False`, matching the species and multi-species plots.

Returns:

- Edge mode: same output pattern as `plot_species_resistance`.
- Raster mode: `(ax, surface)` or `(ax, surface, folium_map)` when `explore=True`.

## `plot_resistance_matrix(R, ax=None, title=None)`
Plots a heatmap of an effective resistance matrix.

Parameters:

- `R`: `N x N` matrix.
- `ax`: optional axis.
- `title`: optional title.

Returns:

- `ax`: matplotlib axis with matrix heatmap.
