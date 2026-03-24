# Visualization Utilities

`multispecies_resistance.viz` provides static matplotlib plots and optional interactive folium views for site distributions, species-specific edge resistance, multi-species summaries, and shared resistance surfaces.

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

## `plot_species_resistance(...)`
Plots one graph's edge resistances as line segments.

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
- `show_sites`: overlay point coordinates.
- `sample_coords_list`: optional per-species point coordinate overrides.
- `ncols`: subplot columns when `overlay=False`.
- `figsize`: base plot size.

Returns:

- Overlay mode: `(ax, gdf)` (plus map when `explore=True`).
- Facet mode: `(axes, gdfs)` (plus maps when `explore=True`).

## `plot_shared_resistance(...)`
Plots the shared model component as colored edges or a rasterized interpolated surface.

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
