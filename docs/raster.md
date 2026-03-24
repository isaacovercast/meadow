# Raster Utilities

`multispecies_resistance.raster` handles extraction of environmental raster values at point locations. It supports single-raster sampling, multi-raster concatenation, path discovery, and a reusable `RasterStack` for repeated sampling.

## `sample_raster_at_points(raster_path, coords, coord_order="latlon", coords_crs="EPSG:4326", fill_method="nan")`
Samples one raster file at coordinate points.

Parameters:

- `raster_path`: raster file path.
- `coords`: `N x 2` query coordinates.
- `coord_order`: coordinate order (`"latlon"` or `"lonlat"`).
- `coords_crs`: CRS of `coords`.
- `fill_method`: nodata handling (`"nan"`, `"mean"`, `"nearest"`).

Returns:

- `values`: `N x B` sampled values.
- `band_names`: list of raster band names.

## `sample_rasters_for_sites(raster_paths, site_coords, coord_order="latlon", coords_crs="EPSG:4326", fill_method="nan")`
Samples multiple rasters and concatenates all bands into one feature matrix.

Parameters:

- `raster_paths`: iterable of raster file paths.
- `site_coords`: `N x 2` coordinate array.
- `coord_order`: coordinate order (`"latlon"` or `"lonlat"`).
- `coords_crs`: CRS of coordinates.
- `fill_method`: nodata handling (`"nan"`, `"mean"`, `"nearest"`).

Returns:

- `site_env`: `N x K` concatenated sampled matrix.
- `env_names`: list of concatenated feature names.

## `resolve_raster_paths(raster_paths, pattern="*.tif", recursive=True)`
Resolves path inputs from files, directories, or glob-like patterns.

Parameters:

- `raster_paths`: one path or iterable of paths/patterns.
- `pattern`: filename pattern used when directories are supplied.
- `recursive`: whether directory search is recursive.

Returns:

- `paths`: sorted unique raster paths.

## `open_raster_stack(...)`
Resolves raster paths and returns an open `RasterStack` plus resolved paths.

Parameters:

- `raster_paths`: one path or iterable of path inputs.
- `pattern`: filename pattern for directory inputs.
- `recursive`: recursive directory search toggle.
- `coord_order`: default coordinate order for sampling.
- `coords_crs`: default coordinate CRS for sampling.
- `fill_method`: default nodata fill mode.

Returns:

- `stack`: open `RasterStack` instance.
- `paths`: resolved raster paths used by the stack.

## `RasterStack`
Reusable context-manager wrapper for repeated point sampling across multiple rasters.

### `RasterStack(raster_paths, coord_order="latlon", coords_crs="EPSG:4326", fill_method="nan")`
Parameters:

- `raster_paths`: raster files to open.
- `coord_order`: default coordinate order for sampling.
- `coords_crs`: default coordinate CRS for sampling.
- `fill_method`: default nodata fill mode.

### `sample_points(coords, fill_method=None)`
Samples all open rasters at query points.

Parameters:

- `coords`: `N x 2` coordinate array.
- `fill_method`: optional per-call override of nodata handling.

Returns:

- `env`: `N x K` concatenated sampled matrix.
- `env_names`: feature names aligned to columns in `env`.

### `close()`
Closes all raster datasets held by the stack.
