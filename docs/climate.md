# Climate Utilities

`multispecies_resistance.climate` adds a download-and-cache layer for WorldClim/BioClim rasters and provides a high-level sampling function that directly returns site-level environmental matrices.

## `download_climate_layers(source="bioclim", variables=None, resolution="2.5m", cache_dir="~/.cache/multispecies_resistance/climate", force_download=False, base_url=None, timeout=120)`
Downloads and extracts requested climate raster groups to a local cache and reuses existing cached files by default.

Parameters:

- `source`: `"bioclim"` or `"worldclim"`.
- `variables`: requested variable/group names (for example `bio12`, `tavg_07`, `bio`, `tavg`).
- `resolution`: WorldClim resolution (`"30s"`, `"2.5m"`, `"5m"`, `"10m"`).
- `cache_dir`: root cache directory for downloads and extracted rasters.
- `force_download`: if `True`, redownloads even when cached rasters exist.
- `base_url`: optional custom base URL (useful for tests or mirrors).
- `timeout`: download timeout in seconds.

Returns:

- `raster_paths`: list of resolved cached GeoTIFF paths in deterministic order.

## `sample_climate_for_sites(site_coords, source="bioclim", variables=None, resolution="2.5m", cache_dir="~/.cache/multispecies_resistance/climate", coord_order="latlon", coords_crs="EPSG:4326", fill_method="nearest", force_download=False, base_url=None, timeout=120)`
Downloads climate layers if needed, samples them at provided coordinates, and optionally subsets to requested variables.

Parameters:

- `site_coords`: `N x 2` site coordinates.
- `source`: `"bioclim"` or `"worldclim"`.
- `variables`: requested variable/group names; `None` keeps all downloaded layers.
- `resolution`: WorldClim resolution token.
- `cache_dir`: climate cache location.
- `coord_order`: coordinate order (`"latlon"` or `"lonlat"`).
- `coords_crs`: CRS of `site_coords`.
- `fill_method`: nodata fill strategy for raster sampling.
- `force_download`: forces re-download of selected groups.
- `base_url`: optional download URL override.
- `timeout`: download timeout in seconds.

Returns:

- `site_env`: `N x K` sampled environmental matrix.
- `env_names`: variable names aligned to `site_env` columns.
- `raster_paths`: raster files used during sampling.

## Typical Usage

```python
from multispecies_resistance.climate import sample_climate_for_sites

site_env, env_names, raster_paths = sample_climate_for_sites(
    site_coords,
    source="bioclim",
    variables=["bio1", "bio12"],
    resolution="2.5m",
)
```
