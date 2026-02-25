# Climate Utilities

Source: `/Users/isaac/src/meems/meems-from-scratch/src/multispecies_resistance/climate.py`

## Download + Cache

- `download_climate_layers(...)`
  - Downloads WorldClim/BioClim archives and extracts GeoTIFFs into a local cache.
  - Reuses cached rasters unless `force_download=True`.
  - Supports `source="bioclim"` or `source="worldclim"`.

## Sampling

- `sample_climate_for_sites(...)`
  - Downloads needed layers (if needed), samples at point coordinates, and returns:
    - `site_env`: sampled matrix
    - `env_names`: canonical variable names
    - `raster_paths`: resolved raster files used

Examples:

```python
from multispecies_resistance.climate import sample_climate_for_sites

site_env, env_names, raster_paths = sample_climate_for_sites(
    site_coords,
    source="bioclim",
    variables=["bio1", "bio12"],
    resolution="2.5m",
)
```

For `source="worldclim"`, `variables` can include:
- group names: `bio`, `prec`, `tavg`, `tmax`, `tmin`, `srad`, `wind`, `vapr`
- individual bands: e.g. `bio12`, `tavg_07`
