# Data Utilities

Source: `/Users/isaac/src/meems/meems-from-scratch/src/multispecies_resistance/data.py`

## `SpeciesData`
Container for one species:
- `name`
- `genotypes`: `N x M` array in `{0,1,2}`
- `sample_sites`: length `N` site index for each sample
- `site_coords`: `S x 2` (`lat, lon` by convention)
- `site_env`: `S x K` site covariates

## `aggregate_site_genotypes(genotypes, sample_sites, num_sites=None, allow_empty=False)`
Averages sample genotypes to site-level means.
- Returns `(site_genotypes, site_counts)`.
- If `allow_empty=True`, empty sites are allowed and get zero vectors.

## `pairwise_site_distance(site_genotypes)`
Computes RMS Euclidean distance for all site pairs.
- Returns `S x S` matrix.

## `grid_nodes_from_bbox(sample_coords, spacing_km=None, spacing_deg=None, grid_type="triangular")`
Builds grid nodes over a coordinate extent.
- `grid_type="triangular"`: staggered triangular mesh.
- `grid_type="rect"`: regular rectangular mesh.

## `build_pseudosites(sample_coords, genotypes, spacing_km=50.0, spacing_deg=None, sample_env=None, coord_order="latlon", grid_type="triangular")`
Assigns each sample to nearest grid node and aggregates by node.
- Returns:
  - `site_coords`
  - `sample_sites`
  - `site_genotypes`
  - `site_counts`
  - `site_env` (or `None`)
