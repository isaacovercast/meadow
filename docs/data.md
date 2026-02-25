# Data Utilities

`multispecies_resistance.data` contains core preprocessing helpers that transform sample-level genotype and coordinate inputs into site-level representations used throughout graph building and model training.

## `SpeciesData`
`SpeciesData` is the canonical in-memory container for one species.

Parameters:  

- `name`: species name.
- `genotypes`: `N x M` sample-by-marker genotype matrix.
- `sample_sites`: length-`N` site index per sample.
- `site_coords`: `S x 2` site coordinates (default convention: `lat, lon`).
- `site_env`: `S x K` site environmental covariates.

Methods:

- `num_sites()`:
  - Returns the number of rows in `site_coords`.

## `remap_sites(sample_sites)`
Maps arbitrary site labels to compact contiguous site IDs.

Parameters:

- `sample_sites`: length-`N` array of input site IDs.

Returns:

- `remapped`: length-`N` compact IDs in `0..S-1`.
- `mapping`: dictionary from original site ID to compact site ID.

## `aggregate_site_genotypes(genotypes, sample_sites, num_sites=None, allow_empty=False)`
Averages sample genotypes within each site and counts samples per site.

Parameters:

- `genotypes`: `N x M` sample genotype matrix.
- `sample_sites`: length-`N` compact site ID per sample.
- `num_sites`: explicit number of sites; inferred from `sample_sites` when omitted.
- `allow_empty`: if `False`, raises when any site has zero samples.

Returns:

- `site_genotypes`: `S x M` per-site mean genotypes.
- `site_counts`: length-`S` sample count per site.

## `pairwise_site_distance(site_genotypes)`
Computes pairwise RMS Euclidean genotype distance between all site pairs.

Parameters:

- `site_genotypes`: `S x M` site-level genotype matrix.

Returns:

- `dist`: `S x S` symmetric distance matrix.

## `grid_nodes_from_bbox(sample_coords, spacing_km=None, spacing_deg=None, grid_type="triangular")`
Builds a regular grid of candidate pseudo-site nodes over the coordinate extent.

Parameters:

- `sample_coords`: `N x 2` sample coordinates in `lat, lon`.
- `spacing_km`: spacing in kilometers.
- `spacing_deg`: spacing in degrees.
- `grid_type`: `"triangular"` for staggered rows or `"rect"` for a rectangular grid.

Returns:

- `nodes`: `G x 2` node coordinates in `lat, lon`.

## `build_pseudosites(sample_coords, genotypes, spacing_km=50.0, spacing_deg=None, sample_env=None, coord_order="latlon", grid_type="triangular")`
Assigns each sample to the nearest grid node and aggregates genotype/environment summaries by node.

Parameters:

- `sample_coords`: `N x 2` sample coordinates.
- `genotypes`: `N x M` sample genotype matrix.
- `spacing_km`: pseudo-site spacing in kilometers.
- `spacing_deg`: pseudo-site spacing in degrees.
- `sample_env`: optional `N x K` sample-level environmental matrix.
- `coord_order`: coordinate order of `sample_coords` (`"latlon"` or `"lonlat"`).
- `grid_type`: pseudo-site grid type (`"triangular"` or `"rect"`).

Returns:

- `site_coords`: `S x 2` pseudo-site coordinates.
- `sample_sites`: length-`N` pseudo-site assignment per sample.
- `site_genotypes`: `S x M` pseudo-site mean genotypes.
- `site_counts`: length-`S` sample count per pseudo-site.
- `site_env`: `S x K` pseudo-site mean environment (or `None` if `sample_env` not provided).
