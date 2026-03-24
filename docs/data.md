# Data Utilities

`multispecies_resistance.data` now defines a strictly sample-level species container and low-level genotype aggregation helpers.

## `SpeciesData`
`SpeciesData` is the canonical in-memory input record for one species.

Parameters:

- `name`: species name.
- `genotypes`: `N x M` sample-by-marker genotype matrix.
- `sample_coords`: `N x 2` sample coordinates (`lat, lon`).

## `aggregate_site_genotypes(genotypes, sample_sites, num_sites=None, allow_empty=False)`
Aggregates sample genotypes into node/site means after a sample-to-node assignment has been computed elsewhere.

Parameters:

- `genotypes`: `N x M` sample genotype matrix.
- `sample_sites`: length-`N` integer node index per sample.
- `num_sites`: number of output nodes/sites (`S`).
- `allow_empty`: whether to allow nodes with zero assigned samples.

Returns:

- `site_genotypes`: `S x M` mean genotype matrix.
- `site_counts`: length-`S` sample counts.

## `pairwise_site_distance(site_genotypes)`
Computes pairwise RMS Euclidean distance between node/site genotype vectors.

Parameters:

- `site_genotypes`: `S x M` node/site genotype matrix.

Returns:

- `dist`: `S x S` symmetric distance matrix.
