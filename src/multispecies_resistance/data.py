from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class SpeciesData:
    """Container for one species using sample-level observations only.

    Parameters
    ----------
    name : str
        Species name.
    genotypes : np.ndarray
        `N x M` sample-by-marker genotype matrix.
    sample_coords : np.ndarray
        `N x 2` sample coordinates in `lat, lon` order.
    """

    name: str
    genotypes: np.ndarray
    sample_coords: np.ndarray


def aggregate_site_genotypes(
    genotypes: np.ndarray,
    sample_sites: np.ndarray,
    num_sites: int | None = None,
    allow_empty: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Aggregate sample-level genotypes into site-level mean genotypes.

    Parameters
    ----------
    genotypes : np.ndarray
        `N x M` genotype matrix across samples and markers.
    sample_sites : np.ndarray
        Length-`N` compact site index per sample.
    num_sites : int | None, optional
        Total number of sites. If `None`, inferred from `sample_sites`.
    allow_empty : bool, optional
        If `False`, raise when any site has zero assigned samples.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        `site_genotypes` (`S x M`) and `site_counts` (length `S`).
    """
    if num_sites is None:
        num_sites = int(sample_sites.max()) + 1

    num_sites = int(num_sites)
    num_snps = genotypes.shape[1]

    site_genotypes = np.zeros((num_sites, num_snps), dtype=np.float64)
    site_counts = np.zeros(num_sites, dtype=np.int64)

    for i in range(genotypes.shape[0]):
        s = int(sample_sites[i])
        site_genotypes[s] += genotypes[i]
        site_counts[s] += 1

    if np.any(site_counts == 0) and not allow_empty:
        raise ValueError("Some sites have zero samples; check sample_sites mapping.")

    site_genotypes = np.divide(
        site_genotypes,
        site_counts[:, None],
        out=np.zeros_like(site_genotypes),
        where=site_counts[:, None] > 0,
    )
    return site_genotypes, site_counts


def pairwise_site_distance(site_genotypes: np.ndarray) -> np.ndarray:
    """Compute pairwise RMS Euclidean distance between site genotype vectors.

    Parameters
    ----------
    site_genotypes : np.ndarray
        `S x M` matrix of site-level genotype means.

    Returns
    -------
    np.ndarray
        `S x S` symmetric distance matrix using `sqrt(mean((g_i - g_j)^2))`.
    """
    g = site_genotypes.astype(np.float64)
    g2 = np.sum(g * g, axis=1)
    dist2 = g2[:, None] + g2[None, :] - 2.0 * (g @ g.T)
    dist2 = np.maximum(dist2, 0.0)
    dist = np.sqrt(dist2 / g.shape[1])
    return dist
