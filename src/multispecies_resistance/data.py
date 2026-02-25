from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class SpeciesData:
    """Container for one species' genotypes, site assignments, and covariates."""

    name: str
    genotypes: np.ndarray
    sample_sites: np.ndarray
    site_coords: np.ndarray
    site_env: np.ndarray

    def num_sites(self) -> int:
        """Return the number of site rows in `site_coords`.

        Returns
        -------
        int
            Number of sites represented for this species.
        """
        return int(self.site_coords.shape[0])


def remap_sites(sample_sites: np.ndarray) -> Tuple[np.ndarray, Dict[int, int]]:
    """Remap arbitrary site IDs to a compact contiguous index.

    Parameters
    ----------
    sample_sites : np.ndarray
        Length-`N` array of site IDs.

    Returns
    -------
    Tuple[np.ndarray, Dict[int, int]]
        Remapped site IDs (length `N`) and a dictionary from original ID to compact ID.
    """
    unique = np.unique(sample_sites)
    mapping = {int(s): i for i, s in enumerate(unique)}
    remapped = np.array([mapping[int(s)] for s in sample_sites], dtype=np.int64)
    return remapped, mapping


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

    if np.any(site_counts == 0):
        if not allow_empty:
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


def _ensure_latlon(coords: np.ndarray, coord_order: str) -> np.ndarray:
    """Return coordinates in `lat, lon` order.

    Parameters
    ----------
    coords : np.ndarray
        `N x 2` coordinate matrix.
    coord_order : str
        Input order, either `"latlon"` or `"lonlat"`.

    Returns
    -------
    np.ndarray
        Coordinates arranged as `lat, lon`.
    """
    if coord_order not in {"latlon", "lonlat"}:
        raise ValueError("coord_order must be 'latlon' or 'lonlat'")
    if coord_order == "lonlat":
        return coords[:, [1, 0]]
    return coords


def grid_nodes_from_bbox(
    sample_coords: np.ndarray,
    spacing_km: float | None = None,
    spacing_deg: float | None = None,
    grid_type: str = "triangular",
) -> np.ndarray:
    """Generate regularly spaced grid nodes covering a coordinate bounding box.

    Parameters
    ----------
    sample_coords : np.ndarray
        `N x 2` sample coordinates in `lat, lon`.
    spacing_km : float | None, optional
        Approximate spacing in kilometers.
    spacing_deg : float | None, optional
        Spacing in degrees. Provide exactly one of `spacing_km` or `spacing_deg`.
    grid_type : str, optional
        `"triangular"` for staggered rows or `"rect"` for a regular lattice.

    Returns
    -------
    np.ndarray
        `G x 2` grid node coordinates in `lat, lon`.
    """
    if spacing_km is None and spacing_deg is None:
        raise ValueError("Provide spacing_km or spacing_deg.")
    if spacing_km is not None and spacing_deg is not None:
        raise ValueError("Provide only one of spacing_km or spacing_deg.")

    if spacing_deg is None:
        mean_lat = float(np.mean(sample_coords[:, 0]))
        deg_lat = spacing_km / 111.0
        deg_lon = spacing_km / (111.0 * np.cos(np.radians(mean_lat)))
    else:
        deg_lat = spacing_deg
        deg_lon = spacing_deg

    lat_min, lon_min = np.min(sample_coords, axis=0)
    lat_max, lon_max = np.max(sample_coords, axis=0)

    grid_type = grid_type.lower()
    if grid_type == "triangular":
        # Triangular grid spacing: horizontal step = dx, vertical step = dy
        dx = deg_lon
        dy = deg_lat * (np.sqrt(3.0) / 2.0)

        lat_grid = np.arange(lat_min, lat_max + dy, dy)
        rows = []
        for r, lat in enumerate(lat_grid):
            offset = 0.5 * dx if (r % 2 == 1) else 0.0
            lon_start = lon_min + offset
            lon_vals = np.arange(lon_start, lon_max + dx, dx)
            if lon_start > lon_min:
                lon_vals = np.concatenate(([lon_start - dx], lon_vals))
            lon_vals = lon_vals[(lon_vals >= lon_min - 1e-9) & (lon_vals <= lon_max + 1e-9)]
            if lon_vals.size == 0:
                continue
            rows.append(np.column_stack([np.full_like(lon_vals, lat), lon_vals]))

        nodes = np.vstack(rows) if rows else np.empty((0, 2), dtype=np.float64)
    elif grid_type == "rect":
        lat_grid = np.arange(lat_min, lat_max + deg_lat, deg_lat)
        lon_grid = np.arange(lon_min, lon_max + deg_lon, deg_lon)
        grid_lat, grid_lon = np.meshgrid(lat_grid, lon_grid, indexing="ij")
        nodes = np.column_stack([grid_lat.ravel(), grid_lon.ravel()])
    else:
        raise ValueError("grid_type must be 'triangular' or 'rect'")
    return nodes


def build_pseudosites(
    sample_coords: np.ndarray,
    genotypes: np.ndarray,
    spacing_km: float | None = 50.0,
    spacing_deg: float | None = None,
    sample_env: np.ndarray | None = None,
    coord_order: str = "latlon",
    grid_type: str = "triangular",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Assign samples to nearest grid nodes and aggregate to pseudo-site summaries.

    Parameters
    ----------
    sample_coords : np.ndarray
        `N x 2` sample coordinates.
    genotypes : np.ndarray
        `N x M` genotype matrix.
    spacing_km : float | None, optional
        Grid spacing in kilometers.
    spacing_deg : float | None, optional
        Grid spacing in degrees.
    sample_env : np.ndarray | None, optional
        Optional sample-level environmental features (`N x K`) to average by pseudo-site.
    coord_order : str, optional
        Coordinate order for `sample_coords`, either `"latlon"` or `"lonlat"`.
    grid_type : str, optional
        Grid geometry, `"triangular"` or `"rect"`.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]
        `site_coords`, `sample_sites`, `site_genotypes`, `site_counts`, and `site_env`.
    """
    from scipy.spatial import cKDTree

    sample_coords = _ensure_latlon(sample_coords, coord_order)
    nodes = grid_nodes_from_bbox(
        sample_coords,
        spacing_km=spacing_km,
        spacing_deg=spacing_deg,
        grid_type=grid_type,
    )
    tree = cKDTree(nodes)
    _, assigned = tree.query(sample_coords, k=1)

    used_nodes = np.unique(assigned)
    remap = {int(n): i for i, n in enumerate(used_nodes)}
    sample_sites = np.array([remap[int(n)] for n in assigned], dtype=np.int64)

    site_coords = nodes[used_nodes]
    site_genotypes, site_counts = aggregate_site_genotypes(
        genotypes, sample_sites, num_sites=site_coords.shape[0]
    )

    site_env = None
    if sample_env is not None:
        site_env = np.zeros((site_coords.shape[0], sample_env.shape[1]), dtype=np.float64)
        for i in range(sample_env.shape[0]):
            s = sample_sites[i]
            site_env[s] += sample_env[i]
        site_env /= site_counts[:, None]

    return site_coords, sample_sites, site_genotypes, site_counts, site_env
