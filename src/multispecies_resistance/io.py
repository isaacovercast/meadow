from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from multispecies_resistance.data import SpeciesData, build_pseudosites
from multispecies_resistance.raster import RasterStack, resolve_raster_paths


def list_pedic_species(root: str | Path) -> List[str]:
    """List species names that have matching PEDIC coordinate and genotype files.

    Parameters
    ----------
    root : str | Path
        Directory containing PEDIC FEEMS-style files.

    Returns
    -------
    List[str]
        Species basenames for valid `<name>_feems_coords.txt` and `<name>_feems_genos.npy` pairs.
    """
    root = Path(root)
    coords_files = sorted(root.glob("*_feems_coords.txt"))
    names = []
    for path in coords_files:
        name = path.name.replace("_feems_coords.txt", "")
        if (root / f"{name}_feems_genos.npy").exists():
            names.append(name)
    return names


def load_pedic_species(
    root: str | Path,
    species_names: Iterable[str] | None = None,
    spacing_km: float | None = 80.0,
    spacing_deg: float | None = None,
    raster_paths: Iterable[str | Path] | None = None,
    raster_root: str | Path | None = None,
    raster_pattern: str = "*.tif",
    raster_recursive: bool = True,
    coords_crs: str = "EPSG:4326",
    raster_fill_method: str = "nan",
    mmap_mode: str | None = None,
) -> Tuple[List[SpeciesData], List[str]]:
    """Load PEDIC FEEMS-style species data and optionally sample raster covariates.

    Parameters
    ----------
    root : str | Path
        Directory containing species files.
    species_names : Iterable[str] | None, optional
        Species to load. If `None`, all valid species in `root` are used.
    spacing_km : float | None, optional
        Pseudo-site grid spacing in kilometers.
    spacing_deg : float | None, optional
        Pseudo-site grid spacing in degrees.
    raster_paths : Iterable[str | Path] | None, optional
        Explicit raster paths used for site covariate sampling.
    raster_root : str | Path | None, optional
        Raster directory or glob root used when `raster_paths` is not provided.
    raster_pattern : str, optional
        File pattern for raster discovery.
    raster_recursive : bool, optional
        Whether raster discovery recurses into subdirectories.
    coords_crs : str, optional
        CRS of coordinates used during raster sampling.
    raster_fill_method : str, optional
        Missing-value fill mode for raster sampling.
    mmap_mode : str | None, optional
        Memory-mapping mode for genotype array loading.

    Returns
    -------
    Tuple[List[SpeciesData], List[str]]
        Loaded species records and sampled environmental feature names.
    """
    root = Path(root)

    if species_names is None:
        species_names = list_pedic_species(root)
    else:
        species_names = list(species_names)

    env_names: List[str] = []
    raster_stack = None
    if raster_paths is None and raster_root is not None:
        raster_paths = resolve_raster_paths(
            raster_root, pattern=raster_pattern, recursive=raster_recursive
        )
    if raster_paths is not None:
        raster_stack = RasterStack(
            raster_paths,
            coord_order="latlon",
            coords_crs=coords_crs,
            fill_method=raster_fill_method,
        )

    species_list: List[SpeciesData] = []
    try:
        for name in species_names:
            coord_path = root / f"{name}_feems_coords.txt"
            geno_path = root / f"{name}_feems_genos.npy"
            if not coord_path.exists() or not geno_path.exists():
                raise FileNotFoundError(f"Missing files for species '{name}'")

            coords_lonlat = np.loadtxt(coord_path)
            if coords_lonlat.ndim != 2 or coords_lonlat.shape[1] != 2:
                raise ValueError(f"Invalid coords file for species '{name}'")

            coords_latlon = coords_lonlat[:, [1, 0]]
            genotypes = np.load(geno_path, mmap_mode=mmap_mode)

            site_coords, sample_sites, _, _, _ = build_pseudosites(
                coords_latlon,
                genotypes,
                spacing_km=spacing_km,
                spacing_deg=spacing_deg,
                coord_order="latlon",
            )

            if raster_stack is not None:
                site_env, env_names = raster_stack.sample_points(site_coords)
            else:
                site_env = np.zeros((site_coords.shape[0], 0), dtype=np.float64)

            species_list.append(
                SpeciesData(
                    name=name,
                    genotypes=genotypes,
                    sample_sites=sample_sites,
                    site_coords=site_coords,
                    site_env=site_env,
                )
            )
    finally:
        if raster_stack is not None:
            raster_stack.close()

    return species_list, env_names
