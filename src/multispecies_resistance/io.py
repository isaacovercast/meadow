from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import numpy as np

from multispecies_resistance.data import SpeciesData


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
    mmap_mode: str | None = None,
) -> List[SpeciesData]:
    """Load PEDIC FEEMS-style species as sample-level records.

    Expected files per species:
    - `<name>_feems_coords.txt` with columns `lon lat` per sample.
    - `<name>_feems_genos.npy` containing an `N x M` genotype matrix.

    Parameters
    ----------
    root : str | Path
        Directory containing species files.
    species_names : Iterable[str] | None, optional
        Species to load. If `None`, all valid species in `root` are used.
    mmap_mode : str | None, optional
        Memory-mapping mode for genotype array loading.

    Returns
    -------
    List[SpeciesData]
        Loaded species records with sample-level coordinates and genotypes.
    """
    root = Path(root)

    if species_names is None:
        species_names = list_pedic_species(root)
    else:
        species_names = list(species_names)

    species_list: List[SpeciesData] = []
    for name in species_names:
        coord_path = root / f"{name}_feems_coords.txt"
        geno_path = root / f"{name}_feems_genos.npy"
        if not coord_path.exists() or not geno_path.exists():
            raise FileNotFoundError(f"Missing files for species '{name}'")

        coords_lonlat = np.loadtxt(coord_path)
        if coords_lonlat.ndim != 2 or coords_lonlat.shape[1] != 2:
            raise ValueError(f"Invalid coords file for species '{name}'")

        # PEDIC coordinates are stored as lon,lat; convert once to lat,lon.
        sample_coords = coords_lonlat[:, [1, 0]].astype(np.float64)
        genotypes = np.load(geno_path, mmap_mode=mmap_mode)
        if genotypes.shape[0] != sample_coords.shape[0]:
            raise ValueError(
                f"Sample count mismatch for species '{name}': "
                f"{genotypes.shape[0]} genotypes vs {sample_coords.shape[0]} coords"
            )

        species_list.append(
            SpeciesData(
                name=name,
                genotypes=genotypes,
                sample_coords=sample_coords,
            )
        )

    return species_list
