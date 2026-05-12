from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import h5py
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer

from multispecies_resistance.data import SpeciesData
from multispecies_resistance.vcf_to_hdf5 import VCFtoHDF5

def list_species(root: str | Path) -> List[str]:
    """List species names that have matching coordinate and genotype files.

    Parameters
    ----------
    root : str | Path
        Directory containing paired genotype and coordinates files, one per pair
        per species..

    Returns
    -------
    List[str]
        Species basenames for valid `<name>_coords.txt` and `<name>_genos.npy` pairs.
    """
    root = Path(root)
    coords_files = sorted(root.glob("*_coords.txt"))
    names = []
    for path in coords_files:
        name = path.name.replace("_coords.txt", "")
        if (root / f"{name}_genos.npy").exists() or (root / f"{name}.vcf").exists():
            names.append(name)
    return names


def load_species(
    root: str | Path,
    coords_order: str,
    species_names: Iterable[str] | None = None,
    mmap_mode: str | None = None,
) -> List[SpeciesData]:
    """Load FEEMS-style species as sample-level records.

    Expected files per species:
    - `<name>_coords.txt` with columns `lon lat` per sample.
    - `<name>_genos.npy` containing an `N x M` genotype matrix.

    Parameters
    ----------
    root : str | Path
        Directory containing species files.
    coords_order : str
        One of 'latlon' or 'lonlat'. You must specify this at load time.
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
        species_names = list_species(root)
    else:
        species_names = list(species_names)

    species_list: List[SpeciesData] = []
    for name in species_names:
        # Find the coords file for this sample, load and validate coords
        coord_path = root / f"{name}_coords.txt"
        if not coord_path.exists():
            raise FileNotFoundError(f"Missing coords file: {coord_path}")

        # Store sample_names from coordinates in dataframe format so we
        # can ensure coords and genos map to the same samples
        sample_names = None
        try:
            # Attempt to load coords data stored as raw numpy arrays
            sample_coords = np.loadtxt(coord_path)
            if sample_coords.ndim != 2 or sample_coords.shape[1] != 2:
                raise ValueError(f"Invalid coords file for species '{name}'")
        except ValueError:
            # Try loading from a 3 column csv (id,lat,lon)
            coords_df = pd.read_csv(coord_path, header=None, index_col=0)
            sample_coords = coords_df.values
            # Retain sample names in coords file order
            sample_names = list(coords_df.index)

        # Force the user to choose orientation of georeference data
        if coords_order not in ["latlon", "lonlat"]:
            raise ValueError(f"`coords_order` must be one of 'latlon' or 'lonlat'")
        if coords_order == "lonlat":
            # If coordinates are stored as lon,lat; convert once to lat,lon.
            sample_coords = sample_coords[:, [1, 0]].astype(np.float64)

        geno_path = root / f"{name}_genos.npy"
        if not geno_path.exists():
            # Try loading and converting a vcf file
            vcf_path = root / f"{name}.vcf"
            if not vcf_path.exists():
                raise FileNotFoundError(f"No genos or vcf found: {geno_path} / {vcf_path}")
            print(f"Converting vcf to hdf5 for - {name}")
            v2h = VCFtoHDF5(str(vcf_path), vcf_path.stem)
            v2h.run(force=True)

            # Pull the snps out of the hdf5 and write to a np array
            with h5py.File(f"analysis-vcf2hdf5/{vcf_path.stem}.snps.hdf5") as io5:
                genos = io5["genos"][:].sum(axis=2).T
                genos[genos == 18] = 9

                # Impute missing values
                imp = SimpleImputer(missing_values=9, strategy="most_frequent")
                genos = imp.fit_transform(genos)

                # Reorder to match name order in the coords file
                idx_map = {name: i for i, name in enumerate(sample_names)}
                # Get genos row indices in target order
                hnames = io5["snps"].attrs["names"]
                print(sample_names)
                print(hnames)
                reorder_idx = [idx_map[name] for name in hnames]
                # Reorder genos array
                genos = genos[reorder_idx]

                np.save(f"{geno_path}", genos)

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
