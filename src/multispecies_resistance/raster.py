from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.warp import transform
from scipy.spatial import cKDTree


def _as_lon_lat(coords: np.ndarray, coord_order: str) -> Tuple[np.ndarray, np.ndarray]:
    """Split coordinates into longitude and latitude arrays.

    Parameters
    ----------
    coords : np.ndarray
        `N x 2` coordinate matrix.
    coord_order : str
        Input order, `"latlon"` or `"lonlat"`.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Longitude array and latitude array.
    """
    if coord_order not in {"latlon", "lonlat"}:
        raise ValueError("coord_order must be 'latlon' or 'lonlat'")
    if coord_order == "latlon":
        lats = coords[:, 0]
        lons = coords[:, 1]
    else:
        lons = coords[:, 0]
        lats = coords[:, 1]
    return lons, lats


def _coords_to_dataset(
    src: rasterio.DatasetReader,
    coords: np.ndarray,
    coord_order: str,
    coords_crs: str | CRS | None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Transform query coordinates into the raster dataset CRS.

    Parameters
    ----------
    src : rasterio.DatasetReader
        Open raster dataset.
    coords : np.ndarray
        `N x 2` point coordinates.
    coord_order : str
        Input coordinate order.
    coords_crs : str | CRS | None
        CRS of input coordinates.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        `x` and `y` arrays in dataset CRS.
    """
    lons, lats = _as_lon_lat(coords, coord_order)
    if coords_crs is not None and src.crs is not None:
        src_crs = src.crs
        in_crs = CRS.from_user_input(coords_crs)
        if src_crs != in_crs:
            xs, ys = transform(in_crs, src_crs, lons.tolist(), lats.tolist())
        else:
            xs, ys = lons, lats
    else:
        xs, ys = lons, lats
    return np.asarray(xs), np.asarray(ys)


def _band_names(src: rasterio.DatasetReader) -> List[str]:
    """Return raster band names using descriptions when available.

    Parameters
    ----------
    src : rasterio.DatasetReader
        Open raster dataset.

    Returns
    -------
    List[str]
        Band names in read order.
    """
    names = []
    for i in range(src.count):
        desc = src.descriptions[i] if src.descriptions else None
        names.append(desc if desc else f"band_{i+1}")
    return names


def _sample_raw(
    src: rasterio.DatasetReader, xs: np.ndarray, ys: np.ndarray
) -> Tuple[np.ndarray, np.ndarray | None]:
    """Sample raster values at point coordinates and handle nodata masks.

    Parameters
    ----------
    src : rasterio.DatasetReader
        Open raster dataset.
    xs : np.ndarray
        Query x coordinates in raster CRS.
    ys : np.ndarray
        Query y coordinates in raster CRS.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray | None]
        Sampled values (`N x B`) and optional missing-value mask.
    """
    try:
        samples = np.array(list(src.sample(zip(xs, ys), masked=True)))
        mask = np.ma.getmaskarray(samples)
        data = np.ma.filled(samples, np.nan).astype(np.float64)
        return data, mask
    except TypeError:
        samples = np.array(list(src.sample(zip(xs, ys)))).astype(np.float64)
        mask = None
        if src.nodata is not None:
            mask = samples == src.nodata
            samples = np.where(mask, np.nan, samples)
        return samples, mask


def _band_means(src: rasterio.DatasetReader) -> np.ndarray:
    """Compute per-band means while respecting raster masks.

    Parameters
    ----------
    src : rasterio.DatasetReader
        Open raster dataset.

    Returns
    -------
    np.ndarray
        Length-`B` vector of band means.
    """
    means = []
    for b in range(1, src.count + 1):
        data = src.read(b, masked=True)
        if np.ma.is_masked(data):
            mean_val = float(data.mean())
        else:
            mean_val = float(np.mean(data))
        means.append(mean_val)
    return np.array(means, dtype=np.float64)


def _build_kdtree(src: rasterio.DatasetReader, band: int):
    """Build a KD-tree over valid pixels for one raster band.

    Parameters
    ----------
    src : rasterio.DatasetReader
        Open raster dataset.
    band : int
        One-based band index.

    Returns
    -------
    tuple
        `(tree, values)` where `tree` is a KD-tree over valid cell centers and
        `values` are corresponding pixel values, or `(None, None)` if no valid cells.
    """
    data = src.read(band, masked=True)
    if not np.ma.is_masked(data):
        mask = np.zeros(data.shape, dtype=bool)
        values = data.ravel()
    else:
        mask = np.ma.getmaskarray(data)
        values = data[~mask]

    if values.size == 0:
        return None, None

    rows, cols = np.where(~mask)
    xs, ys = rasterio.transform.xy(src.transform, rows, cols, offset="center")
    coords = np.column_stack([xs, ys])
    tree = cKDTree(coords)
    return tree, values.astype(np.float64)


def _fill_missing(
    src: rasterio.DatasetReader,
    xs: np.ndarray,
    ys: np.ndarray,
    data: np.ndarray,
    fill_method: str,
    cache: Dict,
) -> np.ndarray:
    """Fill NaN sampled values using selected missing-data strategy.

    Parameters
    ----------
    src : rasterio.DatasetReader
        Open raster dataset.
    xs : np.ndarray
        Query x coordinates in raster CRS.
    ys : np.ndarray
        Query y coordinates in raster CRS.
    data : np.ndarray
        Sampled raster values (`N x B`) potentially containing NaNs.
    fill_method : str
        One of `"nan"`, `"mean"`, or `"nearest"`.
    cache : Dict
        Mutable cache for per-band means and KD-trees.

    Returns
    -------
    np.ndarray
        Filled sample matrix with same shape as `data`.
    """
    if fill_method == "nan":
        return data
    if fill_method not in {"mean", "nearest"}:
        raise ValueError("fill_method must be 'nan', 'mean', or 'nearest'")

    filled = data.copy()
    n_samples, n_bands = filled.shape

    if fill_method == "mean":
        if "means" not in cache:
            cache["means"] = _band_means(src)
        means = cache["means"]
        for b in range(n_bands):
            mask = np.isnan(filled[:, b])
            filled[mask, b] = means[b]
        return filled

    # nearest neighbor fill
    if "kdtree" not in cache:
        cache["kdtree"] = {}
    for b in range(n_bands):
        mask = np.isnan(filled[:, b])
        if not np.any(mask):
            continue
        key = f"band_{b+1}"
        if key not in cache["kdtree"]:
            tree, values = _build_kdtree(src, b + 1)
            cache["kdtree"][key] = (tree, values)
        else:
            tree, values = cache["kdtree"][key]
        if tree is None or values is None:
            continue
        q = np.column_stack([xs[mask], ys[mask]])
        _, idx = tree.query(q, k=1)
        filled[mask, b] = values[idx]
    return filled


def sample_raster_at_points(
    raster_path: str | Path,
    coords: np.ndarray,
    coord_order: str = "latlon",
    coords_crs: str | CRS | None = "EPSG:4326",
    fill_method: str = "nan",
) -> Tuple[np.ndarray, List[str]]:
    """Sample one raster at point coordinates and return values plus band names.

    Parameters
    ----------
    raster_path : str | Path
        Raster file path.
    coords : np.ndarray
        `N x 2` query coordinates.
    coord_order : str, optional
        Coordinate order of `coords`.
    coords_crs : str | CRS | None, optional
        CRS of `coords`.
    fill_method : str, optional
        Missing-value fill mode (`"nan"`, `"mean"`, `"nearest"`).

    Returns
    -------
    Tuple[np.ndarray, List[str]]
        Sample matrix (`N x B`) and band names.
    """
    coords = np.asarray(coords)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must be N x 2")

    with rasterio.open(raster_path) as src:
        xs, ys = _coords_to_dataset(src, coords, coord_order, coords_crs)
        data, _ = _sample_raw(src, xs, ys)
        data = _fill_missing(src, xs, ys, data, fill_method, cache={})
        return data.astype(np.float64), _band_names(src)


def sample_rasters_for_sites(
    raster_paths: Iterable[str | Path],
    site_coords: np.ndarray,
    coord_order: str = "latlon",
    coords_crs: str | CRS | None = "EPSG:4326",
    fill_method: str = "nan",
) -> Tuple[np.ndarray, List[str]]:
    """Sample multiple raster files and concatenate their band values by site.

    Parameters
    ----------
    raster_paths : Iterable[str | Path]
        Raster file paths.
    site_coords : np.ndarray
        `N x 2` site coordinates.
    coord_order : str, optional
        Coordinate order of `site_coords`.
    coords_crs : str | CRS | None, optional
        CRS of `site_coords`.
    fill_method : str, optional
        Missing-value fill mode (`"nan"`, `"mean"`, `"nearest"`).

    Returns
    -------
    Tuple[np.ndarray, List[str]]
        Concatenated sample matrix (`N x K`) and feature names.
    """
    env_blocks = []
    env_names: List[str] = []

    for path in raster_paths:
        values, band_names = sample_raster_at_points(
            path,
            site_coords,
            coord_order=coord_order,
            coords_crs=coords_crs,
            fill_method=fill_method,
        )
        stem = Path(path).stem
        env_blocks.append(values)
        env_names.extend([f"{stem}_{name}" for name in band_names])

    site_env = np.concatenate(env_blocks, axis=1) if env_blocks else np.empty((0, 0))
    return site_env, env_names


def resolve_raster_paths(
    raster_paths: Iterable[str | Path] | str | Path,
    pattern: str = "*.tif",
    recursive: bool = True,
) -> List[Path]:
    """Resolve raster input paths from files, directories, or glob patterns.

    Parameters
    ----------
    raster_paths : Iterable[str | Path] | str | Path
        Path-like inputs to resolve.
    pattern : str, optional
        Glob pattern used when a directory is supplied.
    recursive : bool, optional
        Whether directory searches recurse.

    Returns
    -------
    List[Path]
        Sorted unique raster file paths.
    """
    paths: List[Path] = []

    def add_path(p: Path):
        if p.exists() and p.is_file():
            paths.append(p)
        elif p.exists() and p.is_dir():
            globber = p.rglob(pattern) if recursive else p.glob(pattern)
            paths.extend([q for q in globber if q.is_file()])
        else:
            # treat as glob pattern
            if any(ch in str(p) for ch in ["*", "?", "["]):
                for q in Path().glob(str(p)):
                    if q.is_file():
                        paths.append(q)

    if isinstance(raster_paths, (str, Path)):
        add_path(Path(raster_paths))
    else:
        for item in raster_paths:
            add_path(Path(item))

    unique = sorted({p.resolve() for p in paths})
    return unique


def open_raster_stack(
    raster_paths: Iterable[str | Path] | str | Path,
    pattern: str = "*.tif",
    recursive: bool = True,
    coord_order: str = "latlon",
    coords_crs: str | CRS | None = "EPSG:4326",
    fill_method: str = "nan",
) -> Tuple["RasterStack", List[Path]]:
    """Open a cached raster stack after resolving path inputs.

    Parameters
    ----------
    raster_paths : Iterable[str | Path] | str | Path
        Path-like inputs to resolve.
    pattern : str, optional
        Glob pattern used when a directory is supplied.
    recursive : bool, optional
        Whether directory searches recurse.
    coord_order : str, optional
        Coordinate order expected at sampling time.
    coords_crs : str | CRS | None, optional
        CRS expected for sampling coordinates.
    fill_method : str, optional
        Default missing-value fill mode.

    Returns
    -------
    Tuple["RasterStack", List[Path]]
        Open `RasterStack` instance and resolved raster paths.
    """
    paths = resolve_raster_paths(raster_paths, pattern=pattern, recursive=recursive)
    if not paths:
        raise FileNotFoundError("No raster files found.")
    stack = RasterStack(
        paths, coord_order=coord_order, coords_crs=coords_crs, fill_method=fill_method
    )
    return stack, paths


class RasterStack:
    """Convenience class for sampling multiple rasters with caching."""

    def __init__(
        self,
        raster_paths: Iterable[str | Path],
        coord_order: str = "latlon",
        coords_crs: str | CRS | None = "EPSG:4326",
        fill_method: str = "nan",
    ) -> None:
        """Open raster datasets and initialize caches for repeated sampling.

        Parameters
        ----------
        raster_paths : Iterable[str | Path]
            Raster files to open.
        coord_order : str, optional
            Expected order for input coordinates.
        coords_crs : str | CRS | None, optional
            CRS of input coordinates at sampling time.
        fill_method : str, optional
            Default missing-value fill mode.
        """
        self.raster_paths = [Path(p) for p in raster_paths]
        self.coord_order = coord_order
        self.coords_crs = coords_crs
        self.fill_method = fill_method
        self._datasets = [rasterio.open(p) for p in self.raster_paths]
        self._cache: Dict[int, Dict] = {}

    def close(self) -> None:
        """Close all open raster datasets held by the stack.

        Returns
        -------
        None
            Closes datasets in-place.
        """
        for ds in self._datasets:
            ds.close()

    def __enter__(self):
        """Return the stack object for context-manager usage.

        Returns
        -------
        RasterStack
            This stack instance.
        """
        return self

    def __exit__(self, exc_type, exc, tb):
        """Close datasets when leaving a context-manager block.

        Parameters
        ----------
        exc_type
            Exception type (if raised in the context block).
        exc
            Exception instance (if raised in the context block).
        tb
            Traceback object (if raised in the context block).

        Returns
        -------
        None
            Always closes datasets.
        """
        self.close()

    def sample_points(self, coords: np.ndarray, fill_method: str | None = None):
        """Sample all stack rasters at point coordinates with cache reuse.

        Parameters
        ----------
        coords : np.ndarray
            `N x 2` query coordinates.
        fill_method : str | None, optional
            Override for missing-value fill mode.

        Returns
        -------
        tuple[np.ndarray, List[str]]
            Concatenated values (`N x K`) and feature names.
        """
        coords = np.asarray(coords)
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError("coords must be N x 2")

        fill = fill_method if fill_method is not None else self.fill_method
        env_blocks = []
        env_names: List[str] = []

        for idx, ds in enumerate(self._datasets):
            xs, ys = _coords_to_dataset(ds, coords, self.coord_order, self.coords_crs)
            data, _ = _sample_raw(ds, xs, ys)
            cache = self._cache.setdefault(idx, {})
            data = _fill_missing(ds, xs, ys, data, fill, cache)
            env_blocks.append(data)
            stem = self.raster_paths[idx].stem
            env_names.extend([f"{stem}_{name}" for name in _band_names(ds)])

        env = np.concatenate(env_blocks, axis=1) if env_blocks else np.empty((0, 0))
        return env, env_names
