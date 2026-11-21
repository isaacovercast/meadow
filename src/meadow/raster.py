from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.warp import transform
from rasterio.windows import Window
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
    window : Window | None, optional
        Regional window; `None` reads the full raster.

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
        # Preserve per-band masks and allow NaN fills for integer rasters.
        samples = np.ma.vstack(list(src.sample(zip(xs, ys), masked=True))).astype(
            np.float64
        )
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


def _band_means(
    src: rasterio.DatasetReader, window: Window | None = None
) -> np.ndarray:
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
        data = src.read(b, masked=True, window=window)
        if np.ma.is_masked(data):
            mean_val = float(data.mean())
        else:
            mean_val = float(np.mean(data))
        means.append(mean_val)
    return np.array(means, dtype=np.float64)


def _point_window(
    src: rasterio.DatasetReader,
    xs: np.ndarray,
    ys: np.ndarray,
    padding_pixels: int,
) -> Window:
    """Enclose query pixels and a margin, clipped to the raster extent."""
    cols, rows = ~src.transform * (xs, ys)
    col0 = int(np.clip(np.floor(np.min(cols)), 0, src.width - 1))
    col1 = int(np.clip(np.floor(np.max(cols)) + 1, 1, src.width))
    row0 = int(np.clip(np.floor(np.min(rows)), 0, src.height - 1))
    row1 = int(np.clip(np.floor(np.max(rows)) + 1, 1, src.height))
    col0, row0 = max(0, col0 - padding_pixels), max(0, row0 - padding_pixels)
    col1 = min(src.width, col1 + padding_pixels)
    row1 = min(src.height, row1 + padding_pixels)
    return Window(col0, row0, col1 - col0, row1 - row0)


def _expand_window(src: rasterio.DatasetReader, window: Window) -> Window:
    """Grow a regional search geometrically until it reaches raster bounds."""
    pad_x = max(32, int(np.ceil(window.width / 2)))
    pad_y = max(32, int(np.ceil(window.height / 2)))
    col0, row0 = max(0, window.col_off - pad_x), max(0, window.row_off - pad_y)
    col1 = min(src.width, window.col_off + window.width + pad_x)
    row1 = min(src.height, window.row_off + window.height + pad_y)
    return Window(col0, row0, col1 - col0, row1 - row0)


def _outside_window_distance(
    src: rasterio.DatasetReader, window: Window, xs: np.ndarray, ys: np.ndarray
) -> np.ndarray:
    """Lower-bound distance to any pixel excluded by a crop, in raster units.

    Pixel-space distances to uncropped sides are infinite. The affine map's
    smallest singular value converts pixel-space bounds conservatively, also
    for rotated, sheared or non-square pixels. A nearer candidate than this
    bound is therefore also the nearest candidate in the full raster.
    """
    cols, rows = ~src.transform * (xs, ys)
    distance = np.full(len(xs), np.inf)
    col1, row1 = window.col_off + window.width, window.row_off + window.height
    if window.col_off > 0:
        distance = np.minimum(distance, np.maximum(0.0, cols - window.col_off))
    if col1 < src.width:
        distance = np.minimum(distance, np.maximum(0.0, col1 - cols))
    if window.row_off > 0:
        distance = np.minimum(distance, np.maximum(0.0, rows - window.row_off))
    if row1 < src.height:
        distance = np.minimum(distance, np.maximum(0.0, row1 - rows))
    affine = src.transform
    scale = np.linalg.svd(
        [[affine.a, affine.b], [affine.d, affine.e]], compute_uv=False
    )[-1]
    return distance * scale


def _build_kdtree(
    src: rasterio.DatasetReader, band: int, window: Window | None = None
):
    """Build a nearest-valid-pixel index, optionally over a regional window."""
    data = src.read(band, masked=True, window=window)
    mask = np.ma.getmaskarray(data) | ~np.isfinite(data.data)
    values = data.data[~mask]
    if values.size == 0:
        return None, None
    rows, cols = np.where(~mask)
    affine = src.transform if window is None else src.window_transform(window)
    xs, ys = rasterio.transform.xy(affine, rows, cols, offset="center")
    tree = cKDTree(np.column_stack([xs, ys]))
    return tree, values.astype(np.float64)


def _fill_missing(
    src: rasterio.DatasetReader,
    xs: np.ndarray,
    ys: np.ndarray,
    data: np.ndarray,
    fill_method: str,
    cache: Dict,
    window: Window | None = None,
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
    window : Window | None, optional
        Initial region for filling; nearest searches expand if necessary.

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
            cache["means"] = _band_means(src, window=window)
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
        q = np.column_stack([xs[mask], ys[mask]])
        entry = cache["kdtree"].get(key)
        if entry is None:
            search_window = window
            tree, values = _build_kdtree(src, b + 1, window=search_window)
        else:
            search_window, tree, values = entry

        while True:
            cache["kdtree"][key] = (search_window, tree, values)
            if tree is not None:
                distances, idx = tree.query(q, k=1)
                if search_window is None or np.all(
                    distances < _outside_window_distance(
                        src, search_window, q[:, 0], q[:, 1]
                    )
                ):
                    filled[mask, b] = values[idx]
                    break
            if search_window is None:
                break  # The full raster contains no valid pixels.
            expanded = _expand_window(src, search_window)
            if expanded == search_window:
                break  # Already searched the full raster, with no valid pixels.
            search_window = expanded
            tree, values = _build_kdtree(src, b + 1, window=search_window)
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
        self._windows: Dict[int, Window] = {}

    def crop_to_points(self, coords: np.ndarray, padding_pixels: int = 32) -> None:
        """Restrict raster reads for filling to the query extent plus a margin.

        Bounds are computed independently in each raster's pixel coordinates;
        source files, pixel values and resolution remain unchanged. Nearest
        searches expand automatically if a closer pixel could lie outside the
        crop. Calling this again clears indexes and means for the old region.
        """
        coords = np.asarray(coords)
        if coords.ndim != 2 or coords.shape[1] != 2 or len(coords) == 0:
            raise ValueError("crop coordinates must be a nonempty N x 2 array")
        if not np.isfinite(coords).all():
            raise ValueError("crop coordinates must be finite")
        if not isinstance(padding_pixels, (int, np.integer)) or padding_pixels < 0:
            raise ValueError("padding_pixels must be a nonnegative integer")
        windows = {}
        for idx, ds in enumerate(self._datasets):
            xs, ys = _coords_to_dataset(ds, coords, self.coord_order, self.coords_crs)
            if not (np.isfinite(xs).all() and np.isfinite(ys).all()):
                raise ValueError(
                    f"Cannot transform crop coordinates for {self.raster_paths[idx]}"
                )
            windows[idx] = _point_window(ds, xs, ys, padding_pixels)
        self._windows = windows
        self._cache.clear()

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
            data = _fill_missing(
                ds, xs, ys, data, fill, cache, window=self._windows.get(idx)
            )
            env_blocks.append(data)
            stem = self.raster_paths[idx].stem
            env_names.extend([f"{stem}_{name}" for name in _band_names(ds)])

        env = np.concatenate(env_blocks, axis=1) if env_blocks else np.empty((0, 0))
        return env, env_names
