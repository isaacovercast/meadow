from __future__ import annotations

import re
import shutil
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import rasterio
from rasterio.crs import CRS

from multispecies_resistance.raster import RasterStack

WORLDCLIM_BASE_URL = "https://geodata.ucdavis.edu/climate/worldclim/2_1/base"

_WORLDCLIM_GROUP_BANDS = {
    "bio": 19,
    "prec": 12,
    "tavg": 12,
    "tmax": 12,
    "tmin": 12,
    "srad": 12,
    "wind": 12,
    "vapr": 12,
}
_WORLDCLIM_GROUPS = tuple(_WORLDCLIM_GROUP_BANDS.keys())
_BIO_VARS = tuple(f"bio{i}" for i in range(1, 20))
_SUPPORTED_RESOLUTIONS = {"30s", "2.5m", "5m", "10m"}


def _dedupe(items: Sequence[str]) -> List[str]:
    """Return items in first-seen order with duplicates removed.

    Parameters
    ----------
    items : Sequence[str]
        Input string sequence.

    Returns
    -------
    List[str]
        Deduplicated sequence preserving order.
    """
    out: List[str] = []
    seen = set()
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _normalize_bio_name(value: str) -> str | None:
    """Normalize a BioClim variable token to canonical `bioN` format.

    Parameters
    ----------
    value : str
        Candidate variable token such as `bio1` or `bio_1`.

    Returns
    -------
    str | None
        Canonical variable name if valid, else `None`.
    """
    m = re.fullmatch(r"bio_?(\d{1,2})", value.lower())
    if m is None:
        return None
    idx = int(m.group(1))
    if 1 <= idx <= 19:
        return f"bio{idx}"
    return None


def _normalize_month_name(value: str) -> str | None:
    """Normalize a monthly climate token to canonical `<group>_MM` format.

    Parameters
    ----------
    value : str
        Candidate token such as `tavg7` or `tavg_07`.

    Returns
    -------
    str | None
        Canonical monthly variable name if valid, else `None`.
    """
    m = re.fullmatch(r"(prec|tavg|tmax|tmin|srad|wind|vapr)_?(\d{1,2})", value.lower())
    if m is None:
        return None
    month = int(m.group(2))
    if 1 <= month <= 12:
        return f"{m.group(1)}_{month:02d}"
    return None


def _expand_group(group: str) -> List[str]:
    """Expand a climate group token into all canonical variable names.

    Parameters
    ----------
    group : str
        Group key such as `bio`, `tavg`, or `prec`.

    Returns
    -------
    List[str]
        Variable names associated with that group.
    """
    if group == "bio":
        return list(_BIO_VARS)
    n = _WORLDCLIM_GROUP_BANDS[group]
    return [f"{group}_{i:02d}" for i in range(1, n + 1)]


def _parse_climate_request(
    source: str,
    variables: Sequence[str] | None,
) -> Tuple[List[str], List[str]]:
    """Validate a climate variable request and resolve required groups/bands.

    Parameters
    ----------
    source : str
        Data source name, either `"bioclim"` or `"worldclim"`.
    variables : Sequence[str] | None
        Requested variable or group tokens.

    Returns
    -------
    Tuple[List[str], List[str]]
        Download groups and canonical variable names requested.
    """
    source_norm = source.lower()
    if source_norm not in {"worldclim", "bioclim"}:
        raise ValueError("source must be 'bioclim' or 'worldclim'")

    if variables is None:
        if source_norm == "bioclim":
            return ["bio"], list(_BIO_VARS)
        return ["bio"], _expand_group("bio")

    groups: List[str] = []
    requested: List[str] = []
    for raw in variables:
        token = str(raw).strip().lower()
        if not token:
            continue

        bio_token = _normalize_bio_name(token)
        month_token = _normalize_month_name(token)

        if source_norm == "bioclim":
            if token == "bio":
                groups.append("bio")
                requested.extend(_expand_group("bio"))
            elif bio_token is not None:
                groups.append("bio")
                requested.append(bio_token)
            else:
                raise ValueError(
                    "For source='bioclim', variables must be bio1..bio19 (or 'bio')."
                )
            continue

        # source='worldclim'
        if token in _WORLDCLIM_GROUPS:
            groups.append(token)
            requested.extend(_expand_group(token))
        elif bio_token is not None:
            groups.append("bio")
            requested.append(bio_token)
        elif month_token is not None:
            group = month_token.split("_", 1)[0]
            groups.append(group)
            requested.append(month_token)
        else:
            raise ValueError(
                "Unknown worldclim variable. Use group names (bio, prec, tavg, tmax, "
                "tmin, srad, wind, vapr) or band names like bio12 / tavg_07."
            )

    groups = _dedupe(groups)
    requested = _dedupe(requested)
    if not groups:
        raise ValueError("No valid variables requested.")
    if not requested:
        for group in groups:
            requested.extend(_expand_group(group))
        requested = _dedupe(requested)
    return groups, requested


def _sort_key(path: Path) -> Tuple[str, int, str]:
    """Generate deterministic sort key for climate raster filenames.

    Parameters
    ----------
    path : Path
        Raster file path.

    Returns
    -------
    Tuple[str, int, str]
        `(prefix, numeric_suffix, stem)` key for stable ordering.
    """
    stem = path.stem.lower()
    m = re.search(r"_(\d+)$", stem)
    if m is None:
        return stem, -1, stem
    prefix = stem[: m.start()]
    return prefix, int(m.group(1)), stem


def _safe_extract_zip(zip_path: Path, dst: Path) -> None:
    """Extract a ZIP archive only if all members stay inside destination root.

    Parameters
    ----------
    zip_path : Path
        ZIP archive path.
    dst : Path
        Destination extraction directory.

    Returns
    -------
    None
        Extracts files in-place or raises on unsafe member paths.
    """
    dst_resolved = dst.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            member_path = (dst_resolved / member).resolve()
            if not member_path.is_relative_to(dst_resolved):
                raise ValueError(f"Unsafe zip member path: {member}")
        zf.extractall(dst_resolved)


def _download_file(url: str, dst: Path, timeout: int) -> None:
    """Download a URL to disk with a user-agent header and timeout.

    Parameters
    ----------
    url : str
        Source URL.
    dst : Path
        Output file path.
    timeout : int
        Request timeout in seconds.

    Returns
    -------
    None
        Writes the downloaded file to `dst`.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "multispecies-resistance/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(dst, "wb") as f:
        shutil.copyfileobj(resp, f)


def _find_group_rasters(extract_dir: Path, resolution: str, group: str) -> List[Path]:
    """Locate extracted GeoTIFFs for a climate group under a cache directory.

    Parameters
    ----------
    extract_dir : Path
        Directory where group files were extracted.
    resolution : str
        WorldClim resolution token used in filename matching.
    group : str
        Climate group token.

    Returns
    -------
    List[Path]
        Matching raster paths sorted deterministically.
    """
    prefix = f"wc2.1_{resolution}_{group}"
    matches = sorted(extract_dir.rglob(f"{prefix}*.tif"), key=_sort_key)
    if matches:
        return matches
    return sorted(extract_dir.rglob("*.tif"), key=_sort_key)


def download_climate_layers(
    source: str = "bioclim",
    variables: Sequence[str] | None = None,
    resolution: str = "2.5m",
    cache_dir: str | Path = "~/.cache/multispecies_resistance/climate",
    force_download: bool = False,
    base_url: str | None = None,
    timeout: int = 120,
) -> List[Path]:
    """Download, extract, and cache climate rasters for requested variables.

    Parameters
    ----------
    source : str, optional
        Climate source, `"bioclim"` or `"worldclim"`.
    variables : Sequence[str] | None, optional
        Requested variable or group names. `None` uses default source groups.
    resolution : str, optional
        WorldClim resolution token.
    cache_dir : str | Path, optional
        Root directory for downloads and extracted rasters.
    force_download : bool, optional
        If `True`, clear cached group files before re-downloading.
    base_url : str | None, optional
        Override base URL for file retrieval.
    timeout : int, optional
        Download timeout in seconds.

    Returns
    -------
    List[Path]
        Resolved GeoTIFF paths in deterministic order.
    """
    if resolution not in _SUPPORTED_RESOLUTIONS:
        raise ValueError(
            f"Unsupported resolution '{resolution}'. Supported: {sorted(_SUPPORTED_RESOLUTIONS)}"
        )
    if timeout <= 0:
        raise ValueError("timeout must be > 0")

    groups, _ = _parse_climate_request(source, variables)
    source_norm = source.lower()
    root = Path(cache_dir).expanduser().resolve()
    base = (base_url or WORLDCLIM_BASE_URL).rstrip("/") + "/"

    out_paths: List[Path] = []
    for group in groups:
        extract_dir = root / source_norm / "wc2.1" / resolution / group
        existing = _find_group_rasters(extract_dir, resolution=resolution, group=group)
        if existing and not force_download:
            out_paths.extend(existing)
            continue

        if force_download and extract_dir.exists():
            shutil.rmtree(extract_dir)
        extract_dir.mkdir(parents=True, exist_ok=True)

        base_name = f"wc2.1_{resolution}_{group}"
        archive_path = root / source_norm / "downloads" / f"{base_name}.zip"
        tif_tmp_path = root / source_norm / "downloads" / f"{base_name}.tif"

        downloaded = None
        download_errors: List[str] = []
        for suffix, dst in (("zip", archive_path), ("tif", tif_tmp_path)):
            url = urllib.parse.urljoin(base, f"{base_name}.{suffix}")
            try:
                _download_file(url, dst, timeout=timeout)
                downloaded = dst
                break
            except urllib.error.URLError as exc:
                download_errors.append(f"{url}: {exc}")

        if downloaded is None:
            joined = "; ".join(download_errors)
            raise RuntimeError(f"Failed to download '{base_name}' from {base}. {joined}")

        if downloaded.suffix.lower() == ".zip":
            _safe_extract_zip(downloaded, extract_dir)
        else:
            target = extract_dir / f"{base_name}.tif"
            shutil.move(str(downloaded), str(target))

        found = _find_group_rasters(extract_dir, resolution=resolution, group=group)
        if not found:
            raise FileNotFoundError(
                f"Downloaded {base_name} but found no GeoTIFFs under {extract_dir}."
            )
        out_paths.extend(found)

    # Keep deterministic order while preserving group order.
    out_paths = _dedupe([str(p.resolve()) for p in out_paths])
    return [Path(p) for p in out_paths]


def _single_band_name_from_stem(stem: str) -> str:
    """Infer canonical variable name from a single-band raster filename stem.

    Parameters
    ----------
    stem : str
        Raster filename stem.

    Returns
    -------
    str
        Best-effort canonical variable name.
    """
    lower = stem.lower()

    bio_named = _normalize_bio_name(lower)
    if bio_named is not None:
        return bio_named

    month_named = _normalize_month_name(lower)
    if month_named is not None:
        return month_named

    m = re.search(r"(?:^|_)bio_?(\d{1,2})$", lower)
    if m is not None:
        idx = int(m.group(1))
        if 1 <= idx <= 19:
            return f"bio{idx}"

    m = re.search(r"(?:^|_)(prec|tavg|tmax|tmin|srad|wind|vapr)_?(\d{1,2})$", lower)
    if m is not None:
        month = int(m.group(2))
        if 1 <= month <= 12:
            return f"{m.group(1)}_{month:02d}"

    for group in _WORLDCLIM_GROUPS:
        if re.search(rf"(?:^|_){group}(?:$|_)", lower):
            return group
    return lower


def _multiband_group_from_stem(stem: str) -> str | None:
    """Infer climate group token from a multiband raster filename stem.

    Parameters
    ----------
    stem : str
        Raster filename stem.

    Returns
    -------
    str | None
        Group token if detected, otherwise `None`.
    """
    lower = stem.lower()
    for group in _WORLDCLIM_GROUPS:
        if re.search(rf"(?:^|_){group}(?:$|_)", lower):
            return group
    return None


def _canonical_layer_names(path: Path, band_count: int) -> List[str]:
    """Generate canonical layer names for a raster's bands.

    Parameters
    ----------
    path : Path
        Raster file path.
    band_count : int
        Number of raster bands.

    Returns
    -------
    List[str]
        Canonical names for each band in order.
    """
    if band_count < 1:
        return []

    if band_count == 1:
        return [_single_band_name_from_stem(path.stem)]

    group = _multiband_group_from_stem(path.stem)
    if group == "bio":
        return [f"bio{i}" for i in range(1, band_count + 1)]
    if group in _WORLDCLIM_GROUP_BANDS:
        return [f"{group}_{i:02d}" for i in range(1, band_count + 1)]
    return [f"{path.stem}_band_{i}" for i in range(1, band_count + 1)]


def _infer_layer_names(raster_paths: Iterable[str | Path]) -> List[str]:
    """Infer concatenated layer names across a sequence of rasters.

    Parameters
    ----------
    raster_paths : Iterable[str | Path]
        Raster files used for sampling.

    Returns
    -------
    List[str]
        Layer names aligned to concatenated sampled columns.
    """
    names: List[str] = []
    for path in raster_paths:
        with rasterio.open(path) as src:
            names.extend(_canonical_layer_names(Path(path), src.count))
    return names


def sample_climate_for_sites(
    site_coords: np.ndarray,
    source: str = "bioclim",
    variables: Sequence[str] | None = None,
    resolution: str = "2.5m",
    cache_dir: str | Path = "~/.cache/multispecies_resistance/climate",
    coord_order: str = "latlon",
    coords_crs: str | CRS | None = "EPSG:4326",
    fill_method: str = "nearest",
    force_download: bool = False,
    base_url: str | None = None,
    timeout: int = 120,
) -> Tuple[np.ndarray, List[str], List[Path]]:
    """Download requested climate layers and sample them at site coordinates.

    Parameters
    ----------
    site_coords : np.ndarray
        `N x 2` coordinates for sampling.
    source : str, optional
        Climate source, `"bioclim"` or `"worldclim"`.
    variables : Sequence[str] | None, optional
        Requested variables or groups. `None` keeps all downloaded layers.
    resolution : str, optional
        WorldClim resolution token.
    cache_dir : str | Path, optional
        Cache root for downloaded/extracted files.
    coord_order : str, optional
        Coordinate order of `site_coords`.
    coords_crs : str | CRS | None, optional
        CRS of `site_coords`.
    fill_method : str, optional
        Missing-value fill mode for raster sampling.
    force_download : bool, optional
        Whether to force re-download of cached groups.
    base_url : str | None, optional
        Optional base URL override for downloads.
    timeout : int, optional
        Download timeout in seconds.

    Returns
    -------
    Tuple[np.ndarray, List[str], List[Path]]
        Sample matrix (`N x K`), variable names, and raster files used.
    """
    raster_paths = download_climate_layers(
        source=source,
        variables=variables,
        resolution=resolution,
        cache_dir=cache_dir,
        force_download=force_download,
        base_url=base_url,
        timeout=timeout,
    )
    requested_groups, requested_names = _parse_climate_request(source, variables)
    _ = requested_groups  # keep validation behavior consistent with downloader.

    with RasterStack(
        raster_paths,
        coord_order=coord_order,
        coords_crs=coords_crs,
        fill_method=fill_method,
    ) as stack:
        site_env, _ = stack.sample_points(site_coords)

    inferred_names = _infer_layer_names(raster_paths)
    if site_env.shape[1] != len(inferred_names):
        raise RuntimeError(
            "Internal naming mismatch: sampled columns do not match inferred raster layers."
        )

    if variables is None:
        return site_env, inferred_names, raster_paths

    name_to_idx = {name: i for i, name in enumerate(inferred_names)}
    missing = [name for name in requested_names if name not in name_to_idx]
    if missing:
        raise ValueError(
            "Requested variables were not found in downloaded rasters: "
            + ", ".join(missing)
        )

    idx = [name_to_idx[name] for name in requested_names]
    subset = site_env[:, idx]
    return subset, requested_names, raster_paths
