from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from multispecies_resistance.climate import download_climate_layers, sample_climate_for_sites


def _write_tif(path: Path, value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.full((2, 2), value, dtype=np.float32)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=2,
        width=2,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(0.0, 2.0, 1.0, 1.0),
    ) as dst:
        dst.write(data, 1)


def _zip_files(zip_path: Path, files: list[Path]) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file in files:
            zf.write(file, arcname=file.name)


def test_download_climate_layers_uses_cache_without_redownloading(tmp_path: Path) -> None:
    upstream = tmp_path / "upstream"
    staged = tmp_path / "staged"
    cache = tmp_path / "cache"

    tif = staged / "wc2.1_2.5m_bio_1.tif"
    _write_tif(tif, value=1.0)
    archive = upstream / "wc2.1_2.5m_bio.zip"
    _zip_files(archive, [tif])

    first = download_climate_layers(
        source="bioclim",
        variables=["bio1"],
        resolution="2.5m",
        cache_dir=cache,
        base_url=upstream.as_uri(),
    )
    assert len(first) == 1
    assert first[0].exists()

    archive.unlink()
    second = download_climate_layers(
        source="bioclim",
        variables=["bio1"],
        resolution="2.5m",
        cache_dir=cache,
        base_url=upstream.as_uri(),
    )
    assert second == first


def test_sample_climate_for_sites_bioclim_subset(tmp_path: Path) -> None:
    upstream = tmp_path / "upstream"
    staged = tmp_path / "staged"

    bio1 = staged / "wc2.1_2.5m_bio_1.tif"
    bio12 = staged / "wc2.1_2.5m_bio_12.tif"
    _write_tif(bio1, value=1.0)
    _write_tif(bio12, value=12.0)
    _zip_files(upstream / "wc2.1_2.5m_bio.zip", [bio1, bio12])

    coords = np.array([[1.5, 0.5], [0.5, 1.5]], dtype=np.float64)  # lat/lon
    env, names, paths = sample_climate_for_sites(
        coords,
        source="bioclim",
        variables=["bio12"],
        resolution="2.5m",
        cache_dir=tmp_path / "cache",
        base_url=upstream.as_uri(),
        coord_order="latlon",
        coords_crs="EPSG:4326",
    )
    assert env.shape == (2, 1)
    assert names == ["bio12"]
    assert np.allclose(env[:, 0], 12.0)
    assert len(paths) == 2


def test_sample_climate_for_sites_worldclim_monthly_band(tmp_path: Path) -> None:
    upstream = tmp_path / "upstream"
    staged = tmp_path / "staged"

    tavg1 = staged / "wc2.1_2.5m_tavg_01.tif"
    tavg2 = staged / "wc2.1_2.5m_tavg_02.tif"
    _write_tif(tavg1, value=10.0)
    _write_tif(tavg2, value=20.0)
    _zip_files(upstream / "wc2.1_2.5m_tavg.zip", [tavg1, tavg2])

    coords = np.array([[1.5, 0.5]], dtype=np.float64)  # lat/lon
    env, names, _ = sample_climate_for_sites(
        coords,
        source="worldclim",
        variables=["tavg_02"],
        resolution="2.5m",
        cache_dir=tmp_path / "cache",
        base_url=upstream.as_uri(),
        coord_order="latlon",
        coords_crs="EPSG:4326",
    )
    assert env.shape == (1, 1)
    assert names == ["tavg_02"]
    assert np.isclose(env[0, 0], 20.0)
