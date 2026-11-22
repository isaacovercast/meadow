import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio import Affine
from rasterio.warp import transform

from meadow.raster import RasterStack, sample_raster_at_points
from meadow.train import _sample_node_env


def _write_raster(path: Path, data: np.ndarray, nodata: float) -> None:
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=data.shape[1],
        width=data.shape[2],
        count=data.shape[0],
        dtype=data.dtype,
        crs='EPSG:4326',
        transform=from_origin(0.0, 3.0, 1.0, 1.0),
        nodata=nodata,
    ) as dst:
        dst.write(data)


class TestRasterNoData(unittest.TestCase):
    def test_sampling_preserves_per_band_nodata_masks(self):
        for dtype, nodata in [
            ('float32', float(np.finfo(np.float32).min)),
            ('int16', -32768),
        ]:
            with self.subTest(dtype=dtype), tempfile.TemporaryDirectory() as tmp:
                data = np.array(
                    [
                        [[10, nodata, 30], [40, 50, 60], [70, 80, 90]],
                        [[100, 200, 300], [nodata, 500, 600], [700, 800, 900]],
                    ],
                    dtype=dtype,
                )
                path = Path(tmp) / 'masked.tif'
                _write_raster(path, data, nodata)
                # Each of the first two points is missing in a different band.
                coords = np.array([[2.5, 1.5], [1.5, 0.5], [1.5, 1.5]])
                sampled, _ = sample_raster_at_points(path, coords, fill_method='nan')
                expected = np.array([[np.nan, 200], [40, np.nan], [50, 500]])
                np.testing.assert_allclose(sampled, expected, equal_nan=True)
                self.assertEqual(sampled.dtype, np.float64)

    def test_nearest_fill_replaces_nodata_without_changing_valid_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            nodata = float(np.finfo(np.float32).min)
            data = np.array(
                [[[nodata, 20, 30], [40, 50, 60], [70, 80, 90]]], dtype=np.float32
            )
            path = Path(tmp) / 'nearest.tif'
            _write_raster(path, data, nodata)
            # The missing point is uniquely closest to the pixel valued 20.
            coords = np.array([[2.4, 0.9], [1.5, 1.5]])
            sampled, _ = sample_raster_at_points(path, coords, fill_method='nearest')
            np.testing.assert_allclose(sampled[:, 0], [20, 50])


class _WindowReadGuard:
    """Fail a test if fill indexing tries to read a whole raster."""

    def __init__(self, src):
        self.src = src
        self.windows = []

    def read(self, *args, **kwargs):
        window = kwargs.get('window')
        if window is None:
            raise AssertionError('Expected a regional window for raster reads')
        self.windows.append(window)
        return self.src.read(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.src, name)


def _write_region(path, data, affine, crs='EPSG:4326'):
    with rasterio.open(
        path, 'w', driver='GTiff', height=data.shape[0], width=data.shape[1],
        count=1, dtype=data.dtype, crs=crs, transform=affine, nodata=-9999,
    ) as dst:
        dst.write(data, 1)


def _point(affine, col, row):
    x, y = affine * (col, row)
    return np.array([[y, x]])


class TestRegionalRasterSampling(unittest.TestCase):
    def test_graph_sampling_uses_regional_reads_and_reuses_the_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'large.tif'
            affine = from_origin(-10, 10, 0.01, 0.01)
            data = np.arange(400 * 400, dtype=np.float32).reshape(400, 400)
            data[198:203, 198:203] = -9999
            _write_region(path, data, affine)
            coords = _point(affine, 200.2, 200.3)
            expected, _ = sample_raster_at_points(path, coords, fill_method='nearest')
            with RasterStack([path], fill_method='nearest') as stack:
                guard = _WindowReadGuard(stack._datasets[0])
                stack._datasets[0] = guard
                actual = _sample_node_env(stack, coords, len(coords))
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(len(guard.windows), 1)
                self.assertLess(guard.windows[0].width, 100)
                self.assertLess(guard.windows[0].height, 100)
                again, _ = stack.sample_points(coords)
                np.testing.assert_array_equal(again, expected)
                self.assertEqual(len(guard.windows), 1)

    def test_crop_expands_when_a_closer_pixel_is_outside_the_initial_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'expand.tif'
            affine = from_origin(-10, 10, 0.01, 0.01)
            data = np.full((100, 100), -9999, dtype=np.float32)
            data[52, 52] = 1  # Inside the initial crop, but farther from query.
            data[50, 47] = 2  # Outside the crop and the true nearest valid pixel.
            _write_region(path, data, affine)
            coords = _point(affine, 50.1, 50.1)
            with RasterStack([path], fill_method='nearest') as stack:
                guard = _WindowReadGuard(stack._datasets[0])
                stack._datasets[0] = guard
                stack.crop_to_points(coords, padding_pixels=2)
                values, _ = stack.sample_points(coords)
                self.assertEqual(values[0, 0], 2)
                self.assertGreaterEqual(len(guard.windows), 2)

    def test_new_region_clears_indexes_from_the_old_region(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'regions.tif'
            affine = from_origin(-10, 10, 0.01, 0.01)
            data = np.arange(400 * 400, dtype=np.float32).reshape(400, 400)
            for i in (60, 340):
                data[i, i] = -9999
            _write_region(path, data, affine)
            with RasterStack([path], fill_method='nearest') as stack:
                answers = []
                for i in (60, 340):
                    coords = _point(affine, i + 0.2, i + 0.3)
                    expected, _ = sample_raster_at_points(path, coords, fill_method='nearest')
                    stack.crop_to_points(coords)
                    actual, _ = stack.sample_points(coords)
                    np.testing.assert_array_equal(actual, expected)
                    answers.append(actual[0, 0])
                self.assertNotEqual(*answers)

    def test_projected_and_rotated_rasters_match_uncropped_sampling(self):
        cases = [
            ('EPSG:3857', from_origin(-20000, 20000, 100, 200)),
            ('EPSG:4326', Affine.translation(-10, 10) * Affine.rotation(15) * Affine.scale(0.01, -0.02)),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            for k, (crs, affine) in enumerate(cases):
                with self.subTest(crs=crs):
                    path = Path(tmp) / f'case{k}.tif'
                    data = np.arange(200 * 200, dtype=np.float32).reshape(200, 200)
                    data[100, 100] = -9999
                    _write_region(path, data, affine, crs=crs)
                    xy = _point(affine, 100.2, 100.3)
                    x, y = transform(crs, 'EPSG:4326', xy[:, 1], xy[:, 0])
                    coords = np.column_stack([y, x])
                    expected, _ = sample_raster_at_points(path, coords, fill_method='nearest')
                    with RasterStack([path], fill_method='nearest') as stack:
                        stack.crop_to_points(coords)
                        actual, _ = stack.sample_points(coords)
                    np.testing.assert_array_equal(actual, expected)

    def test_nearest_expands_past_an_empty_crop_and_handles_an_empty_raster(self):
        with tempfile.TemporaryDirectory() as tmp:
            affine = from_origin(-10, 10, 0.01, 0.01)
            coords = _point(affine, 50.1, 50.1)
            for empty in (False, True):
                with self.subTest(empty=empty):
                    path = Path(tmp) / f'empty{empty}.tif'
                    data = np.full((100, 100), -9999, dtype=np.float32)
                    if not empty:
                        data[5, 5] = 7
                    _write_region(path, data, affine)
                    with RasterStack([path], fill_method='nearest') as stack:
                        stack.crop_to_points(coords, padding_pixels=2)
                        values, _ = stack.sample_points(coords)
                    if empty:
                        self.assertTrue(np.isnan(values[0, 0]))
                    else:
                        self.assertEqual(values[0, 0], 7)

    def test_mean_fill_uses_the_regional_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'mean.tif'
            affine = from_origin(-10, 10, 0.01, 0.01)
            data = np.arange(200 * 200, dtype=np.float32).reshape(200, 200)
            data[50, 50] = -9999
            _write_region(path, data, affine)
            coords = _point(affine, 50.2, 50.3)
            with RasterStack([path], fill_method='mean') as stack:
                guard = _WindowReadGuard(stack._datasets[0])
                stack._datasets[0] = guard
                stack.crop_to_points(coords)
                window = stack._windows[0]
                expected = guard.src.read(1, masked=True, window=window).mean()
                values, _ = stack.sample_points(coords)
                self.assertAlmostEqual(values[0, 0], expected)
                self.assertEqual(len(guard.windows), 1)


if __name__ == '__main__':
    unittest.main()
