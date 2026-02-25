from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.spatial import cKDTree

from multispecies_resistance.data import grid_nodes_from_bbox
from rasterio.crs import CRS
from rasterio.warp import transform


def haversine_km(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute great-circle distances in kilometers for paired coordinates.

    Parameters
    ----------
    a : np.ndarray
        Coordinates with shape `(..., 2)` in `lat, lon`.
    b : np.ndarray
        Coordinates with shape `(..., 2)` in `lat, lon`.

    Returns
    -------
    np.ndarray
        Elementwise distance array with shape `...`.
    """
    lat1 = np.radians(a[..., 0])
    lon1 = np.radians(a[..., 1])
    lat2 = np.radians(b[..., 0])
    lon2 = np.radians(b[..., 1])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    sin_dlat = np.sin(dlat / 2.0)
    sin_dlon = np.sin(dlon / 2.0)

    h = sin_dlat ** 2 + np.cos(lat1) * np.cos(lat2) * sin_dlon ** 2
    h = np.minimum(1.0, np.maximum(0.0, h))
    return 6371.0 * 2.0 * np.arcsin(np.sqrt(h))


def _as_lon_lat(coords: np.ndarray, coord_order: str) -> Tuple[np.ndarray, np.ndarray]:
    """Split coordinates into longitude and latitude arrays.

    Parameters
    ----------
    coords : np.ndarray
        `N x 2` coordinate array.
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


def project_coords(
    coords: np.ndarray,
    coord_order: str = "latlon",
    coords_crs: str | CRS | None = "EPSG:4326",
    target_crs: str | CRS = "EPSG:3857",
) -> np.ndarray:
    """Project input coordinates from `coords_crs` into `target_crs`.

    Parameters
    ----------
    coords : np.ndarray
        `N x 2` coordinate array.
    coord_order : str, optional
        Input order, `"latlon"` or `"lonlat"`.
    coords_crs : str | CRS | None, optional
        CRS of input coordinates.
    target_crs : str | CRS, optional
        Output CRS to project into.

    Returns
    -------
    np.ndarray
        `N x 2` projected coordinates as `x, y`.
    """
    if coords_crs is None:
        raise ValueError("coords_crs is required for projection")

    lons, lats = _as_lon_lat(coords, coord_order)
    src_crs = CRS.from_user_input(coords_crs)
    dst_crs = CRS.from_user_input(target_crs)
    xs, ys = transform(src_crs, dst_crs, lons.tolist(), lats.tolist())
    return np.column_stack([xs, ys]).astype(np.float64)


def build_knn_graph(
    site_coords: np.ndarray,
    k: int = 6,
    project_to: str | CRS | None = None,
    coord_order: str = "latlon",
    coords_crs: str | CRS | None = "EPSG:4326",
) -> np.ndarray:
    """Build an undirected k-nearest-neighbor graph from site coordinates.

    Parameters
    ----------
    site_coords : np.ndarray
        `S x 2` site coordinates.
    k : int, optional
        Number of nearest neighbors per node.
    project_to : str | CRS | None, optional
        Optional projection CRS used before nearest-neighbor search.
    coord_order : str, optional
        Coordinate order of `site_coords`.
    coords_crs : str | CRS | None, optional
        CRS of `site_coords`.

    Returns
    -------
    np.ndarray
        `E x 2` undirected edge list with sorted node indices.
    """
    if k < 1:
        raise ValueError("k must be >= 1")

    coords = site_coords
    if project_to is not None:
        coords = project_coords(
            site_coords, coord_order=coord_order, coords_crs=coords_crs, target_crs=project_to
        )

    tree = cKDTree(coords)
    dists, idx = tree.query(coords, k=k + 1)

    edges = set()
    for i in range(idx.shape[0]):
        for j in idx[i, 1:]:
            if i == j:
                continue
            a, b = (i, j) if i < j else (j, i)
            edges.add((a, b))

    edge_index = np.array(sorted(edges), dtype=np.int64)
    return edge_index


def build_delaunay_graph(
    site_coords: np.ndarray,
    project_to: str | CRS | None = None,
    coord_order: str = "latlon",
    coords_crs: str | CRS | None = "EPSG:4326",
) -> np.ndarray:
    """Build an undirected graph from Delaunay triangulation edges.

    Parameters
    ----------
    site_coords : np.ndarray
        `S x 2` site coordinates.
    project_to : str | CRS | None, optional
        Optional projection CRS used before triangulation.
    coord_order : str, optional
        Coordinate order of `site_coords`.
    coords_crs : str | CRS | None, optional
        CRS of `site_coords`.

    Returns
    -------
    np.ndarray
        `E x 2` undirected edge list with sorted node indices.
    """
    from scipy.spatial import Delaunay

    coords = site_coords
    if project_to is not None:
        coords = project_coords(
            site_coords, coord_order=coord_order, coords_crs=coords_crs, target_crs=project_to
        )

    tri = Delaunay(coords)
    edges = set()
    for simplex in tri.simplices:
        for a, b in ((0, 1), (1, 2), (0, 2)):
            i = int(simplex[a])
            j = int(simplex[b])
            if i == j:
                continue
            u, v = (i, j) if i < j else (j, i)
            edges.add((u, v))

    edge_index = np.array(sorted(edges), dtype=np.int64)
    return edge_index


def build_dense_mesh_graph(
    coords_list: list[np.ndarray],
    spacing_km: float | None = 50.0,
    spacing_deg: float | None = None,
    grid_type: str = "triangular",
    mesh_graph_type: str = "delaunay",
    k: int = 6,
    project_to: str | CRS | None = None,
    coord_order: str = "latlon",
    coords_crs: str | CRS | None = "EPSG:4326",
    buffer_km: float = 0.0,
    bbox: str | None = "square",
    bbox_file: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a shared mesh of nodes and edges covering all species locations.

    Parameters
    ----------
    coords_list : list[np.ndarray]
        Per-species coordinate arrays (`S_i x 2`).
    spacing_km : float | None, optional
        Mesh node spacing in kilometers.
    spacing_deg : float | None, optional
        Mesh node spacing in degrees.
    grid_type : str, optional
        Node layout type, `"triangular"` or `"rect"`.
    mesh_graph_type : str, optional
        Edge construction type, `"delaunay"` or `"knn"`.
    k : int, optional
        `k` used when `mesh_graph_type="knn"`.
    project_to : str | CRS | None, optional
        Optional projection CRS used for graph construction.
    coord_order : str, optional
        Coordinate order of arrays in `coords_list`.
    coords_crs : str | CRS | None, optional
        CRS of coordinates in `coords_list`.
    buffer_km : float, optional
        Bounding-area expansion before mesh generation.
    bbox : str | None, optional
        Bounding shape: `"square"`, `"convex_hull"`, `"polygon"`, or `None`.
    bbox_file : str | None, optional
        Path to polygon coordinates (`lat lon`) when using polygon clipping.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        `mesh_coords` (`M x 2`) and `edge_index` (`E x 2`).
    """
    if not coords_list:
        raise ValueError("coords_list is empty")
    all_coords = np.vstack(coords_list)

    # Convert to lat/lon for bbox logic
    if coord_order == "lonlat":
        all_coords_latlon = all_coords[:, [1, 0]]
    else:
        all_coords_latlon = all_coords

    if buffer_km < 0:
        raise ValueError("buffer_km must be >= 0")

    mean_lat = float(np.mean(all_coords_latlon[:, 0]))
    dlat = buffer_km / 111.0
    dlon = buffer_km / (111.0 * np.cos(np.radians(mean_lat)))

    lat_min, lon_min = np.min(all_coords_latlon, axis=0)
    lat_max, lon_max = np.max(all_coords_latlon, axis=0)
    lat_min -= dlat
    lat_max += dlat
    lon_min -= dlon
    lon_max += dlon

    coords_for_grid = np.vstack(
        [all_coords_latlon, np.array([[lat_min, lon_min], [lat_max, lon_max]])]
    )

    mesh_coords = grid_nodes_from_bbox(
        coords_for_grid,
        spacing_km=spacing_km,
        spacing_deg=spacing_deg,
        grid_type=grid_type,
    )

    if bbox is not None and not isinstance(bbox, str):
        bbox = str(bbox)

    if bbox_file is None and bbox not in {None, "square", "convex_hull", "polygon"}:
        try:
            from pathlib import Path

            if Path(str(bbox)).exists():
                bbox_file = str(bbox)
                bbox = "polygon"
        except Exception:
            pass

    if bbox_file is not None:
        bbox = "polygon"

    if bbox is None:
        bbox = "square"

    bbox = bbox.lower()
    if bbox not in {"square", "convex_hull", "polygon"}:
        raise ValueError("bbox must be 'square', 'convex_hull', or 'polygon'")

    if bbox in {"convex_hull", "polygon"}:
        import geopandas as gpd
        from shapely.geometry import Point

        if bbox == "polygon":
            if bbox_file is None:
                raise ValueError("bbox_file is required when bbox='polygon'")
            poly_coords = np.loadtxt(bbox_file)
            if poly_coords.ndim != 2 or poly_coords.shape[1] != 2:
                raise ValueError("bbox_file must contain two columns (lat lon)")
            if not np.allclose(poly_coords[0], poly_coords[-1]):
                raise ValueError("bbox_file polygon must be closed (first and last point identical)")
            from shapely.geometry import Polygon

            polygon = Polygon([(lon, lat) for lat, lon in poly_coords])
            gseries = gpd.GeoSeries([polygon], crs=coords_crs)
        else:
            gseries = gpd.GeoSeries(
                [Point(lon, lat) for lat, lon in all_coords_latlon], crs=coords_crs
            )

        proj_crs = CRS.from_user_input("EPSG:3857")
        gseries_proj = gseries.to_crs(proj_crs)
        if bbox == "polygon":
            hull = gseries_proj.unary_union
        else:
            hull = gseries_proj.unary_union.convex_hull
        if buffer_km > 0:
            hull = hull.buffer(buffer_km * 1000.0)

        mesh_series = gpd.GeoSeries(
            [Point(lon, lat) for lat, lon in mesh_coords], crs=coords_crs
        ).to_crs(proj_crs)
        mask = mesh_series.within(hull) | mesh_series.touches(hull)
        mesh_coords = mesh_coords[np.array(mask)]

    if mesh_coords.size == 0:
        raise ValueError("Mesh generation produced zero nodes. Check bbox/buffer/grid spacing.")

    mesh_graph_type = mesh_graph_type.lower()
    if mesh_graph_type == "delaunay":
        edge_index = build_delaunay_graph(
            mesh_coords,
            project_to=project_to,
            coord_order=coord_order,
            coords_crs=coords_crs,
        )
    elif mesh_graph_type == "knn":
        edge_index = build_knn_graph(
            mesh_coords,
            k=k,
            project_to=project_to,
            coord_order=coord_order,
            coords_crs=coords_crs,
        )
    else:
        raise ValueError("mesh_graph_type must be 'delaunay' or 'knn'")

    return mesh_coords, edge_index


def edge_features(
    site_coords: np.ndarray,
    site_env: np.ndarray,
    edge_index: np.ndarray,
) -> np.ndarray:
    """Compute edge features from geodesic length and environmental differences.

    Parameters
    ----------
    site_coords : np.ndarray
        `S x 2` site coordinates in `lat, lon`.
    site_env : np.ndarray
        `S x K` site covariates, or empty array for no covariates.
    edge_index : np.ndarray
        `E x 2` edge list indexing site rows.

    Returns
    -------
    np.ndarray
        `E x (1+K)` matrix with distance in column 0 and absolute env diffs after.
    """
    a = site_coords[edge_index[:, 0]]
    b = site_coords[edge_index[:, 1]]

    geo_dist = haversine_km(a, b)[:, None]
    if site_env is None or site_env.size == 0:
        env_diff = np.zeros((edge_index.shape[0], 0), dtype=np.float64)
    else:
        env_diff = np.abs(site_env[edge_index[:, 0]] - site_env[edge_index[:, 1]])

    feats = np.concatenate([geo_dist, env_diff], axis=1)
    return feats.astype(np.float64)


def standardize_features(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize feature columns to zero mean and unit variance.

    Parameters
    ----------
    x : np.ndarray
        Feature matrix with shape `N x K`.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Standardized matrix, column means, and column standard deviations.
    """
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std == 0.0, 1.0, std)
    return (x - mean) / std, mean, std
