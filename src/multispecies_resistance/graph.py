from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Tuple

import numpy as np

from rasterio.crs import CRS
from rasterio.warp import transform


@dataclass
class SpeciesGraph:
    """Container for one species graph plus pairwise training targets.

    Parameters
    ----------
    name : str
        Species name.
    edge_index : np.ndarray
        `E x 2` edge list over graph nodes.
    edge_features : np.ndarray
        `E x F` edge feature matrix.
    node_coords : np.ndarray
        `N x 2` graph node coordinates in `lat, lon`.
    sample_coords : np.ndarray
        `S x 2` observed sample coordinates in `lat, lon`.
    pair_i : np.ndarray
        Pair row node indices for target distances.
    pair_j : np.ndarray
        Pair column node indices for target distances.
    pair_dist : np.ndarray
        Pairwise target distances aligned with `(pair_i, pair_j)`.
    num_nodes : int
        Number of graph nodes.
    edge_nbr_i : np.ndarray | None, optional
        Edge indices for the first member of each neighboring-edge pair.
    edge_nbr_j : np.ndarray | None, optional
        Edge indices for the second member of each neighboring-edge pair.
    val_pair_i : np.ndarray | None, optional
        Optional validation pair row indices.
    val_pair_j : np.ndarray | None, optional
        Optional validation pair column indices.
    val_pair_dist : np.ndarray | None, optional
        Optional validation target distances.
    """

    name: str
    edge_index: np.ndarray
    edge_features: np.ndarray
    node_coords: np.ndarray
    sample_coords: np.ndarray
    pair_i: np.ndarray
    pair_j: np.ndarray
    pair_dist: np.ndarray
    num_nodes: int
    edge_nbr_i: np.ndarray | None = None
    edge_nbr_j: np.ndarray | None = None
    val_pair_i: np.ndarray | None = None
    val_pair_j: np.ndarray | None = None
    val_pair_dist: np.ndarray | None = None

    def plot(
        self,
        edge_feature_idx: int | None = None,
        ax=None,
        basemap: bool | object = True,
        basemap_crs: str = "EPSG:3857",
        coord_order: str = "latlon",
        coords_crs: str = "EPSG:4326",
        sample_size: float = 12.0,
        edge_width: float = 2.0,
        edge_cmap: str = "viridis",
        sample_color: str = "black",
        sample_alpha: float = 0.8,
        edge_alpha: float = 0.9,
        edge_color: str = "#1f77b4",
        add_colorbar: bool = True,
        title: str | None = None,
    ):
        """Plot graph edges with optional edge-feature coloring and sample overlay.

        Parameters
        ----------
        edge_feature_idx : int | None, optional
            Column index in `edge_features` used for edge coloring. When `None`,
            edges are drawn with a constant color.
        ax : matplotlib.axes.Axes | None, optional
            Existing axis to draw on. A new one is created when omitted.
        basemap : bool | object, optional
            `True` uses CartoDB Positron, `False` disables basemap, or provide a
            contextily tile provider object.
        basemap_crs : str, optional
            CRS used when rendering with basemap tiles.
        coord_order : str, optional
            Coordinate order for plotting (`"latlon"` or `"lonlat"`).
        coords_crs : str, optional
            CRS of stored coordinates.
        sample_size : float, optional
            Marker size for sample points.
        edge_width : float, optional
            Edge line width.
        edge_cmap : str, optional
            Colormap used when `edge_feature_idx` is provided.
        sample_color : str, optional
            Marker color for sample points.
        sample_alpha : float, optional
            Marker alpha for sample points.
        edge_alpha : float, optional
            Edge alpha value.
        edge_color : str, optional
            Constant edge color used when `edge_feature_idx=None`.
        add_colorbar : bool, optional
            Whether to add a colorbar when feature coloring is enabled.
        title : str | None, optional
            Optional plot title; defaults to species name.

        Returns
        -------
        tuple
            `(ax, gdf_edges)` where `gdf_edges` is a GeoDataFrame of edge lines.
        """
        import geopandas as gpd
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        from shapely.geometry import LineString

        node_coords = np.asarray(self.node_coords, dtype=np.float64)
        sample_coords = np.asarray(self.sample_coords, dtype=np.float64)
        edge_index = np.asarray(self.edge_index, dtype=np.int64)
        edge_features = np.asarray(self.edge_features, dtype=np.float64)

        if node_coords.ndim != 2 or node_coords.shape[1] != 2:
            raise ValueError("node_coords must have shape (N, 2).")
        if sample_coords.ndim != 2 or sample_coords.shape[1] != 2:
            raise ValueError("sample_coords must have shape (S, 2).")
        if edge_index.ndim != 2 or edge_index.shape[1] != 2:
            raise ValueError("edge_index must have shape (E, 2).")
        if edge_features.ndim != 2:
            raise ValueError("edge_features must have shape (E, F).")
        if edge_features.shape[0] != edge_index.shape[0]:
            raise ValueError("edge_features row count must equal edge_index row count.")
        if coord_order not in {"latlon", "lonlat"}:
            raise ValueError("coord_order must be 'latlon' or 'lonlat'.")
        if node_coords.shape[0] == 0:
            raise ValueError("node_coords is empty.")

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 5))

        if basemap is not None and basemap is not False:
            node_xy = project_coords(
                node_coords,
                coord_order=coord_order,
                coords_crs=coords_crs,
                target_crs=basemap_crs,
            )
            sample_xy = project_coords(
                sample_coords,
                coord_order=coord_order,
                coords_crs=coords_crs,
                target_crs=basemap_crs,
            )
            x = node_xy[:, 0]
            y = node_xy[:, 1]
            xs = sample_xy[:, 0]
            ys = sample_xy[:, 1]
            plot_crs = basemap_crs
            xlabel, ylabel = "X", "Y"
        else:
            if coord_order == "latlon":
                x = node_coords[:, 1]
                y = node_coords[:, 0]
                xs = sample_coords[:, 1]
                ys = sample_coords[:, 0]
            else:
                x = node_coords[:, 0]
                y = node_coords[:, 1]
                xs = sample_coords[:, 0]
                ys = sample_coords[:, 1]
            plot_crs = coords_crs
            xlabel, ylabel = "Longitude", "Latitude"

        segments = [[(x[i], y[i]), (x[j], y[j])] for i, j in edge_index]
        line_collection: LineCollection
        edge_values = np.full(edge_index.shape[0], np.nan, dtype=np.float64)
        edge_feature_col = (
            np.full(edge_index.shape[0], -1, dtype=np.int64)
            if edge_feature_idx is None
            else np.full(edge_index.shape[0], int(edge_feature_idx), dtype=np.int64)
        )

        if edge_feature_idx is None:
            line_collection = LineCollection(
                segments,
                colors=edge_color,
                linewidths=edge_width,
                alpha=edge_alpha,
            )
        else:
            if edge_features.shape[1] == 0:
                raise ValueError("edge_features has zero columns; cannot color by feature index.")
            if edge_feature_idx < 0 or edge_feature_idx >= edge_features.shape[1]:
                raise IndexError(
                    f"edge_feature_idx={edge_feature_idx} out of range for "
                    f"edge_features with {edge_features.shape[1]} columns."
                )
            edge_values = edge_features[:, edge_feature_idx]
            line_collection = LineCollection(
                segments,
                cmap=edge_cmap,
                linewidths=edge_width,
                alpha=edge_alpha,
            )
            line_collection.set_array(edge_values)

        ax.add_collection(line_collection)
        if sample_coords.shape[0] > 0:
            ax.scatter(xs, ys, s=sample_size, c=sample_color, alpha=sample_alpha, zorder=3)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if title is None:
            title = self.name
        if title:
            ax.set_title(title)

        if sample_coords.shape[0] > 0:
            x_min = min(float(np.min(x)), float(np.min(xs)))
            x_max = max(float(np.max(x)), float(np.max(xs)))
            y_min = min(float(np.min(y)), float(np.min(ys)))
            y_max = max(float(np.max(y)), float(np.max(ys)))
        else:
            x_min = float(np.min(x))
            x_max = float(np.max(x))
            y_min = float(np.min(y))
            y_max = float(np.max(y))
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        if edge_feature_idx is not None and add_colorbar:
            plt.colorbar(
                line_collection,
                ax=ax,
                label=f"edge_features[:, {edge_feature_idx}]",
            )

        if basemap is not None and basemap is not False:
            try:
                import contextily as ctx
            except Exception as exc:
                raise ImportError(
                    "contextily is required when basemap is enabled. "
                    "Install with `conda install -c conda-forge contextily` "
                    "or disable basemap with basemap=False."
                ) from exc
            basemap_source = ctx.providers.CartoDB.Positron if basemap is True else basemap
            ctx.add_basemap(ax, source=basemap_source, crs=basemap_crs, reset_extent=False)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

        gdf_edges = gpd.GeoDataFrame(
            {
                "u": edge_index[:, 0],
                "v": edge_index[:, 1],
                "edge_feature_idx": edge_feature_col,
                "edge_value": edge_values,
            },
            geometry=[LineString(seg) for seg in segments],
            crs=plot_crs,
        )
        return ax, gdf_edges


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


def _coords_to_latlon(
    coords: np.ndarray,
    coord_order: str = "latlon",
    coords_crs: str | CRS | None = "EPSG:4326",
) -> np.ndarray:
    """Convert input coordinates into `lat, lon` order in EPSG:4326."""
    if coords_crs is None:
        raise ValueError("coords_crs is required for coordinate conversion")

    lons, lats = _as_lon_lat(coords, coord_order)
    src_crs = CRS.from_user_input(coords_crs)
    dst_crs = CRS.from_user_input("EPSG:4326")
    lon_out, lat_out = transform(src_crs, dst_crs, lons.tolist(), lats.tolist())
    return np.column_stack([lat_out, lon_out]).astype(np.float64)


def _cartesian_to_latlon(vertices: np.ndarray) -> np.ndarray:
    """Convert 3D Cartesian points on a sphere into `lat, lon` coordinates."""
    xyz = np.asarray(vertices, dtype=np.float64)
    radius = np.linalg.norm(xyz, axis=1)
    if np.any(radius <= 0.0):
        raise ValueError("vertices must lie away from the origin")

    x = xyz[:, 0] / radius
    y = xyz[:, 1] / radius
    z = np.clip(xyz[:, 2] / radius, -1.0, 1.0)
    lat = np.degrees(np.arcsin(z))
    lon = np.degrees(np.arctan2(y, x))
    return np.column_stack([lat, lon]).astype(np.float64)


def _edge_index_from_faces(faces: np.ndarray) -> np.ndarray:
    """Build a unique undirected edge list from triangular face indices."""
    face_array = np.asarray(faces, dtype=np.int64)
    if face_array.ndim != 2 or face_array.shape[1] != 3:
        raise ValueError("faces must have shape (F, 3)")

    edges = np.vstack(
        [
            face_array[:, [0, 1]],
            face_array[:, [1, 2]],
            face_array[:, [0, 2]],
        ]
    )
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    keep = edges[:, 0] != edges[:, 1]
    return edges[keep].astype(np.int64, copy=False)


def _median_edge_length_km(node_coords: np.ndarray, edge_index: np.ndarray) -> float:
    """Compute the median great-circle edge length for a graph."""
    if edge_index.size == 0:
        raise ValueError("edge_index is empty")
    lengths = haversine_km(node_coords[edge_index[:, 0]], node_coords[edge_index[:, 1]])
    positive = lengths[lengths > 0.0]
    if positive.size == 0:
        raise ValueError("edge_index does not contain positive-length edges")
    return float(np.median(positive))


@lru_cache(maxsize=None)
def _icosphere_geometry(subdivisions: int) -> tuple[np.ndarray, np.ndarray]:
    """Construct one cached icosphere and return vertices plus triangular faces."""
    if subdivisions < 0:
        raise ValueError("subdivisions must be >= 0")

    try:
        import trimesh
    except ImportError as exc:
        raise ImportError(
            "trimesh is required for build_geodesic_mesh_graph(...). "
            "Install it in the project environment before using this function."
        ) from exc

    mesh = trimesh.creation.icosphere(subdivisions=int(subdivisions), radius=1.0)
    return (
        np.asarray(mesh.vertices, dtype=np.float64),
        np.asarray(mesh.faces, dtype=np.int64),
    )


@lru_cache(maxsize=None)
def _geodesic_mesh_for_subdivision(subdivisions: int) -> tuple[np.ndarray, np.ndarray]:
    """Build one cached geodesic mesh at a fixed subdivision level."""
    vertices, faces = _icosphere_geometry(int(subdivisions))
    node_coords = _cartesian_to_latlon(vertices)
    edge_index = _edge_index_from_faces(faces)
    return node_coords, edge_index


def _choose_icosphere_subdivision_for_spacing(
    spacing_km: float,
    max_subdivisions: int = 7,
) -> tuple[int, float]:
    """Map a target edge spacing in kilometers to the closest icosphere level."""
    if spacing_km <= 0.0:
        raise ValueError("spacing_km must be > 0.")

    candidates: list[tuple[float, int, float]] = []
    for subdivisions in range(max_subdivisions + 1):
        node_coords, edge_index = _geodesic_mesh_for_subdivision(subdivisions)
        median_spacing = _median_edge_length_km(node_coords, edge_index)
        candidates.append((abs(median_spacing - spacing_km), subdivisions, median_spacing))

    _, best_subdivisions, best_spacing = min(candidates)
    return int(best_subdivisions), float(best_spacing)


def _spacing_km_from_deg(spacing_deg: float, mean_lat: float) -> float:
    """Approximate a kilometer mesh spacing from a degree spacing."""
    if spacing_deg <= 0.0:
        raise ValueError("spacing_deg must be > 0.")

    lat_km = spacing_deg * 111.0
    lon_km = spacing_deg * 111.0 * max(np.cos(np.radians(mean_lat)), 1e-6)
    return float(0.5 * (lat_km + lon_km))


def _resolve_bbox_spec(
    bbox: str | None,
    bbox_file: str | None,
) -> tuple[str, str | None]:
    """Normalize bbox arguments into one of the supported clipping modes."""
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
    return bbox, bbox_file


def _study_region_geometry(
    all_coords_latlon: np.ndarray,
    buffer_km: float,
    bbox: str | None,
    bbox_file: str | None,
    project_to: str | CRS | None,
) -> tuple[object, CRS]:
    """Construct the projected clipping geometry for a shared study region."""
    import geopandas as gpd
    from shapely.geometry import Point, Polygon, box

    bbox, bbox_file = _resolve_bbox_spec(bbox, bbox_file)

    if project_to is None:
        region_crs = CRS.from_user_input("EPSG:3857")
    else:
        maybe_crs = CRS.from_user_input(project_to)
        region_crs = CRS.from_user_input("EPSG:3857") if maybe_crs.is_geographic else maybe_crs

    coords_xy = project_coords(
        all_coords_latlon,
        coord_order="latlon",
        coords_crs="EPSG:4326",
        target_crs=region_crs,
    )

    if bbox == "square":
        x_min = float(np.min(coords_xy[:, 0]))
        x_max = float(np.max(coords_xy[:, 0]))
        y_min = float(np.min(coords_xy[:, 1]))
        y_max = float(np.max(coords_xy[:, 1]))
        region = box(x_min, y_min, x_max, y_max)
    elif bbox == "convex_hull":
        gseries = gpd.GeoSeries(
            [Point(lon, lat) for lat, lon in all_coords_latlon],
            crs="EPSG:4326",
        ).to_crs(region_crs)
        region = gseries.unary_union.convex_hull
    else:
        if bbox_file is None:
            raise ValueError("bbox_file is required when bbox='polygon'")
        poly_coords = np.loadtxt(bbox_file)
        if poly_coords.ndim != 2 or poly_coords.shape[1] != 2:
            raise ValueError("bbox_file must contain two columns (lat lon)")
        if not np.allclose(poly_coords[0], poly_coords[-1]):
            raise ValueError("bbox_file polygon must be closed (first and last point identical)")
        polygon = Polygon([(lon, lat) for lat, lon in poly_coords])
        region = gpd.GeoSeries([polygon], crs="EPSG:4326").to_crs(region_crs).unary_union

    if buffer_km < 0:
        raise ValueError("buffer_km must be >= 0")
    if buffer_km > 0:
        region = region.buffer(buffer_km * 1000.0)
    return region, region_crs


def _clip_graph_to_region(
    mesh_coords: np.ndarray,
    edge_index: np.ndarray,
    region: object,
    region_crs: CRS,
) -> tuple[np.ndarray, np.ndarray]:
    """Clip graph nodes to a region and retain only surviving inherited edges."""
    import geopandas as gpd
    from shapely.geometry import Point

    mesh_series = gpd.GeoSeries(
        [Point(lon, lat) for lat, lon in mesh_coords],
        crs="EPSG:4326",
    ).to_crs(region_crs)
    mask = np.asarray(mesh_series.within(region) | mesh_series.touches(region), dtype=bool)
    keep_nodes = np.flatnonzero(mask)
    if keep_nodes.size == 0:
        return np.empty((0, 2), dtype=np.float64), np.empty((0, 2), dtype=np.int64)

    old_to_new = np.full(mesh_coords.shape[0], -1, dtype=np.int64)
    old_to_new[keep_nodes] = np.arange(keep_nodes.size, dtype=np.int64)

    clipped_coords = mesh_coords[keep_nodes]
    edge_mask = mask[edge_index[:, 0]] & mask[edge_index[:, 1]]
    clipped_edges = edge_index[edge_mask]
    if clipped_edges.size == 0:
        return clipped_coords, np.empty((0, 2), dtype=np.int64)

    clipped_edges = old_to_new[clipped_edges]
    clipped_edges = np.sort(clipped_edges, axis=1)
    clipped_edges = np.unique(clipped_edges, axis=0)
    keep = clipped_edges[:, 0] != clipped_edges[:, 1]
    return clipped_coords.astype(np.float64, copy=False), clipped_edges[keep].astype(np.int64)


def _largest_connected_component(
    mesh_coords: np.ndarray,
    edge_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep the largest connected component of a graph and reindex its nodes."""
    num_nodes = mesh_coords.shape[0]
    if num_nodes == 0:
        return mesh_coords, edge_index
    if edge_index.size == 0:
        raise ValueError("Graph clipping produced no edges.")

    adjacency: list[list[int]] = [[] for _ in range(num_nodes)]
    for u, v in np.asarray(edge_index, dtype=np.int64):
        adjacency[int(u)].append(int(v))
        adjacency[int(v)].append(int(u))

    visited = np.zeros(num_nodes, dtype=bool)
    best_component: list[int] = []
    for start in range(num_nodes):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        component: list[int] = []
        while stack:
            node = stack.pop()
            component.append(node)
            for nbr in adjacency[node]:
                if not visited[nbr]:
                    visited[nbr] = True
                    stack.append(nbr)
        if len(component) > len(best_component):
            best_component = component

    keep_nodes = np.asarray(sorted(best_component), dtype=np.int64)
    if keep_nodes.size < 2:
        raise ValueError("Largest connected component has fewer than two nodes.")

    keep_mask = np.zeros(num_nodes, dtype=bool)
    keep_mask[keep_nodes] = True
    old_to_new = np.full(num_nodes, -1, dtype=np.int64)
    old_to_new[keep_nodes] = np.arange(keep_nodes.size, dtype=np.int64)

    component_coords = mesh_coords[keep_nodes]
    edge_mask = keep_mask[edge_index[:, 0]] & keep_mask[edge_index[:, 1]]
    component_edges = old_to_new[edge_index[edge_mask]]
    if component_edges.size == 0:
        raise ValueError("Largest connected component has no edges.")

    component_edges = np.sort(component_edges, axis=1)
    component_edges = np.unique(component_edges, axis=0)
    return component_coords.astype(np.float64, copy=False), component_edges.astype(np.int64, copy=False)


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


def build_edge_neighbor_pairs(
    edge_index: np.ndarray,
    num_nodes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Enumerate pairs of edges that meet at a shared graph node.

    Parameters
    ----------
    edge_index : np.ndarray
        `E x 2` edge list over graph nodes.
    num_nodes : int
        Number of graph nodes referenced by `edge_index`.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Parallel integer arrays of neighboring-edge indices.
    """
    incident_edges: list[list[int]] = [[] for _ in range(num_nodes)]
    for edge_id, (u, v) in enumerate(np.asarray(edge_index, dtype=np.int64)):
        incident_edges[int(u)].append(edge_id)
        incident_edges[int(v)].append(edge_id)

    pairs: set[tuple[int, int]] = set()
    for edges_at_node in incident_edges:
        for i, edge_i in enumerate(edges_at_node):
            for edge_j in edges_at_node[i + 1 :]:
                a, b = (edge_i, edge_j) if edge_i < edge_j else (edge_j, edge_i)
                pairs.add((a, b))

    if not pairs:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty

    pair_array = np.asarray(sorted(pairs), dtype=np.int64)
    return pair_array[:, 0], pair_array[:, 1]


def _filter_long_mesh_edges(
    mesh_coords: np.ndarray,
    edge_index: np.ndarray,
    max_ratio: float = 1.25,
) -> np.ndarray:
    """Drop Delaunay edges that are much longer than the nominal mesh step.

    The nominal step is estimated from the shortest positive edge length in the
    candidate graph, which corresponds to the local mesh spacing for the regular
    grids used here.
    """
    if edge_index.size == 0:
        return edge_index

    edge_lengths = haversine_km(
        mesh_coords[edge_index[:, 0]],
        mesh_coords[edge_index[:, 1]],
    )
    positive = edge_lengths[edge_lengths > 0.0]
    if positive.size == 0:
        return edge_index

    nominal_step = float(np.min(positive))
    keep = edge_lengths <= (nominal_step * max_ratio)
    return edge_index[keep]


def build_geodesic_mesh_graph(
    coords_list: list[np.ndarray],
    spacing_km: float | None = 50.0,
    spacing_deg: float | None = None,
    grid_type: str = "triangular",
    project_to: str | CRS | None = None,
    coord_order: str = "latlon",
    coords_crs: str | CRS | None = "EPSG:4326",
    buffer_km: float = 0.0,
    bbox: str | None = "square",
    bbox_file: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a shared geodesic triangular mesh clipped to the study region.

    Parameters
    ----------
    coords_list : list[np.ndarray]
        Per-species coordinate arrays (`S_i x 2`).
    spacing_km : float | None, optional
        Target edge spacing in kilometers.
    spacing_deg : float | None, optional
        Approximate spacing in degrees. Used only when `spacing_km` is omitted.
    grid_type : str, optional
        Accepted for compatibility; only `"triangular"` is supported.
    project_to : str | CRS | None, optional
        Optional projected CRS used for region clipping.
    coord_order : str, optional
        Coordinate order of arrays in `coords_list`.
    coords_crs : str | CRS | None, optional
        CRS of coordinates in `coords_list`.
    buffer_km : float, optional
        Buffer applied to the clipping geometry.
    bbox : str | None, optional
        Bounding shape: `"square"`, `"convex_hull"`, `"polygon"`, or `None`.
    bbox_file : str | None, optional
        Path to polygon coordinates (`lat lon`) when using polygon clipping.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        `mesh_coords` (`M x 2`) and `edge_index` (`E x 2`) in native mesh topology.
    """
    if not coords_list:
        raise ValueError("coords_list is empty")
    if spacing_km is not None and spacing_deg is not None:
        raise ValueError("Provide only one of spacing_km or spacing_deg.")

    grid_type = grid_type.lower()
    if grid_type != "triangular":
        raise ValueError("build_geodesic_mesh_graph only supports grid_type='triangular'.")

    all_coords = np.vstack(coords_list)
    all_coords_latlon = _coords_to_latlon(
        all_coords,
        coord_order=coord_order,
        coords_crs=coords_crs,
    )
    mean_lat = float(np.mean(all_coords_latlon[:, 0]))

    if spacing_km is None:
        if spacing_deg is None:
            raise ValueError("Provide spacing_km or spacing_deg.")
        spacing_km = _spacing_km_from_deg(spacing_deg, mean_lat=mean_lat)
    elif spacing_km <= 0.0:
        raise ValueError("spacing_km must be > 0.")

    subdivisions, _ = _choose_icosphere_subdivision_for_spacing(float(spacing_km))
    mesh_coords, edge_index = _geodesic_mesh_for_subdivision(subdivisions)
    region, region_crs = _study_region_geometry(
        all_coords_latlon,
        buffer_km=buffer_km,
        bbox=bbox,
        bbox_file=bbox_file,
        project_to=project_to,
    )
    mesh_coords, edge_index = _clip_graph_to_region(mesh_coords, edge_index, region, region_crs)
    if mesh_coords.size == 0:
        raise ValueError(
            "Geodesic mesh generation produced zero nodes. Check bbox/buffer/grid spacing."
        )

    mesh_coords, edge_index = _largest_connected_component(mesh_coords, edge_index)
    return mesh_coords, edge_index


def build_dense_mesh_graph(
    coords_list: list[np.ndarray],
    spacing_km: float | None = 50.0,
    spacing_deg: float | None = None,
    grid_type: str = "triangular",
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

    edge_index = build_delaunay_graph(
        mesh_coords,
        project_to=project_to,
        coord_order=coord_order,
        coords_crs=coords_crs,
    )
    edge_index = _filter_long_mesh_edges(mesh_coords, edge_index)

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
