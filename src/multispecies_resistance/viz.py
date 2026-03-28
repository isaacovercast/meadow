from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib.colors as clr
import math

edge_cmap = "RdBu_r"

def _normalize_explore_kwargs(
    edge_values: np.ndarray,
    cmap,
    explore_kwargs: dict | None,
) -> dict:
    """Build safe default kwargs for GeoPandas `explore` edge rendering.

    Parameters
    ----------
    edge_values : np.ndarray
        Edge scalar values used for color scaling.
    cmap
        Matplotlib colormap or explore-compatible colormap object.
    explore_kwargs : dict | None
        User overrides for `GeoDataFrame.explore`.

    Returns
    -------
    dict
        Keyword arguments with robust `vmin`/`vmax` defaults.
    """
    values = np.asarray(edge_values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin = float(finite.min())
        vmax = float(finite.max())
        if vmax <= vmin:
            vmax = vmin + 1e-9

    cmap_for_explore = cmap
    if isinstance(cmap, clr.Colormap):
        cmap_for_explore = [clr.to_hex(cmap(i / 255.0)) for i in range(256)]

    kwargs = dict(
        column="edge_value",
        cmap=cmap_for_explore,
        tiles="CartoDB positron",
        vmin=vmin,
        vmax=vmax,
    )
    if explore_kwargs:
        kwargs.update(explore_kwargs)
    if kwargs.get("vmin") is None:
        kwargs["vmin"] = vmin
    if kwargs.get("vmax") is None:
        kwargs["vmax"] = vmax
    return kwargs


def _explore_safe(gdf, edge_values: np.ndarray, cmap, explore_kwargs: dict | None):
    """Render an interactive map with compatibility fallback for colormap issues.

    Parameters
    ----------
    gdf
        GeoDataFrame containing edge geometries and values.
    edge_values : np.ndarray
        Edge scalar values for color scaling.
    cmap
        Preferred colormap.
    explore_kwargs : dict | None
        User overrides for `GeoDataFrame.explore`.

    Returns
    -------
    Any
        Folium map object returned by `gdf.explore`.
    """
    kwargs = _normalize_explore_kwargs(edge_values, cmap, explore_kwargs)
    try:
        return gdf.explore(**kwargs)
    except TypeError:
        # Fallback for geopandas/folium versions that choke on colormap objects/lists.
        kwargs["cmap"] = "RdBu_r"
        if kwargs.get("vmin") is None or kwargs.get("vmax") is None:
            finite = np.asarray(edge_values, dtype=float)
            finite = finite[np.isfinite(finite)]
            if finite.size == 0:
                kwargs["vmin"], kwargs["vmax"] = 0.0, 1.0
            else:
                kwargs["vmin"] = float(finite.min())
                vmax = float(finite.max())
                kwargs["vmax"] = vmax if vmax > kwargs["vmin"] else kwargs["vmin"] + 1e-9
        return gdf.explore(**kwargs)


def plot_sites(
    site_coords: np.ndarray,
    ax=None,
    title: Optional[str] = None,
    alpha: float = 0.75,
):
    """Scatter-plot site coordinates in longitude/latitude space.

    Parameters
    ----------
    site_coords : np.ndarray
        `N x 2` coordinates in `lat, lon`.
    ax : matplotlib.axes.Axes | None, optional
        Existing axis; a new figure/axis is created when omitted.
    title : Optional[str], optional
        Plot title text.
    alpha : float, optional
        Marker transparency.

    Returns
    -------
    matplotlib.axes.Axes
        Axis containing the plotted points.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(site_coords[:, 1], site_coords[:, 0], s=15, c="black", alpha=alpha)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    if title:
        ax.set_title(title)
    return ax


def plot_species_resistance(
    site_coords: np.ndarray,
    edge_index: np.ndarray,
    edge_values: np.ndarray | None = None,
    ax=None,
    cmap=edge_cmap,
    basemap=True,
    basemap_crs: str = "EPSG:3857",
    coord_order: str = "latlon",
    coords_crs: str = "EPSG:4326",
    explore: bool = False,
    explore_kwargs: dict | None = None,
    model=None,
    edge_features: np.ndarray | None = None,
    species_idx: int = 0,
    show_sites: bool = False,
    sample_coords: np.ndarray | None = None,
    value_label: str = "Edge resistance",
):
    """Plot one species graph with edges colored by resistance values.

    Parameters
    ----------
    site_coords : np.ndarray
        `S x 2` graph node coordinates.
    edge_index : np.ndarray
        `E x 2` edge list.
    edge_values : np.ndarray | None, optional
        Edge values to color. If omitted, values are computed from `model`.
    ax : matplotlib.axes.Axes | None, optional
        Existing axis; a new one is created when omitted.
    cmap : Any, optional
        Colormap used for edge coloring.
    basemap : Any, optional
        `True`/provider object for background basemap, or `False`/`None` to disable.
    basemap_crs : str, optional
        Projection CRS used with basemap rendering.
    coord_order : str, optional
        Coordinate order for `site_coords`.
    coords_crs : str, optional
        CRS of input coordinates.
    explore : bool, optional
        If `True`, also return an interactive folium map.
    explore_kwargs : dict | None, optional
        Extra kwargs passed to `GeoDataFrame.explore`.
    model : Any, optional
        Model used to compute edge values when `edge_values` is not provided.
    edge_features : np.ndarray | None, optional
        Edge feature matrix used with `model`.
    species_idx : int, optional
        Species index used with `model`.
    show_sites : bool, optional
        Whether to overlay sample/site points.
    sample_coords : np.ndarray | None, optional
        Coordinates to plot when `show_sites=True`; defaults to `site_coords`.
    value_label : str, optional
        Colorbar label for the plotted edge values.

    Returns
    -------
    tuple
        `(ax, gdf)` or `(ax, gdf, folium_map)` when `explore=True`.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    import geopandas as gpd
    from shapely.geometry import LineString

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    if coord_order not in {"latlon", "lonlat"}:
        raise ValueError("coord_order must be 'latlon' or 'lonlat'")

    if basemap is not None and basemap is not False:
        from multispecies_resistance.graph import project_coords

        coords = project_coords(
            site_coords,
            coord_order=coord_order,
            coords_crs=coords_crs,
            target_crs=basemap_crs,
        )
        x = coords[:, 0]
        y = coords[:, 1]
    else:
        if coord_order == "latlon":
            y = site_coords[:, 0]
            x = site_coords[:, 1]
        else:
            x = site_coords[:, 0]
            y = site_coords[:, 1]

    if edge_values is None:
        if model is None or edge_features is None:
            raise ValueError("Provide edge_values or (model and edge_features).")
        import torch

        edge_feat = torch.from_numpy(edge_features)
        edge_values, _, _ = model.edge_resistance(species_idx, edge_feat)
        edge_values = edge_values.detach().numpy()

    segments = []
    for i, j in edge_index:
        segments.append([(x[i], y[i]), (x[j], y[j])])

    lc = LineCollection(segments, cmap=cmap)
    lc.set_array(edge_values)
    lc.set_linewidth(2.0)
    ax.add_collection(lc)
    ax.autoscale()
    ax.set_xlabel("Longitude" if basemap is None or basemap is False else "X")
    ax.set_ylabel("Latitude" if basemap is None or basemap is False else "Y")
    plt.colorbar(lc, ax=ax, label=value_label)

    xlim = None
    ylim = None
    if show_sites:
        coords_plot = site_coords if sample_coords is None else sample_coords
        if basemap is not None and basemap is not False:
            from multispecies_resistance.graph import project_coords

            coords_s = project_coords(
                coords_plot,
                coord_order=coord_order,
                coords_crs=coords_crs,
                target_crs=basemap_crs,
            )
            x_s = coords_s[:, 0]
            y_s = coords_s[:, 1]
        else:
            if coord_order == "latlon":
                y_s = coords_plot[:, 0]
                x_s = coords_plot[:, 1]
            else:
                x_s = coords_plot[:, 0]
                y_s = coords_plot[:, 1]

        xmin = min(float(x.min()), float(x_s.min()))
        xmax = max(float(x.max()), float(x_s.max()))
        ymin = min(float(y.min()), float(y_s.min()))
        ymax = max(float(y.max()), float(y_s.max()))
        xlim = (xmin, xmax)
        ylim = (ymin, ymax)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.scatter(x_s, y_s, s=10, c="black", alpha=0.75, zorder=3)

    if basemap is not None and basemap is not False:
        import contextily as ctx

        if basemap is True:
            basemap_source = ctx.providers.CartoDB.Positron
        else:
            basemap_source = basemap
        ctx.add_basemap(ax, source=basemap_source, crs=basemap_crs, reset_extent=False)
        if xlim is not None and ylim is not None:
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)

    crs = basemap_crs if basemap is not None and basemap is not False else coords_crs
    gdf = gpd.GeoDataFrame(
        {"edge_value": np.asarray(edge_values)},
        geometry=[LineString(seg) for seg in segments],
        crs=crs,
    )

    folium_map = None
    if explore:
        folium_map = _explore_safe(gdf, edge_values, cmap, explore_kwargs)
        return ax, gdf, folium_map

    return ax, gdf


def plot_multi_edge_resistance(
    species_list,
    graphs,
    model,
    cmap=edge_cmap,
    basemap=True,
    basemap_crs: str = "EPSG:3857",
    coord_order: str = "latlon",
    coords_crs: str = "EPSG:4326",
    explore: bool = False,
    explore_kwargs: dict | None = None,
    overlay: bool = False,
    overlay_stat: str = "mean",
    combine_with_shared: bool = True,
    show_sites: bool = False,
    sample_coords_list: list[np.ndarray] | None = None,
    ncols: int = 2,
    figsize: tuple[int, int] = (6, 5),
):
    """Plot resistance edges for multiple species as facets or an overlay summary.

    Parameters
    ----------
    species_list : list
        Species records used for titles and optional site overlays.
    graphs : list
        Graph objects with `edge_index`, `edge_features`, and `node_coords`.
    model : Any
        Trained model used to predict edge resistance.
    cmap : Any, optional
        Colormap used for edge coloring.
    basemap : Any, optional
        `True`/provider object for background basemap, or `False`/`None` to disable.
    basemap_crs : str, optional
        Projection CRS used with basemap rendering.
    coord_order : str, optional
        Coordinate order for graph coordinates.
    coords_crs : str, optional
        CRS of graph coordinates.
    explore : bool, optional
        If `True`, also return interactive folium maps.
    explore_kwargs : dict | None, optional
        Extra kwargs passed to `GeoDataFrame.explore`.
    overlay : bool, optional
        If `True`, aggregate all species into one edge map.
    overlay_stat : str, optional
        Aggregation statistic for overlay mode (`"mean"` or `"std"`).
    combine_with_shared : bool, optional
        If `True`, plot the full species resistance obtained by combining the
        shared and species-specific components. If `False`, plot only the
        species-specific logit component.
    show_sites : bool, optional
        Whether to overlay sample/site points.
    sample_coords_list : list[np.ndarray] | None, optional
        Per-species coordinate overrides for plotted points.
    ncols : int, optional
        Number of subplot columns in faceted mode.
    figsize : tuple[int, int], optional
        Base figure size.

    Returns
    -------
    tuple
        Facet mode returns `(axes, gdfs)` (plus maps when `explore=True`);
        overlay mode returns `(ax, gdf)` (plus map when `explore=True`).
    """
    import matplotlib.pyplot as plt
    import torch
    import geopandas as gpd
    from shapely.geometry import LineString
    from matplotlib.collections import LineCollection

    if len(species_list) != len(graphs):
        raise ValueError("species_list and graphs must have the same length")
    if sample_coords_list is not None and len(sample_coords_list) != len(species_list):
        raise ValueError("sample_coords_list must match species_list length")

    def _coords_for_plot(_site_coords: np.ndarray):
        if coord_order not in {"latlon", "lonlat"}:
            raise ValueError("coord_order must be 'latlon' or 'lonlat'")

        if basemap is not None and basemap is not False:
            from multispecies_resistance.graph import project_coords

            coords = project_coords(
                _site_coords,
                coord_order=coord_order,
                coords_crs=coords_crs,
                target_crs=basemap_crs,
            )
            return coords[:, 0], coords[:, 1], basemap_crs

        if coord_order == "latlon":
            return _site_coords[:, 1], _site_coords[:, 0], coords_crs
        return _site_coords[:, 0], _site_coords[:, 1], coords_crs

    def _edge_values_for_species(species_idx: int, graph) -> np.ndarray:
        edge_feat = torch.from_numpy(graph.edge_features)
        if combine_with_shared:
            edge_values, _, _ = model.edge_resistance(species_idx, edge_feat)
        else:
            _, edge_values = model.edge_logits(species_idx, edge_feat)
        return edge_values.detach().numpy()

    colorbar_label = "Edge resistance" if combine_with_shared else "Species-specific edge logit"

    if overlay:
        fig, ax = plt.subplots(figsize=figsize)
        values_by_species = []
        xs_all = []
        ys_all = []

        if not graphs:
            raise ValueError("graphs is empty")

        base_graph = graphs[0]
        for g in graphs[1:]:
            if not np.array_equal(g.edge_index, base_graph.edge_index):
                raise ValueError("overlay=True requires identical edge_index across species.")
            if g.node_coords.shape != base_graph.node_coords.shape or not np.allclose(
                g.node_coords, base_graph.node_coords
            ):
                raise ValueError("overlay=True requires identical node_coords across species.")

        for idx, (sp, g) in enumerate(zip(species_list, graphs)):
            values_by_species.append(_edge_values_for_species(idx, g))
            x, y, crs = _coords_for_plot(g.node_coords)
            xs_all.append(x)
            ys_all.append(y)

        values_by_species = np.vstack(values_by_species)
        stat = overlay_stat.lower()
        if stat == "mean":
            values_all = values_by_species.mean(axis=0)
        elif stat in {"std", "sd"}:
            values_all = values_by_species.std(axis=0)
        else:
            raise ValueError("overlay_stat must be 'mean' or 'std'")

        x_base, y_base, crs = _coords_for_plot(base_graph.node_coords)
        segments_all = []
        for i, j in base_graph.edge_index:
            segments_all.append([(x_base[i], y_base[i]), (x_base[j], y_base[j])])

        lc = LineCollection(segments_all, cmap=cmap)
        lc.set_array(values_all)
        lc.set_linewidth(2.0)
        ax.add_collection(lc)
        ax.autoscale()
        ax.set_xlabel("Longitude" if basemap is None or basemap is False else "X")
        ax.set_ylabel("Latitude" if basemap is None or basemap is False else "Y")
        plt.colorbar(lc, ax=ax, label=colorbar_label)

        if show_sites:
            for sp_idx, sp in enumerate(species_list):
                coords_plot = (
                    sample_coords_list[sp_idx]
                    if sample_coords_list is not None
                    else sp.sample_coords
                )
                x_s, y_s, _ = _coords_for_plot(coords_plot)
                xs_all.append(x_s)
                ys_all.append(y_s)
            if xs_all and ys_all:
                x_concat = np.concatenate(xs_all)
                y_concat = np.concatenate(ys_all)
                xlim = (float(x_concat.min()), float(x_concat.max()))
                ylim = (float(y_concat.min()), float(y_concat.max()))
                ax.set_xlim(*xlim)
                ax.set_ylim(*ylim)

        if basemap is not None and basemap is not False:
            import contextily as ctx

            if basemap is True:
                basemap_source = ctx.providers.CartoDB.Positron
            else:
                basemap_source = basemap
            ctx.add_basemap(ax, source=basemap_source, crs=basemap_crs, reset_extent=False)
            if show_sites and xs_all and ys_all:
                ax.set_xlim(*xlim)
                ax.set_ylim(*ylim)

        if show_sites:
            for sp_idx, sp in enumerate(species_list):
                coords_plot = (
                    sample_coords_list[sp_idx]
                    if sample_coords_list is not None
                    else sp.sample_coords
                )
                x_s, y_s, _ = _coords_for_plot(coords_plot)
                ax.scatter(x_s, y_s, s=10, c="black", alpha=0.75, zorder=3)

        gdf = gpd.GeoDataFrame(
            {
                "edge_value": values_all,
                "overlay_stat": stat,
                "n_species": len(species_list),
            },
            geometry=[LineString(seg) for seg in segments_all],
            crs=crs,
        )

        folium_map = None
        if explore:
            folium_map = _explore_safe(gdf, values_all, cmap, explore_kwargs)
            return ax, gdf, folium_map
        return ax, gdf

    num = len(species_list)
    cols = max(1, min(ncols, num))
    rows = math.ceil(num / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(figsize[0] * cols, figsize[1] * rows))
    axes_arr = np.array(axes).reshape(-1)

    gdfs = []
    maps = []

    for idx, (sp, g) in enumerate(zip(species_list, graphs)):
        edge_values = _edge_values_for_species(idx, g)

        ax_i = axes_arr[idx]
        coords_plot = (
            sample_coords_list[idx] if sample_coords_list is not None else sp.sample_coords
        )
        ax_i, gdf_i, fmap_i = plot_species_resistance(
            g.node_coords,
            g.edge_index,
            edge_values,
            ax=ax_i,
            cmap=cmap,
            basemap=basemap,
            basemap_crs=basemap_crs,
            coord_order=coord_order,
            coords_crs=coords_crs,
            explore=explore,
            explore_kwargs=explore_kwargs,
            show_sites=show_sites,
            sample_coords=coords_plot if show_sites else None,
            value_label=colorbar_label,
        )
        ax_i.set_title(sp.name)
        gdfs.append(gdf_i)
        if explore:
            maps.append(fmap_i)

    for ax_extra in axes_arr[num:]:
        ax_extra.remove()

    if explore:
        return axes_arr[:num], gdfs, maps
    return axes_arr[:num], gdfs


def plot_shared_resistance(
    species_list,
    graphs,
    model,
    graph_index: int = 0,
    cmap=edge_cmap,
    basemap=True,
    basemap_crs: str = "EPSG:3857",
    coord_order: str = "latlon",
    coords_crs: str = "EPSG:4326",
    explore: bool = False,
    explore_kwargs: dict | None = None,
    rasterize: bool = False,
    grid_size: int = 200,
    interpolation: str = "midpoint",
    interp_method: str = "linear",
    fill_method: str = "nearest",
    rbf_function: str = "linear",
    rbf_kwargs: dict | None = None,
    kriging_kwargs: dict | None = None,
    show_sites: bool = False,
):
    """Plot the shared cross-species resistance component as edges or a surface.

    Parameters
    ----------
    species_list : list
        Species records aligned with `graphs`.
    graphs : list
        Graph objects with edge features and geometry.
    model : Any
        Trained model with shared edge network.
    graph_index : int, optional
        Which graph/species entry to visualize.
    cmap : Any, optional
        Colormap used for plotting.
    basemap : Any, optional
        `True`/provider object for basemap, or `False`/`None` to disable.
    basemap_crs : str, optional
        Projection CRS used with basemap rendering.
    coord_order : str, optional
        Coordinate order for graph coordinates.
    coords_crs : str, optional
        CRS of graph coordinates.
    explore : bool, optional
        If `True`, also return an interactive folium map.
    explore_kwargs : dict | None, optional
        Extra kwargs passed to `GeoDataFrame.explore`.
    rasterize : bool, optional
        If `True`, interpolate shared edge values to a raster surface.
    grid_size : int, optional
        Number of pixels along each raster axis when `rasterize=True`.
    interpolation : str, optional
        Interpolation family (`"midpoint"`, `"rbf"`, `"kriging"`).
    interp_method : str, optional
        SciPy griddata method for midpoint interpolation.
    fill_method : str, optional
        Fill strategy for midpoint interpolation (`"nearest"` or `"nan"`).
    rbf_function : str, optional
        RBF kernel function when `interpolation="rbf"`.
    rbf_kwargs : dict | None, optional
        Extra kwargs for `scipy.interpolate.Rbf`.
    kriging_kwargs : dict | None, optional
        Extra kwargs for `pykrige.ok.OrdinaryKriging`.
    show_sites : bool, optional
        Whether to overlay site points.

    Returns
    -------
    tuple
        Edge mode returns `plot_species_resistance` output;
        raster mode returns `(ax, surface)` (plus map when `explore=True`).
    """
    import torch

    if not graphs:
        raise ValueError("graphs is empty")
    if graph_index < 0 or graph_index >= len(graphs):
        raise IndexError("graph_index out of range")

    g = graphs[graph_index]
    sp = species_list[graph_index]

    edge_feat = torch.from_numpy(g.edge_features)
    shared_logits = model.shared(edge_feat).squeeze(-1)
    edge_resistance = (torch.nn.functional.softplus(shared_logits) + 1e-4).detach().numpy()

    if not rasterize:
        return plot_species_resistance(
            g.node_coords,
            g.edge_index,
            edge_resistance,
            cmap=cmap,
            basemap=basemap,
            basemap_crs=basemap_crs,
            coord_order=coord_order,
            coords_crs=coords_crs,
            explore=explore,
            explore_kwargs=explore_kwargs,
            show_sites=show_sites,
            sample_coords=sp.sample_coords if show_sites else None,
        )

    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata

    if coord_order not in {"latlon", "lonlat"}:
        raise ValueError("coord_order must be 'latlon' or 'lonlat'")

    if basemap is not None and basemap is not False:
        from multispecies_resistance.graph import project_coords

        coords = project_coords(
            g.node_coords,
            coord_order=coord_order,
            coords_crs=coords_crs,
            target_crs=basemap_crs,
        )
        crs = basemap_crs
        x = coords[:, 0]
        y = coords[:, 1]
    else:
        crs = coords_crs
        if coord_order == "latlon":
            y = g.node_coords[:, 0]
            x = g.node_coords[:, 1]
        else:
            x = g.node_coords[:, 0]
            y = g.node_coords[:, 1]

    mid_x = (x[g.edge_index[:, 0]] + x[g.edge_index[:, 1]]) / 2.0
    mid_y = (y[g.edge_index[:, 0]] + y[g.edge_index[:, 1]]) / 2.0
    points = np.column_stack([mid_x, mid_y])

    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    xi = np.linspace(xmin, xmax, grid_size)
    yi = np.linspace(ymin, ymax, grid_size)
    grid_x, grid_y = np.meshgrid(xi, yi)

    from shapely.geometry import MultiPoint
    from matplotlib.path import Path as MplPath

    hull = MultiPoint(np.column_stack([x, y])).convex_hull
    hull_path = None
    if hasattr(hull, "exterior"):
        hull_path = MplPath(np.asarray(hull.exterior.coords))

    interpolation = interpolation.lower()
    if interpolation == "midpoint":
        grid_z = griddata(points, edge_resistance, (grid_x, grid_y), method=interp_method)
        if fill_method == "nearest":
            grid_z_near = griddata(points, edge_resistance, (grid_x, grid_y), method="nearest")
            grid_z = np.where(np.isnan(grid_z), grid_z_near, grid_z)
        elif fill_method == "nan":
            pass
        else:
            raise ValueError("fill_method must be 'nearest' or 'nan'")
    elif interpolation == "rbf":
        from scipy.interpolate import Rbf

        kwargs = dict(function=rbf_function)
        if rbf_kwargs:
            kwargs.update(rbf_kwargs)
        rbf = Rbf(points[:, 0], points[:, 1], edge_resistance, **kwargs)
        grid_z = rbf(grid_x, grid_y)
    elif interpolation == "kriging":
        try:
            from pykrige.ok import OrdinaryKriging
        except Exception as exc:
            raise ImportError(
                "pykrige is required for kriging interpolation. "
                "Install with `conda install -c conda-forge pykrige` "
                "or `pip install pykrige`."
            ) from exc

        kwargs = dict(variogram_model="linear")
        if kriging_kwargs:
            kwargs.update(kriging_kwargs)
        ok = OrdinaryKriging(points[:, 0], points[:, 1], edge_resistance, **kwargs)
        grid_z, _ = ok.execute("grid", xi, yi)
        grid_z = np.asarray(grid_z)
    else:
        raise ValueError("interpolation must be 'midpoint', 'rbf', or 'kriging'")

    if not np.isfinite(grid_z).any():
        grid_z = griddata(points, edge_resistance, (grid_x, grid_y), method="nearest")

    if hull_path is not None:
        mask = hull_path.contains_points(
            np.column_stack([grid_x.ravel(), grid_y.ravel()])
        ).reshape(grid_x.shape)
        grid_z = np.where(mask, grid_z, np.nan)

    fig, ax = plt.subplots(figsize=(6, 5))
    extent = (xmin, xmax, ymin, ymax)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    if basemap is not None and basemap is not False:
        import contextily as ctx

        if basemap is True:
            basemap_source = ctx.providers.CartoDB.Positron
        else:
            basemap_source = basemap
        ctx.add_basemap(ax, source=basemap_source, crs=basemap_crs, reset_extent=False)

    im = ax.imshow(
        grid_z,
        origin="lower",
        extent=extent,
        cmap=cmap,
        aspect="auto",
        alpha=0.85,
        zorder=1,
    )
    plt.colorbar(im, ax=ax, label="Shared resistance")
    ax.set_xlabel("Longitude" if basemap is None or basemap is False else "X")
    ax.set_ylabel("Latitude" if basemap is None or basemap is False else "Y")

    if show_sites:
        if coord_order == "latlon":
            y_s = sp.sample_coords[:, 0]
            x_s = sp.sample_coords[:, 1]
        else:
            x_s = sp.sample_coords[:, 0]
            y_s = sp.sample_coords[:, 1]
        if basemap is not None and basemap is not False:
            from multispecies_resistance.graph import project_coords

            coords_s = project_coords(
                sp.sample_coords,
                coord_order=coord_order,
                coords_crs=coords_crs,
                target_crs=basemap_crs,
            )
            x_s = coords_s[:, 0]
            y_s = coords_s[:, 1]
        ax.scatter(x_s, y_s, s=10, c="black", alpha=0.75, zorder=3)

    surface = {
        "grid": grid_z,
        "x": xi,
        "y": yi,
        "extent": (xmin, xmax, ymin, ymax),
        "crs": crs,
    }
    if explore:
        import folium
        from matplotlib import cm, colors
        from rasterio.crs import CRS
        from rasterio.warp import transform

        grid = np.asarray(grid_z)
        vmin = float(np.nanmin(grid))
        vmax = float(np.nanmax(grid))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin, vmax = 0.0, 1.0
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        mapper = cm.get_cmap(cmap)
        rgba = mapper(norm(grid))
        rgba[..., 3] = np.where(np.isnan(grid), 0.0, rgba[..., 3])
        img = (rgba * 255).astype(np.uint8)

        if crs != "EPSG:4326":
            src_crs = CRS.from_user_input(crs)
            dst_crs = CRS.from_user_input("EPSG:4326")
            lon_min, lat_min = transform(src_crs, dst_crs, [xmin], [ymin])
            lon_max, lat_max = transform(src_crs, dst_crs, [xmax], [ymax])
            bounds = [[lat_min[0], lon_min[0]], [lat_max[0], lon_max[0]]]
        else:
            bounds = [[ymin, xmin], [ymax, xmax]]

        m = folium.Map()
        folium.raster_layers.ImageOverlay(image=img, bounds=bounds, opacity=0.85).add_to(m)
        return ax, surface, m

    return ax, surface


def plot_resistance_matrix(R: np.ndarray, ax=None, title: Optional[str] = None):
    """Plot a heatmap for a resistance matrix.

    Parameters
    ----------
    R : np.ndarray
        `N x N` resistance matrix.
    ax : matplotlib.axes.Axes | None, optional
        Existing axis; a new one is created when omitted.
    title : Optional[str], optional
        Title text.

    Returns
    -------
    matplotlib.axes.Axes
        Axis containing the matrix heatmap.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(R, cmap="magma")
    plt.colorbar(im, ax=ax, label="Effective resistance")
    if title:
        ax.set_title(title)
    return ax
