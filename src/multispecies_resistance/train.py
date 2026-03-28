from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch

from multispecies_resistance.data import SpeciesData, aggregate_site_genotypes, pairwise_site_distance
from multispecies_resistance.graph import (
    SpeciesGraph,
    build_edge_neighbor_pairs,
    build_dense_mesh_graph,
    build_geodesic_mesh_graph,
    compute_edge_support_weight,
    edge_features,
    project_coords,
    standardize_features,
)
from multispecies_resistance.model import MultiSpeciesResistanceModel
from multispecies_resistance.raster import RasterStack, resolve_raster_paths


torch.set_default_dtype(torch.float64)


def _resolved_raster_list(
    raster_paths: Iterable[str | Path] | None,
    raster_root: str | Path | None,
    raster_pattern: str,
    raster_recursive: bool,
) -> list[Path] | None:
    if raster_paths is None and raster_root is not None:
        raster_paths = resolve_raster_paths(
            raster_root,
            pattern=raster_pattern,
            recursive=raster_recursive,
        )
    elif raster_paths is not None:
        raster_paths = resolve_raster_paths(
            raster_paths,
            pattern=raster_pattern,
            recursive=raster_recursive,
        )

    if raster_paths is None:
        return None

    raster_paths = list(raster_paths)
    if not raster_paths:
        raise FileNotFoundError("No raster files found.")
    return raster_paths


def _sample_node_env(
    raster_stack: RasterStack | None,
    node_coords: np.ndarray,
    num_nodes: int,
) -> np.ndarray:
    if raster_stack is None:
        return np.zeros((num_nodes, 0), dtype=np.float64)
    node_env, _ = raster_stack.sample_points(node_coords)
    return node_env


def choose_mesh_spacing_km(
    species_list: List[SpeciesData],
    project_to: str = "EPSG:3857",
    coord_order: str = "latlon",
    coords_crs: str = "EPSG:4326",
    quantile: float = 0.5,
    scale_factor: float = 1.25,
    min_spacing_km: float = 5.0,
) -> float:
    """Choose a mesh spacing from per-species nearest-neighbor sample distances.

    The heuristic pools nearest-neighbor distances within each species,
    summarizes them with a robust quantile, and then applies a modest
    coarse-graining factor so the default mesh is slightly smoother than the
    raw sample spacing.
    """
    from scipy.spatial import cKDTree

    if not species_list:
        raise ValueError("species_list is empty")
    if not (0.0 < quantile <= 1.0):
        raise ValueError("quantile must be in (0, 1].")
    if scale_factor <= 0.0:
        raise ValueError("scale_factor must be > 0.")
    if min_spacing_km <= 0.0:
        raise ValueError("min_spacing_km must be > 0.")

    nn_km: list[np.ndarray] = []
    for sp in species_list:
        coords = np.asarray(sp.sample_coords, dtype=np.float64)
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(f"Species '{sp.name}' has invalid sample_coords shape.")
        if coords.shape[0] < 2:
            continue

        xy = project_coords(
            coords,
            coord_order=coord_order,
            coords_crs=coords_crs,
            target_crs=project_to,
        )
        tree = cKDTree(xy)
        dists, _ = tree.query(xy, k=2)
        nearest_m = np.asarray(dists[:, 1], dtype=np.float64)
        nearest_km = nearest_m[np.isfinite(nearest_m) & (nearest_m > 0.0)] / 1000.0
        if nearest_km.size > 0:
            nn_km.append(nearest_km)

    if not nn_km:
        raise ValueError("Cannot auto-pick mesh spacing: need at least one species with >=2 distinct samples.")

    pooled = np.concatenate(nn_km)
    spacing_km = float(np.quantile(pooled, quantile) * scale_factor)
    return max(spacing_km, min_spacing_km)


def build_species_graphs(
    species_list: List[SpeciesData],
    project_to: str | None = "EPSG:3857",
    coord_order: str = "latlon",
    coords_crs: str = "EPSG:4326",
    standardize: bool = True,
    mesh_spacing_km: float | None = None,
    mesh_spacing_deg: float | None = None,
    mesh_grid_type: str = "triangular",
    mesh_env: np.ndarray | None = None,
    buffer_km: float = 0.0,
    bbox: str | None = "square",
    bbox_file: str | None = None,
    input_graph: str | None = None,
    raster_paths: Iterable[str | Path] | None = None,
    raster_root: str | Path | None = None,
    raster_pattern: str = "*.tif",
    raster_recursive: bool = True,
    raster_fill_method: str = "nan",
    raster_coord_order: str = "latlon",
    raster_coords_crs: str = "EPSG:4326",
    use_geodesic: bool = False,
    support_decay_km: float | None = None,
    support_floor: float = 0.01,
) -> tuple[List[SpeciesGraph], dict | None]:
    """Convert sample-level species inputs into graph-based training datasets.

    When `input_graph` is provided, samples are assigned to the nearest `.gml`
    graph node. Otherwise a shared dense mesh is built from species sample
    coordinates and samples are assigned to the nearest mesh node. If
    `mesh_spacing_km` is `None`, a spacing is chosen automatically from
    nearest-neighbor sample distances.
    Environmental covariates are sampled at graph nodes when raster inputs are
    provided. Optional support attenuation weights can also be precomputed from
    graph distance to occupied nodes; setting `support_decay_km=None` disables
    this behavior.
    """
    if not species_list:
        raise ValueError("species_list is empty")
    if support_decay_km is not None and support_decay_km <= 0.0:
        raise ValueError("support_decay_km must be > 0 when provided.")
    if not (0.0 <= support_floor <= 1.0):
        raise ValueError("support_floor must lie in [0, 1].")

    graphs: List[SpeciesGraph] = []
    all_feats = []

    resolved_rasters = _resolved_raster_list(
        raster_paths,
        raster_root,
        raster_pattern,
        raster_recursive,
    )

    raster_stack: RasterStack | None = None
    if resolved_rasters is not None:
        raster_stack = RasterStack(
            resolved_rasters,
            coord_order=raster_coord_order,
            coords_crs=raster_coords_crs,
            fill_method=raster_fill_method,
        )

    try:
        if input_graph is not None:
            import networkx as nx
            from scipy.spatial import cKDTree

            gml = nx.read_gml(input_graph)
            nodes = list(gml.nodes())
            node_coords = []
            node_index = {}
            for idx, node in enumerate(nodes):
                data = gml.nodes[node]
                if "lat" in data and "lon" in data:
                    lat = float(data["lat"])
                    lon = float(data["lon"])
                elif "y" in data and "x" in data:
                    lat = float(data["y"])
                    lon = float(data["x"])
                elif "pos" in data:
                    lonlat = np.fromstring(data["pos"].strip('[]'), sep=' ')
                    lon = lonlat[0]
                    lat = lonlat[1]
                else:
                    raise ValueError("GML nodes must have (lat, lon) or (x, y) attributes.")
                node_coords.append([lat, lon])
                node_index[node] = idx

            node_coords = np.asarray(node_coords, dtype=np.float64)
            edge_index = []
            for u, v in gml.edges():
                edge_index.append((node_index[u], node_index[v]))
            edge_index = np.asarray(edge_index, dtype=np.int64)

            if raster_stack is not None and mesh_env is not None:
                raise ValueError("Provide either raster inputs or mesh_env, not both.")

            if raster_stack is not None:
                node_env = _sample_node_env(raster_stack, node_coords, node_coords.shape[0])
            elif mesh_env is not None:
                if mesh_env.shape[0] != node_coords.shape[0]:
                    raise ValueError("mesh_env must have the same number of rows as graph nodes.")
                node_env = mesh_env
            else:
                node_env = np.zeros((node_coords.shape[0], 0), dtype=np.float64)

            shared_edge_features = edge_features(node_coords, node_env, edge_index)
            edge_nbr_i, edge_nbr_j = build_edge_neighbor_pairs(edge_index, node_coords.shape[0])
            tree = cKDTree(node_coords)

            for sp in species_list:
                _, sample_sites = tree.query(sp.sample_coords, k=1)
                sample_sites = sample_sites.astype(np.int64)

                site_genos, site_counts = aggregate_site_genotypes(
                    sp.genotypes,
                    sample_sites,
                    num_sites=node_coords.shape[0],
                    allow_empty=True,
                )
                valid = np.where(site_counts > 0)[0]
                if valid.size < 2:
                    raise ValueError(f"Species '{sp.name}' has <2 occupied graph nodes.")

                dist = pairwise_site_distance(site_genos[valid])
                pair_i, pair_j = np.triu_indices(dist.shape[0], k=1)
                pair_dist = dist[pair_i, pair_j]
                pair_i = valid[pair_i]
                pair_j = valid[pair_j]
                edge_support_weight = None
                if support_decay_km is not None:
                    edge_support_weight = compute_edge_support_weight(
                        node_coords,
                        edge_index,
                        valid,
                        support_decay_km=support_decay_km,
                        support_floor=support_floor,
                    )

                graph = SpeciesGraph(
                    name=sp.name,
                    edge_index=edge_index,
                    edge_features=shared_edge_features.copy(),
                    node_coords=node_coords,
                    sample_coords=sp.sample_coords,
                    pair_i=pair_i,
                    pair_j=pair_j,
                    pair_dist=pair_dist,
                    num_nodes=node_coords.shape[0],
                    edge_nbr_i=edge_nbr_i,
                    edge_nbr_j=edge_nbr_j,
                    edge_support_weight=edge_support_weight,
                )
                graphs.append(graph)
                all_feats.append(shared_edge_features)

        else:
            from scipy.spatial import cKDTree

            if mesh_spacing_km is None and mesh_spacing_deg is None:
                mesh_spacing_km = choose_mesh_spacing_km(
                    species_list,
                    project_to=project_to or "EPSG:3857",
                    coord_order=coord_order,
                    coords_crs=coords_crs,
                )

            graph_fn = build_geodesic_mesh_graph if use_geodesic else build_dense_mesh_graph
            coords_list = [sp.sample_coords for sp in species_list]
            mesh_coords, edge_index = graph_fn(
                coords_list,
                spacing_km=mesh_spacing_km,
                spacing_deg=mesh_spacing_deg,
                grid_type=mesh_grid_type,
                project_to=project_to,
                coord_order=coord_order,
                coords_crs=coords_crs,
                buffer_km=buffer_km,
                bbox=bbox,
                bbox_file=bbox_file,
            )

            if raster_stack is not None and mesh_env is not None:
                raise ValueError("Provide either raster inputs or mesh_env, not both.")

            if raster_stack is not None:
                node_env = _sample_node_env(raster_stack, mesh_coords, mesh_coords.shape[0])
            elif mesh_env is not None:
                if mesh_env.shape[0] != mesh_coords.shape[0]:
                    raise ValueError("mesh_env must have the same number of rows as mesh_coords.")
                node_env = mesh_env
            else:
                node_env = np.zeros((mesh_coords.shape[0], 0), dtype=np.float64)

            shared_edge_features = edge_features(mesh_coords, node_env, edge_index)
            edge_nbr_i, edge_nbr_j = build_edge_neighbor_pairs(edge_index, mesh_coords.shape[0])
            tree = cKDTree(mesh_coords)

            for sp in species_list:
                _, sample_sites = tree.query(sp.sample_coords, k=1)
                sample_sites = sample_sites.astype(np.int64)

                site_genos, site_counts = aggregate_site_genotypes(
                    sp.genotypes,
                    sample_sites,
                    num_sites=mesh_coords.shape[0],
                    allow_empty=True,
                )
                valid = np.where(site_counts > 0)[0]
                if valid.size < 2:
                    raise ValueError(f"Species '{sp.name}' has <2 occupied mesh nodes.")

                dist = pairwise_site_distance(site_genos[valid])
                pair_i, pair_j = np.triu_indices(dist.shape[0], k=1)
                pair_dist = dist[pair_i, pair_j]
                pair_i = valid[pair_i]
                pair_j = valid[pair_j]
                edge_support_weight = None
                if support_decay_km is not None:
                    edge_support_weight = compute_edge_support_weight(
                        mesh_coords,
                        edge_index,
                        valid,
                        support_decay_km=support_decay_km,
                        support_floor=support_floor,
                    )

                graph = SpeciesGraph(
                    name=sp.name,
                    edge_index=edge_index,
                    edge_features=shared_edge_features.copy(),
                    node_coords=mesh_coords,
                    sample_coords=sp.sample_coords,
                    pair_i=pair_i,
                    pair_j=pair_j,
                    pair_dist=pair_dist,
                    num_nodes=mesh_coords.shape[0],
                    edge_nbr_i=edge_nbr_i,
                    edge_nbr_j=edge_nbr_j,
                    edge_support_weight=edge_support_weight,
                )
                graphs.append(graph)
                all_feats.append(shared_edge_features)

    finally:
        if raster_stack is not None:
            raster_stack.close()

    stats = None
    if standardize and all_feats:
        all_feats = np.vstack(all_feats)
        _, mean, std = standardize_features(all_feats)
        for g in graphs:
            g.edge_features = (g.edge_features - mean) / std
        stats = {"mean": mean, "std": std}

    return graphs, stats


def prepare_pairs(distance_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a full symmetric distance matrix into unique upper-triangle pairs.

    Parameters
    ----------
    distance_matrix : np.ndarray
        `N x N` pairwise distance matrix.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Pair row indices, pair column indices, and pair distances.
    """
    pair_i, pair_j = np.triu_indices(distance_matrix.shape[0], k=1)
    pair_dist = distance_matrix[pair_i, pair_j]
    return pair_i, pair_j, pair_dist


def split_pairs(
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    pair_dist: np.ndarray,
    num_nodes: int,
    val_fraction: float = 0.2,
    strategy: str = "site",
    seed: int = 0,
    min_val_pairs: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split pairwise targets into train and validation subsets.

    Parameters
    ----------
    pair_i : np.ndarray
        Row node indices for pair targets.
    pair_j : np.ndarray
        Column node indices for pair targets.
    pair_dist : np.ndarray
        Target distances for each `(pair_i, pair_j)` entry.
    num_nodes : int
        Total number of nodes represented by pair indices.
    val_fraction : float, optional
        Fraction of examples assigned to validation.
    strategy : str, optional
        `"site"` for site holdout, `"pair"` for random pair holdout.
    seed : int, optional
        Random seed for split reproducibility.
    min_val_pairs : int, optional
        Minimum pairs required in each split before accepting site-based holdout.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Train `pair_i`, train `pair_j`, train distances, validation `pair_i`,
        validation `pair_j`, validation distances.
    """
    if val_fraction <= 0.0 or val_fraction >= 1.0:
        raise ValueError("val_fraction must be in (0, 1)")

    rng = np.random.default_rng(seed)

    if strategy == "site" and num_nodes >= 4:
        n_val = max(1, int(round(val_fraction * num_nodes)))
        val_sites = rng.choice(num_nodes, size=n_val, replace=False)
        in_val_i = np.isin(pair_i, val_sites)
        in_val_j = np.isin(pair_j, val_sites)
        val_mask = in_val_i & in_val_j
        train_mask = (~in_val_i) & (~in_val_j)

        if val_mask.sum() >= min_val_pairs and train_mask.sum() >= min_val_pairs:
            return (
                pair_i[train_mask],
                pair_j[train_mask],
                pair_dist[train_mask],
                pair_i[val_mask],
                pair_j[val_mask],
                pair_dist[val_mask],
            )

    # fallback to random pair holdout
    num_pairs = pair_i.shape[0]
    n_val = max(1, int(round(val_fraction * num_pairs)))
    perm = rng.permutation(num_pairs)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    return (
        pair_i[train_idx],
        pair_j[train_idx],
        pair_dist[train_idx],
        pair_i[val_idx],
        pair_j[val_idx],
        pair_dist[val_idx],
    )


def train_model(
    species_graphs: List[SpeciesGraph],
    hidden_dim: int = 32,
    lr: float = 1e-2,
    epochs: int = 200,
    l2_shared: float = 1e-4,
    l2_species: float = 1e-4,
    edge_smoothing: float = 0.0,
    log_every: int = 25,
    val_fraction: float = 0.2,
    val_strategy: str = "site",
    min_val_pairs: int = 50,
    patience: int = 30,
    min_delta: float = 1e-4,
    restore_best: bool = True,
    seed: int = 0,
) -> MultiSpeciesResistanceModel:
    """Train the multi-species resistance model on graph pair-distance targets.

    Parameters
    ----------
    species_graphs : List[SpeciesGraph]
        Per-species graph features and pairwise targets.
    hidden_dim : int, optional
        Hidden width for model edge networks.
    lr : float, optional
        Adam learning rate.
    epochs : int, optional
        Maximum number of optimization epochs.
    l2_shared : float, optional
        L2 penalty weight for shared logits.
    l2_species : float, optional
        L2 penalty weight for species logits.
    edge_smoothing : float, optional
        Edge-neighbor smoothing control in `[0, 1]`. Larger values more strongly
        penalize differences between logits on edges that share a node.
    log_every : int, optional
        Epoch interval for console logging.
    val_fraction : float, optional
        Validation fraction for automatic split when validation pairs are absent.
    val_strategy : str, optional
        Validation split strategy (`"site"` or `"pair"`).
    min_val_pairs : int, optional
        Minimum validation pair count for site-based splitting.
    patience : int, optional
        Early-stopping patience in epochs.
    min_delta : float, optional
        Minimum validation improvement to reset patience.
    restore_best : bool, optional
        Whether to restore best validation parameters after early stop.
    seed : int, optional
        Base random seed used during split creation.

    Returns
    -------
    MultiSpeciesResistanceModel
        Trained model instance.
    """
    if not (0.0 <= edge_smoothing <= 1.0):
        raise ValueError("edge_smoothing must be in [0, 1].")

    num_species = len(species_graphs)
    edge_feat_dim = species_graphs[0].edge_features.shape[1]
    smooth_weight = (edge_smoothing / max(1e-6, 1.0 - edge_smoothing)) ** 2

    model = MultiSpeciesResistanceModel(num_species, edge_feat_dim, hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if val_fraction and val_fraction > 0:
        for s, g in enumerate(species_graphs):
            if g.val_pair_i is None:
                (
                    g.pair_i,
                    g.pair_j,
                    g.pair_dist,
                    g.val_pair_i,
                    g.val_pair_j,
                    g.val_pair_dist,
                ) = split_pairs(
                    g.pair_i,
                    g.pair_j,
                    g.pair_dist,
                    g.num_nodes,
                    val_fraction=val_fraction,
                    strategy=val_strategy,
                    seed=seed + s,
                    min_val_pairs=min_val_pairs,
                )

    best_val = float("inf")
    best_state = None
    epochs_since_improve = 0

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        total_loss = 0.0
        total_val = 0.0
        has_val = False
        val_count = 0

        for s, g in enumerate(species_graphs):
            edge_feat = torch.from_numpy(g.edge_features)
            edge_index = torch.from_numpy(g.edge_index)
            edge_support_weight = (
                None
                if g.edge_support_weight is None
                else torch.from_numpy(g.edge_support_weight)
            )

            R, shared_logits, species_logits = model.resistance_matrix(
                s, edge_index, edge_feat, g.num_nodes, edge_support_weight=edge_support_weight
            )
            pred = model.alpha[s] + model.beta[s] * R[g.pair_i, g.pair_j]

            dist = torch.from_numpy(g.pair_dist)
            mse = torch.mean((pred - dist) ** 2)

            reg = l2_shared * torch.mean(shared_logits ** 2) + l2_species * torch.mean(
                species_logits ** 2
            )
            smooth_penalty = shared_logits.new_tensor(0.0)
            if (
                smooth_weight > 0.0
                and g.edge_nbr_i is not None
                and g.edge_nbr_j is not None
                and g.edge_nbr_i.size > 0
            ):
                edge_nbr_i = torch.from_numpy(g.edge_nbr_i)
                edge_nbr_j = torch.from_numpy(g.edge_nbr_j)
                combined_logits = shared_logits + species_logits
                smooth_penalty = torch.mean(
                    (combined_logits[edge_nbr_i] - combined_logits[edge_nbr_j]) ** 2
                )

            loss = mse + reg + smooth_weight * smooth_penalty
            total_loss = total_loss + loss

            if g.val_pair_i is not None and g.val_pair_dist is not None:
                has_val = True
                val_count += 1
                val_pred = model.alpha[s] + model.beta[s] * R[g.val_pair_i, g.val_pair_j]
                val_dist = torch.from_numpy(g.val_pair_dist)
                val_mse = torch.mean((val_pred - val_dist) ** 2)
                total_val = total_val + val_mse

        total_loss.backward()
        optimizer.step()

        if has_val:
            val_loss = (total_val / max(1, val_count)).item()
            if val_loss < best_val - min_delta:
                best_val = val_loss
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                epochs_since_improve = 0
            else:
                epochs_since_improve += 1

            if patience and epochs_since_improve >= patience:
                if restore_best and best_state is not None:
                    model.load_state_dict(best_state)
                if log_every:
                    print(
                        f"early stop at epoch {epoch} val {val_loss:.6f} best {best_val:.6f}"
                    )
                break

        if log_every and epoch % log_every == 0:
            if has_val:
                print(
                    f"epoch {epoch:4d} loss {total_loss.item():.6f} val {val_loss:.6f}"
                )
            else:
                print(f"epoch {epoch:4d} loss {total_loss.item():.6f}")

    return model
