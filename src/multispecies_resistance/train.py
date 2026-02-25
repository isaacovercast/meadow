from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import torch

from multispecies_resistance.model import MultiSpeciesResistanceModel
from multispecies_resistance.data import aggregate_site_genotypes, pairwise_site_distance
from multispecies_resistance.graph import (
    build_delaunay_graph,
    build_dense_mesh_graph,
    build_knn_graph,
    edge_features,
    standardize_features,
)


torch.set_default_dtype(torch.float64)


@dataclass
class SpeciesGraphData:
    """Per-species graph features and pairwise distance targets used for training."""

    name: str
    edge_index: np.ndarray
    edge_features: np.ndarray
    node_coords: np.ndarray
    pair_i: np.ndarray
    pair_j: np.ndarray
    pair_dist: np.ndarray
    num_nodes: int
    val_pair_i: np.ndarray | None = None
    val_pair_j: np.ndarray | None = None
    val_pair_dist: np.ndarray | None = None


def build_species_graphs(
    species_list: List,
    graph_type: str = "delaunay",
    k: int = 6,
    project_to: str | None = "EPSG:3857",
    coord_order: str = "latlon",
    coords_crs: str = "EPSG:4326",
    standardize: bool = True,
    mesh_spacing_km: float | None = 80.0,
    mesh_spacing_deg: float | None = None,
    mesh_grid_type: str = "triangular",
    mesh_graph_type: str = "delaunay",
    mesh_k: int = 6,
    mesh_coords: np.ndarray | None = None,
    mesh_env: np.ndarray | None = None,
    buffer_km: float = 0.0,
    bbox: str | None = "square",
    bbox_file: str | None = None,
    input_graph: str | None = None,
) -> tuple[List[SpeciesGraphData], dict | None]:
    """Convert species inputs into graph-based training datasets.

    Parameters
    ----------
    species_list : List
        Sequence of species records with site coordinates, assignments, and genotypes.
    graph_type : str, optional
        Graph strategy: `"delaunay"`, `"knn"`, or `"dense_mesh"`.
    k : int, optional
        Neighbor count when `graph_type="knn"`.
    project_to : str | None, optional
        Optional projection CRS for graph construction.
    coord_order : str, optional
        Coordinate order for input site coordinates.
    coords_crs : str, optional
        CRS of input coordinates.
    standardize : bool, optional
        Whether to standardize edge features across all species.
    mesh_spacing_km : float | None, optional
        Shared mesh spacing in kilometers for dense-mesh mode.
    mesh_spacing_deg : float | None, optional
        Shared mesh spacing in degrees for dense-mesh mode.
    mesh_grid_type : str, optional
        Grid shape for dense-mesh node generation.
    mesh_graph_type : str, optional
        Graph builder for dense mesh (`"delaunay"` or `"knn"`).
    mesh_k : int, optional
        `k` for dense-mesh knn graphs.
    mesh_coords : np.ndarray | None, optional
        Precomputed shared mesh coordinates.
    mesh_env : np.ndarray | None, optional
        Environmental covariates defined on `mesh_coords`.
    buffer_km : float, optional
        Bounding-area buffer before mesh generation.
    bbox : str | None, optional
        Mesh clipping mode.
    bbox_file : str | None, optional
        Polygon file used for mesh clipping.
    input_graph : str | None, optional
        Optional path to a global GML graph overriding graph construction.

    Returns
    -------
    tuple[List[SpeciesGraphData], dict | None]
        Built graph objects and optional standardization statistics (`mean`, `std`).
    """
    graphs: List[SpeciesGraphData] = []
    all_feats = []

    graph_type = graph_type.lower()
    if input_graph is not None:
        import networkx as nx

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
            else:
                raise ValueError("GML nodes must have (lat, lon) or (x, y) attributes.")
            node_coords.append([lat, lon])
            node_index[node] = idx

        node_coords = np.asarray(node_coords, dtype=np.float64)
        edge_index = []
        for u, v in gml.edges():
            edge_index.append((node_index[u], node_index[v]))
        edge_index = np.asarray(edge_index, dtype=np.int64)

        if mesh_env is None:
            mesh_env = np.zeros((node_coords.shape[0], 0), dtype=np.float64)
        elif mesh_env.shape[0] != node_coords.shape[0]:
            raise ValueError("mesh_env must have the same number of rows as GML nodes.")
        shared_edge_features = edge_features(node_coords, mesh_env, edge_index)

        from scipy.spatial import cKDTree

        tree = cKDTree(node_coords)

        for sp in species_list:
            _, site_to_mesh = tree.query(sp.site_coords, k=1)
            sample_sites = site_to_mesh[sp.sample_sites]

            site_genos, site_counts = aggregate_site_genotypes(
                sp.genotypes, sample_sites, num_sites=node_coords.shape[0], allow_empty=True
            )
            valid = np.where(site_counts > 0)[0]
            if valid.size < 2:
                raise ValueError(f"Species '{sp.name}' has <2 occupied mesh nodes.")

            dist = pairwise_site_distance(site_genos[valid])
            pair_i, pair_j = np.triu_indices(dist.shape[0], k=1)
            pair_dist = dist[pair_i, pair_j]
            pair_i = valid[pair_i]
            pair_j = valid[pair_j]

            graph = SpeciesGraphData(
                name=sp.name,
                edge_index=edge_index,
                edge_features=shared_edge_features.copy(),
                node_coords=node_coords,
                pair_i=pair_i,
                pair_j=pair_j,
                pair_dist=pair_dist,
                num_nodes=node_coords.shape[0],
            )
            graphs.append(graph)
            all_feats.append(shared_edge_features)

        # skip standardization of mesh_env features, handled later

    elif graph_type == "dense_mesh":
        if mesh_coords is None:
            coords_list = [sp.site_coords for sp in species_list]
            mesh_coords, edge_index = build_dense_mesh_graph(
                coords_list,
                spacing_km=mesh_spacing_km,
                spacing_deg=mesh_spacing_deg,
                grid_type=mesh_grid_type,
                mesh_graph_type=mesh_graph_type,
                k=mesh_k,
                project_to=project_to,
                coord_order=coord_order,
                coords_crs=coords_crs,
                buffer_km=buffer_km,
                bbox=bbox,
                bbox_file=bbox_file,
            )
        else:
            if mesh_graph_type == "knn":
                edge_index = build_knn_graph(
                    mesh_coords,
                    k=mesh_k,
                    project_to=project_to,
                    coord_order=coord_order,
                    coords_crs=coords_crs,
                )
            else:
                edge_index = build_delaunay_graph(
                    mesh_coords,
                    project_to=project_to,
                    coord_order=coord_order,
                    coords_crs=coords_crs,
                )

        if mesh_env is None:
            mesh_env = np.zeros((mesh_coords.shape[0], 0), dtype=np.float64)

        # Precompute shared edge features
        shared_edge_features = edge_features(mesh_coords, mesh_env, edge_index)

        from scipy.spatial import cKDTree

        tree = cKDTree(mesh_coords)

        for sp in species_list:
            # Map each species site to nearest mesh node, then assign samples to mesh nodes
            _, site_to_mesh = tree.query(sp.site_coords, k=1)
            sample_sites = site_to_mesh[sp.sample_sites]

            site_genos, site_counts = aggregate_site_genotypes(
                sp.genotypes, sample_sites, num_sites=mesh_coords.shape[0], allow_empty=True
            )
            valid = np.where(site_counts > 0)[0]
            if valid.size < 2:
                raise ValueError(f"Species '{sp.name}' has <2 occupied mesh nodes.")

            dist = pairwise_site_distance(site_genos[valid])
            pair_i, pair_j = np.triu_indices(dist.shape[0], k=1)
            pair_dist = dist[pair_i, pair_j]
            pair_i = valid[pair_i]
            pair_j = valid[pair_j]

            graph = SpeciesGraphData(
                name=sp.name,
                edge_index=edge_index,
                edge_features=shared_edge_features.copy(),
                node_coords=mesh_coords,
                pair_i=pair_i,
                pair_j=pair_j,
                pair_dist=pair_dist,
                num_nodes=mesh_coords.shape[0],
            )
            graphs.append(graph)
            all_feats.append(shared_edge_features)
    else:
        for sp in species_list:
            site_genos, _ = aggregate_site_genotypes(
                sp.genotypes, sp.sample_sites, num_sites=sp.num_sites()
            )
            dist = pairwise_site_distance(site_genos)

            if graph_type == "delaunay":
                edges = build_delaunay_graph(
                    sp.site_coords,
                    project_to=project_to,
                    coord_order=coord_order,
                    coords_crs=coords_crs,
                )
            elif graph_type == "knn":
                edges = build_knn_graph(
                    sp.site_coords,
                    k=k,
                    project_to=project_to,
                    coord_order=coord_order,
                    coords_crs=coords_crs,
                )
            else:
                raise ValueError("graph_type must be 'delaunay', 'knn', or 'dense_mesh'")

            feats = edge_features(sp.site_coords, sp.site_env, edges)
            pair_i, pair_j = np.triu_indices(dist.shape[0], k=1)
            pair_dist = dist[pair_i, pair_j]

            graph = SpeciesGraphData(
                name=sp.name,
                edge_index=edges,
                edge_features=feats,
                node_coords=sp.site_coords,
                pair_i=pair_i,
                pair_j=pair_j,
                pair_dist=pair_dist,
                num_nodes=dist.shape[0],
            )
            graphs.append(graph)
            all_feats.append(feats)

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
    species_graphs: List[SpeciesGraphData],
    hidden_dim: int = 32,
    lr: float = 1e-2,
    epochs: int = 200,
    l2_shared: float = 1e-4,
    l2_species: float = 1e-4,
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
    species_graphs : List[SpeciesGraphData]
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
    num_species = len(species_graphs)
    edge_feat_dim = species_graphs[0].edge_features.shape[1]

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

            R, shared_logits, species_logits = model.resistance_matrix(
                s, edge_index, edge_feat, g.num_nodes
            )
            pred = model.alpha[s] + model.beta[s] * R[g.pair_i, g.pair_j]

            dist = torch.from_numpy(g.pair_dist)
            mse = torch.mean((pred - dist) ** 2)

            reg = l2_shared * torch.mean(shared_logits ** 2) + l2_species * torch.mean(
                species_logits ** 2
            )
            loss = mse + reg
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
