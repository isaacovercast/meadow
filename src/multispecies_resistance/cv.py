from __future__ import annotations

import sys
from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Sequence

import numpy as np
import torch
from scipy.spatial import cKDTree

from multispecies_resistance.graph import SpeciesGraph, haversine_km, project_coords
from multispecies_resistance.train import train_model


@dataclass
class SpeciesFoldStats:
    """Summary of one species' data support inside a cross-validation fold.

    Parameters
    ----------
    train_pairs : int
        Number of pairwise training targets kept outside the held-out fold.
    val_pairs : int
        Number of pairwise validation targets fully contained in the held-out fold.
    train_nodes : int
        Number of occupied graph nodes available for training.
    val_nodes : int
        Number of occupied graph nodes inside the held-out fold.
    train_samples : int
        Number of observed samples assigned to training nodes.
    val_samples : int
        Number of observed samples assigned to validation nodes.
    used_for_training : bool
        Whether this species is retained in model fitting for the fold.
    used_for_validation : bool
        Whether this species contributes to fold scoring.
    """

    train_pairs: int
    val_pairs: int
    train_nodes: int
    val_nodes: int
    train_samples: int
    val_samples: int
    used_for_training: bool
    used_for_validation: bool


@dataclass
class GraphCVFoldSummary:
    """Compact description of one graph-based cross-validation fold.

    Parameters
    ----------
    fold_id : int
        Integer identifier of the fold.
    num_nodes : int
        Number of graph nodes assigned to the fold.
    total_samples : int
        Total observed samples from all species assigned to fold nodes.
    num_supported_species : int
        Number of species that satisfy the validation thresholds in the fold.
    species_stats : dict[str, SpeciesFoldStats]
        Per-species training and validation support summary.
    """

    fold_id: int
    num_nodes: int
    total_samples: int
    num_supported_species: int
    species_stats: dict[str, SpeciesFoldStats]


@dataclass
class EdgeSmoothingCVCandidate:
    """Cross-validation scores for one candidate edge-smoothing value.

    Parameters
    ----------
    edge_smoothing : float
        Candidate smoothing value that was evaluated.
    mean_validation_loss : float
        Mean fold loss averaged after species balancing.
    std_validation_loss : float
        Standard deviation of fold losses.
    fold_losses : np.ndarray
        Validation loss for each evaluated fold.
    species_counts : np.ndarray
        Number of scored species in each evaluated fold.
    """

    edge_smoothing: float
    mean_validation_loss: float
    std_validation_loss: float
    fold_losses: np.ndarray
    species_counts: np.ndarray


@dataclass
class EdgeSmoothingCVResult:
    """Best smoothing choice plus the fold structure and candidate scores.

    Parameters
    ----------
    best_edge_smoothing : float
        Candidate smoothing value with the lowest mean validation loss.
    candidates : list[EdgeSmoothingCVCandidate]
        All evaluated candidates and their scores.
    fold_assignments : np.ndarray
        Length-`N` fold id for each shared graph node.
    fold_summaries : list[GraphCVFoldSummary]
        Validation-support summary for each active fold.
    """

    best_edge_smoothing: float
    candidates: list[EdgeSmoothingCVCandidate]
    fold_assignments: np.ndarray
    fold_summaries: list[GraphCVFoldSummary]


def _validate_shared_graph(species_graphs: Sequence[SpeciesGraph]) -> SpeciesGraph:
    """Validate that all species graphs share the same node and edge scaffold."""
    if not species_graphs:
        raise ValueError("species_graphs is empty.")

    template = species_graphs[0]
    for graph in species_graphs[1:]:
        if graph.num_nodes != template.num_nodes:
            raise ValueError("All species graphs must have the same num_nodes.")
        if not np.array_equal(graph.edge_index, template.edge_index):
            raise ValueError("All species graphs must share the same edge_index.")
        if not np.allclose(graph.node_coords, template.node_coords):
            raise ValueError("All species graphs must share the same node_coords.")
    return template


def _sample_node_counts(
    species_graphs: Sequence[SpeciesGraph],
    project_to: str,
    coord_order: str,
    coords_crs: str,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Assign samples to nearest shared graph nodes and count them per species."""
    template = _validate_shared_graph(species_graphs)
    node_xy = project_coords(
        template.node_coords,
        coord_order=coord_order,
        coords_crs=coords_crs,
        target_crs=project_to,
    )
    tree = cKDTree(node_xy)

    counts = np.zeros((len(species_graphs), template.num_nodes), dtype=np.int64)
    sample_sites: list[np.ndarray] = []
    for species_idx, graph in enumerate(species_graphs):
        sample_xy = project_coords(
            graph.sample_coords,
            coord_order=coord_order,
            coords_crs=coords_crs,
            target_crs=project_to,
        )
        _, assigned = tree.query(sample_xy, k=1)
        assigned = np.asarray(assigned, dtype=np.int64)
        counts[species_idx] = np.bincount(assigned, minlength=template.num_nodes)
        sample_sites.append(assigned)
    return counts, sample_sites


def _build_adjacency(node_coords: np.ndarray, edge_index: np.ndarray) -> list[list[tuple[int, float]]]:
    """Build weighted undirected adjacency lists from a shared edge list."""
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(node_coords.shape[0])]
    edge_lengths = haversine_km(node_coords[edge_index[:, 0]], node_coords[edge_index[:, 1]])
    edge_lengths = np.maximum(edge_lengths.astype(np.float64), 1e-9)
    for edge_id, (u, v) in enumerate(edge_index):
        i = int(u)
        j = int(v)
        w = float(edge_lengths[edge_id])
        adjacency[i].append((j, w))
        adjacency[j].append((i, w))
    return adjacency


def _select_seed_nodes(
    node_xy: np.ndarray,
    occupied_nodes: np.ndarray,
    node_weights: np.ndarray,
    n_folds: int,
) -> np.ndarray:
    """Choose seed nodes by weighted farthest-point sampling over occupied nodes."""
    if occupied_nodes.size < 2:
        raise ValueError("Need at least two occupied graph nodes for cross-validation.")

    n_folds = min(int(n_folds), int(occupied_nodes.size))
    weights = np.maximum(node_weights[occupied_nodes].astype(np.float64), 1.0)

    first_seed = int(occupied_nodes[np.argmax(weights)])
    seeds = [first_seed]

    diff = node_xy[occupied_nodes] - node_xy[first_seed]
    min_dist2 = np.sum(diff * diff, axis=1)
    chosen_mask = occupied_nodes == first_seed

    while len(seeds) < n_folds:
        scores = np.sqrt(np.maximum(min_dist2, 0.0)) * weights
        scores[chosen_mask] = -np.inf
        next_seed = int(occupied_nodes[np.argmax(scores)])
        seeds.append(next_seed)
        chosen_mask |= occupied_nodes == next_seed

        diff = node_xy[occupied_nodes] - node_xy[next_seed]
        min_dist2 = np.minimum(min_dist2, np.sum(diff * diff, axis=1))

    return np.asarray(seeds, dtype=np.int64)


def _assign_nodes_to_folds(
    adjacency: list[list[tuple[int, float]]],
    node_xy: np.ndarray,
    seeds: np.ndarray,
) -> np.ndarray:
    """Assign each graph node to the nearest seed by graph distance."""
    num_nodes = len(adjacency)
    fold_ids = np.full(num_nodes, -1, dtype=np.int64)
    best_dist = np.full(num_nodes, np.inf, dtype=np.float64)
    heap: list[tuple[float, int, int]] = []

    for fold_id, seed in enumerate(np.asarray(seeds, dtype=np.int64)):
        fold_ids[seed] = fold_id
        best_dist[seed] = 0.0
        heappush(heap, (0.0, fold_id, int(seed)))

    while heap:
        dist, fold_id, node = heappop(heap)
        if dist > best_dist[node] + 1e-12:
            continue
        if abs(dist - best_dist[node]) <= 1e-12 and fold_id != fold_ids[node]:
            continue

        for nbr, weight in adjacency[node]:
            cand = dist + weight
            if cand < best_dist[nbr] - 1e-12 or (
                abs(cand - best_dist[nbr]) <= 1e-12 and fold_id < fold_ids[nbr]
            ):
                best_dist[nbr] = cand
                fold_ids[nbr] = fold_id
                heappush(heap, (cand, fold_id, nbr))

    if np.any(fold_ids < 0):
        seed_xy = node_xy[seeds]
        tree = cKDTree(seed_xy)
        unassigned = np.where(fold_ids < 0)[0]
        _, nearest_seed = tree.query(node_xy[unassigned], k=1)
        fold_ids[unassigned] = np.asarray(nearest_seed, dtype=np.int64)

    return fold_ids


def _relabel_folds(fold_ids: np.ndarray) -> np.ndarray:
    """Relabel fold ids so they are consecutive integers starting at zero."""
    active = sorted(int(fold) for fold in np.unique(fold_ids))
    mapping = {old: new for new, old in enumerate(active)}
    return np.asarray([mapping[int(fold)] for fold in fold_ids], dtype=np.int64)


def _fold_masks(
    graph: SpeciesGraph,
    fold_ids: np.ndarray,
    held_out_fold: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct train and validation pair masks for one held-out fold."""
    in_fold_i = fold_ids[graph.pair_i] == held_out_fold
    in_fold_j = fold_ids[graph.pair_j] == held_out_fold
    val_mask = in_fold_i & in_fold_j
    train_mask = (~in_fold_i) & (~in_fold_j)
    return train_mask, val_mask


def _summarize_folds(
    species_graphs: Sequence[SpeciesGraph],
    node_sample_counts: np.ndarray,
    fold_ids: np.ndarray,
    min_train_pairs: int,
    min_val_pairs: int,
    min_train_nodes: int,
    min_val_nodes: int,
) -> list[GraphCVFoldSummary]:
    """Summarize training and validation support for each fold and species."""
    summaries: list[GraphCVFoldSummary] = []
    active_folds = sorted(int(fold) for fold in np.unique(fold_ids))

    for fold_id in active_folds:
        node_mask = fold_ids == fold_id
        species_stats: dict[str, SpeciesFoldStats] = {}
        supported = 0

        for species_idx, graph in enumerate(species_graphs):
            train_mask, val_mask = _fold_masks(graph, fold_ids, fold_id)
            counts = node_sample_counts[species_idx]

            val_nodes = int(np.count_nonzero(counts[node_mask] > 0))
            train_nodes = int(np.count_nonzero(counts[~node_mask] > 0))
            val_samples = int(np.sum(counts[node_mask]))
            train_samples = int(np.sum(counts[~node_mask]))
            train_pairs = int(np.count_nonzero(train_mask))
            val_pairs = int(np.count_nonzero(val_mask))

            used_for_training = (
                train_pairs >= min_train_pairs
                and train_nodes >= min_train_nodes
                and train_samples > 0
            )
            used_for_validation = (
                used_for_training
                and val_pairs >= min_val_pairs
                and val_nodes >= min_val_nodes
                and val_samples > 0
            )
            if used_for_validation:
                supported += 1

            species_stats[graph.name] = SpeciesFoldStats(
                train_pairs=train_pairs,
                val_pairs=val_pairs,
                train_nodes=train_nodes,
                val_nodes=val_nodes,
                train_samples=train_samples,
                val_samples=val_samples,
                used_for_training=used_for_training,
                used_for_validation=used_for_validation,
            )

        summaries.append(
            GraphCVFoldSummary(
                fold_id=fold_id,
                num_nodes=int(np.count_nonzero(node_mask)),
                total_samples=int(np.sum(node_sample_counts[:, node_mask])),
                num_supported_species=supported,
                species_stats=species_stats,
            )
        )

    return summaries


def _fold_neighbors(edge_index: np.ndarray, fold_ids: np.ndarray, fold_id: int) -> dict[int, int]:
    """Count boundary edges between one fold and its neighboring folds."""
    neighbors: dict[int, int] = {}
    for u, v in np.asarray(edge_index, dtype=np.int64):
        fold_u = int(fold_ids[u])
        fold_v = int(fold_ids[v])
        if fold_u == fold_v:
            continue
        if fold_u == fold_id:
            neighbors[fold_v] = neighbors.get(fold_v, 0) + 1
        elif fold_v == fold_id:
            neighbors[fold_u] = neighbors.get(fold_u, 0) + 1
    return neighbors


def build_graph_cv_folds(
    species_graphs: Sequence[SpeciesGraph],
    n_folds: int = 5,
    project_to: str = "EPSG:3857",
    coord_order: str = "latlon",
    coords_crs: str = "EPSG:4326",
    min_train_pairs: int = 25,
    min_val_pairs: int = 10,
    min_train_nodes: int = 3,
    min_val_nodes: int = 2,
    min_supported_species: int | None = None,
) -> tuple[np.ndarray, list[GraphCVFoldSummary]]:
    """Partition a shared graph into density-adaptive spatial CV folds.

    The partition is defined on graph nodes, seeded by weighted occupied nodes,
    and then merged until each fold has enough per-species support to be useful
    for validation.
    """
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2.")

    template = _validate_shared_graph(species_graphs)
    if min_supported_species is None:
        min_supported_species = min(2, len(species_graphs))

    node_sample_counts, _ = _sample_node_counts(
        species_graphs,
        project_to=project_to,
        coord_order=coord_order,
        coords_crs=coords_crs,
    )
    node_weights = np.sum(node_sample_counts, axis=0)
    occupied_nodes = np.flatnonzero(node_weights > 0)
    if occupied_nodes.size < 2:
        raise ValueError("Need at least two occupied graph nodes for cross-validation.")

    node_xy = project_coords(
        template.node_coords,
        coord_order=coord_order,
        coords_crs=coords_crs,
        target_crs=project_to,
    )
    seeds = _select_seed_nodes(node_xy, occupied_nodes, node_weights, n_folds=n_folds)
    adjacency = _build_adjacency(template.node_coords, template.edge_index)
    fold_ids = _assign_nodes_to_folds(adjacency, node_xy, seeds)
    fold_ids = _relabel_folds(fold_ids)

    while True:
        summaries = _summarize_folds(
            species_graphs,
            node_sample_counts=node_sample_counts,
            fold_ids=fold_ids,
            min_train_pairs=min_train_pairs,
            min_val_pairs=min_val_pairs,
            min_train_nodes=min_train_nodes,
            min_val_nodes=min_val_nodes,
        )
        if len(summaries) <= 2:
            break

        weakest = min(
            summaries,
            key=lambda summary: (
                summary.num_supported_species,
                summary.total_samples,
                summary.num_nodes,
            ),
        )
        if weakest.num_supported_species >= min_supported_species:
            break

        neighbors = _fold_neighbors(template.edge_index, fold_ids, weakest.fold_id)
        if not neighbors:
            break

        merge_target = max(
            neighbors,
            key=lambda fold_id: (
                neighbors[fold_id],
                next(summary.total_samples for summary in summaries if summary.fold_id == fold_id),
            ),
        )
        fold_ids[fold_ids == weakest.fold_id] = int(merge_target)
        fold_ids = _relabel_folds(fold_ids)

    summaries = _summarize_folds(
        species_graphs,
        node_sample_counts=node_sample_counts,
        fold_ids=fold_ids,
        min_train_pairs=min_train_pairs,
        min_val_pairs=min_val_pairs,
        min_train_nodes=min_train_nodes,
        min_val_nodes=min_val_nodes,
    )

    if len(summaries) < 2:
        raise ValueError("Cross-validation partition collapsed to fewer than two usable folds.")
    supported_folds = sum(summary.num_supported_species > 0 for summary in summaries)
    if supported_folds < 2:
        raise ValueError("Need at least two folds with usable validation support.")

    return fold_ids, summaries


def _clone_graph_for_fold(
    graph: SpeciesGraph,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
) -> SpeciesGraph | None:
    """Create a fold-specific graph copy with explicit train and validation pairs."""
    if int(np.count_nonzero(train_mask)) == 0:
        return None

    return SpeciesGraph(
        name=graph.name,
        edge_index=graph.edge_index,
        edge_features=graph.edge_features,
        node_coords=graph.node_coords,
        sample_coords=graph.sample_coords,
        pair_i=graph.pair_i[train_mask].copy(),
        pair_j=graph.pair_j[train_mask].copy(),
        pair_dist=graph.pair_dist[train_mask].copy(),
        num_nodes=graph.num_nodes,
        edge_nbr_i=graph.edge_nbr_i,
        edge_nbr_j=graph.edge_nbr_j,
        edge_support_weight=graph.edge_support_weight,
        val_pair_i=graph.pair_i[val_mask].copy() if np.any(val_mask) else None,
        val_pair_j=graph.pair_j[val_mask].copy() if np.any(val_mask) else None,
        val_pair_dist=graph.pair_dist[val_mask].copy() if np.any(val_mask) else None,
    )


def _evaluate_fold_model(
    model,
    species_graphs: Sequence[SpeciesGraph],
) -> tuple[float, int]:
    """Compute species-balanced validation MSE for one trained fold model."""
    model.eval()
    species_losses: list[float] = []
    with torch.no_grad():
        for species_idx, graph in enumerate(species_graphs):
            if graph.val_pair_i is None or graph.val_pair_dist is None:
                continue

            edge_feat = torch.from_numpy(graph.edge_features)
            edge_index = torch.from_numpy(graph.edge_index)
            resistance, _, _ = model.resistance_matrix(
                species_idx,
                edge_index,
                edge_feat,
                graph.num_nodes,
            )
            pred = model.alpha[species_idx] + model.beta[species_idx] * resistance[
                graph.val_pair_i,
                graph.val_pair_j,
            ]
            target = torch.from_numpy(graph.val_pair_dist)
            species_losses.append(float(torch.mean((pred - target) ** 2).item()))

    if not species_losses:
        raise ValueError("Fold evaluation produced no validation losses.")
    return float(np.mean(species_losses)), len(species_losses)


def _print_progress_line(message: str, width: int = 120) -> None:
    """Render one in-place progress line in interactive terminal output."""
    clipped = message[:width]
    padded = clipped.ljust(width)
    sys.stdout.write(f"\r{padded}")
    sys.stdout.flush()


def _finish_progress_line() -> None:
    """Terminate an in-place progress line with a newline."""
    sys.stdout.write("\n")
    sys.stdout.flush()


def choose_edge_smoothing_cv(
    species_graphs: Sequence[SpeciesGraph],
    smoothing_values: Sequence[float] = (0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 0.93, 0.97),
    n_folds: int = 5,
    project_to: str = "EPSG:3857",
    coord_order: str = "latlon",
    coords_crs: str = "EPSG:4326",
    min_train_pairs: int = 25,
    min_val_pairs: int = 10,
    min_train_nodes: int = 3,
    min_val_nodes: int = 2,
    min_supported_species: int | None = None,
    hidden_dim: int = 16,
    lr: float = 2e-2,
    epochs: int = 400,
    l2_shared: float = 1e-4,
    l2_species: float = 1e-4,
    log_every: int = 0,
    patience: int = 12,
    min_delta: float = 5e-4,
    restore_best: bool = True,
    seed: int = 0,
    show_progress: bool = True,
) -> EdgeSmoothingCVResult:
    """Choose `edge_smoothing` by graph-based cross-validation on held-out folds.

    The fold partition is built once from shared graph topology and observed
    sample density, then each candidate smoothing value is scored by species-
    balanced validation MSE across those held-out folds. The training defaults
    are intentionally lighter than `train_model(...)` so the search returns a
    coarse but useful smoothing estimate quickly.
    """
    if not smoothing_values:
        raise ValueError("smoothing_values is empty.")
    smoothing_values = tuple(float(value) for value in smoothing_values)
    for value in smoothing_values:
        if not (0.0 <= float(value) <= 1.0):
            raise ValueError("All smoothing_values must lie in [0, 1].")

    fold_ids, fold_summaries = build_graph_cv_folds(
        species_graphs,
        n_folds=n_folds,
        project_to=project_to,
        coord_order=coord_order,
        coords_crs=coords_crs,
        min_train_pairs=min_train_pairs,
        min_val_pairs=min_val_pairs,
        min_train_nodes=min_train_nodes,
        min_val_nodes=min_val_nodes,
        min_supported_species=min_supported_species,
    )

    candidate_results: list[EdgeSmoothingCVCandidate] = []
    active_folds = [summary.fold_id for summary in fold_summaries]
    summary_by_fold = {summary.fold_id: summary for summary in fold_summaries}
    usable_folds = []
    for fold_id in active_folds:
        fold_summary = summary_by_fold[fold_id]
        if any(stats.used_for_validation for stats in fold_summary.species_stats.values()):
            usable_folds.append(fold_id)

    if len(usable_folds) < 2:
        raise ValueError("Need at least two folds with usable validation species.")

    total_steps = len(smoothing_values) * len(usable_folds)
    completed_steps = 0

    if show_progress:
        _print_progress_line(
            f"choose_edge_smoothing_cv: evaluating {len(smoothing_values)} smoothing values "
            f"across {len(usable_folds)} folds ({total_steps} train/eval runs)"
        )

    for candidate_idx, smoothing in enumerate(smoothing_values, start=1):
        fold_losses: list[float] = []
        species_counts: list[int] = []

        for fold_idx, fold_id in enumerate(usable_folds, start=1):
            fold_summary = summary_by_fold[fold_id]
            fold_graphs: list[SpeciesGraph] = []
            for graph in species_graphs:
                train_mask, val_mask = _fold_masks(graph, fold_ids, fold_id)
                stats = fold_summary.species_stats[graph.name]
                if not stats.used_for_training:
                    continue

                graph_copy = _clone_graph_for_fold(graph, train_mask, val_mask)
                if graph_copy is None:
                    continue
                if not stats.used_for_validation:
                    graph_copy.val_pair_i = None
                    graph_copy.val_pair_j = None
                    graph_copy.val_pair_dist = None
                fold_graphs.append(graph_copy)

            if not fold_graphs:
                continue
            if not any(graph.val_pair_i is not None for graph in fold_graphs):
                continue

            if show_progress:
                scored_species = sum(graph.val_pair_i is not None for graph in fold_graphs)
                _print_progress_line(
                    f"candidate {candidate_idx}/{len(smoothing_values)} "
                    f"edge_smoothing={smoothing:.4f} "
                    f"fold {fold_idx}/{len(usable_folds)} "
                    f"(graph fold {fold_id}, {scored_species} scored species) "
                    f"step {completed_steps + 1}/{total_steps}"
                )

            model = train_model(
                fold_graphs,
                hidden_dim=hidden_dim,
                lr=lr,
                epochs=epochs,
                l2_shared=l2_shared,
                l2_species=l2_species,
                edge_smoothing=float(smoothing),
                log_every=log_every,
                val_fraction=0.0,
                patience=patience,
                min_delta=min_delta,
                restore_best=restore_best,
                seed=seed + int(fold_id),
            )
            fold_loss, species_count = _evaluate_fold_model(model, fold_graphs)
            fold_losses.append(fold_loss)
            species_counts.append(species_count)
            completed_steps += 1

        if not fold_losses:
            raise ValueError(
                f"Candidate edge_smoothing={smoothing:.6f} produced no usable validation folds."
            )

        candidate = EdgeSmoothingCVCandidate(
            edge_smoothing=smoothing,
            mean_validation_loss=float(np.mean(fold_losses)),
            std_validation_loss=float(np.std(fold_losses)),
            fold_losses=np.asarray(fold_losses, dtype=np.float64),
            species_counts=np.asarray(species_counts, dtype=np.int64),
        )
        candidate_results.append(candidate)

    best = min(candidate_results, key=lambda result: result.mean_validation_loss)
    if show_progress:
        _print_progress_line(
            f"choose_edge_smoothing_cv: best edge_smoothing={best.edge_smoothing:.4f} "
            f"(mean_validation_loss={best.mean_validation_loss:.6f})"
        )
        _finish_progress_line()
        print("edge_smoothing CV summary:", flush=True)
        for candidate in candidate_results:
            print(
                f"  edge_smoothing={candidate.edge_smoothing:.4f} "
                f"mean_validation_loss={candidate.mean_validation_loss:.6f} "
                f"std={candidate.std_validation_loss:.6f}",
                flush=True,
            )
    return EdgeSmoothingCVResult(
        best_edge_smoothing=best.edge_smoothing,
        candidates=candidate_results,
        fold_assignments=fold_ids,
        fold_summaries=fold_summaries,
    )
