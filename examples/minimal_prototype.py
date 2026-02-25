import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from multispecies_resistance.data import (
    SpeciesData,
    build_pseudosites,
    aggregate_site_genotypes,
    pairwise_site_distance,
)
from multispecies_resistance.graph import (
    build_delaunay_graph,
    edge_features,
    standardize_features,
)
from multispecies_resistance.train import SpeciesGraphData, train_model


torch.set_default_dtype(torch.float64)


def simulate_species(
    name: str,
    num_sites: int = 30,
    num_snps: int = 2000,
    num_env: int = 3,
    samples_per_site: tuple[int, int] = (5, 10),
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    # Lat/long roughly in the continental US
    coords = np.column_stack(
        [
            rng.uniform(34.0, 48.0, size=num_sites),
            rng.uniform(-120.0, -100.0, size=num_sites),
        ]
    )

    env = rng.normal(0.0, 1.0, size=(num_sites, num_env))

    feats = np.concatenate([coords, env], axis=1)
    weights = rng.normal(0.0, 0.5, size=(feats.shape[1], num_snps))
    logits = feats @ weights + rng.normal(0.0, 0.5, size=(num_sites, num_snps))
    freq = 1.0 / (1.0 + np.exp(-logits))
    freq = np.clip(freq, 0.01, 0.99)

    genotypes = []
    sample_coords = []
    sample_env = []
    for s in range(num_sites):
        n = int(rng.integers(samples_per_site[0], samples_per_site[1] + 1))
        g = rng.binomial(2, freq[s], size=(n, num_snps))
        genotypes.append(g)
        sample_coords.append(np.repeat(coords[s][None, :], n, axis=0))
        sample_env.append(np.repeat(env[s][None, :], n, axis=0))

    genotypes = np.vstack(genotypes)
    sample_coords = np.vstack(sample_coords)
    sample_env = np.vstack(sample_env)

    return genotypes, sample_coords, sample_env


def build_species_graph(species: SpeciesData) -> tuple[SpeciesGraphData, np.ndarray]:
    site_genotypes, _ = aggregate_site_genotypes(
        species.genotypes, species.sample_sites, num_sites=species.num_sites()
    )
    dist = pairwise_site_distance(site_genotypes)

    edges = build_delaunay_graph(species.site_coords)
    feats = edge_features(species.site_coords, species.site_env, edges)
    pair_i, pair_j = np.triu_indices(dist.shape[0], k=1)
    pair_dist = dist[pair_i, pair_j]

    graph = SpeciesGraphData(
        name=species.name,
        edge_index=edges,
        edge_features=feats,
        node_coords=species.site_coords,
        pair_i=pair_i,
        pair_j=pair_j,
        pair_dist=pair_dist,
        num_nodes=dist.shape[0],
    )
    return graph, feats


def main():
    species_list = []
    for name, seed in [("species_a", 1), ("species_b", 2), ("species_c", 3)]:
        genotypes, sample_coords, sample_env = simulate_species(name, seed=seed)
        site_coords, sample_sites, _, _, site_env = build_pseudosites(
            sample_coords,
            genotypes,
            spacing_km=80.0,
            sample_env=sample_env,
        )
        species_list.append(
            SpeciesData(
                name=name,
                genotypes=genotypes,
                sample_sites=sample_sites,
                site_coords=site_coords,
                site_env=site_env,
            )
        )

    graphs = []
    all_feats = []

    for sp in species_list:
        graph, feats = build_species_graph(sp)
        graphs.append(graph)
        all_feats.append(feats)

    all_feats = np.vstack(all_feats)
    _, mean, std = standardize_features(all_feats)

    # apply global standardization
    for g in graphs:
        g.edge_features = (g.edge_features - mean) / std

    model = train_model(graphs, hidden_dim=32, lr=1e-2, epochs=200, log_every=25)

    # Example: report average learned resistance per species
    for s, g in enumerate(graphs):
        edge_feat = torch.from_numpy(g.edge_features)
        edge_index = torch.from_numpy(g.edge_index)
        R, _, _ = model.resistance_matrix(s, edge_index, edge_feat, g.num_nodes)
        print(f"{g.name}: mean effective resistance {R.mean().item():.4f}")


if __name__ == "__main__":
    main()
