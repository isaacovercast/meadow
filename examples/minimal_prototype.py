import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from multispecies_resistance.data import SpeciesData
from multispecies_resistance.train import build_species_graphs, train_model


torch.set_default_dtype(torch.float64)


def simulate_species(
    name: str,
    num_sites: int = 30,
    num_snps: int = 2000,
    samples_per_site: tuple[int, int] = (5, 10),
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    # Lat/long roughly in the continental US
    site_coords = np.column_stack(
        [
            rng.uniform(34.0, 48.0, size=num_sites),
            rng.uniform(-120.0, -100.0, size=num_sites),
        ]
    )

    # Simulate latent site-level allele frequencies.
    latent = np.column_stack([site_coords, rng.normal(0.0, 1.0, size=(num_sites, 2))])
    weights = rng.normal(0.0, 0.5, size=(latent.shape[1], num_snps))
    logits = latent @ weights + rng.normal(0.0, 0.5, size=(num_sites, num_snps))
    freq = 1.0 / (1.0 + np.exp(-logits))
    freq = np.clip(freq, 0.01, 0.99)

    genotypes = []
    sample_coords = []
    for s in range(num_sites):
        n = int(rng.integers(samples_per_site[0], samples_per_site[1] + 1))
        g = rng.binomial(2, freq[s], size=(n, num_snps))
        genotypes.append(g)
        sample_coords.append(np.repeat(site_coords[s][None, :], n, axis=0))

    genotypes = np.vstack(genotypes)
    sample_coords = np.vstack(sample_coords)

    return genotypes, sample_coords


def main():
    species_list = []
    for name, seed in [("species_a", 1), ("species_b", 2), ("species_c", 3)]:
        genotypes, sample_coords = simulate_species(name, seed=seed)
        species_list.append(
            SpeciesData(
                name=name,
                genotypes=genotypes,
                sample_coords=sample_coords,
            )
        )

    graphs, _ = build_species_graphs(
        species_list,
        standardize=True,
    )

    model = train_model(graphs, hidden_dim=32, lr=1e-2, epochs=200, log_every=25)

    # Example: report average learned resistance per species
    for s, g in enumerate(graphs):
        edge_feat = torch.from_numpy(g.edge_features)
        edge_index = torch.from_numpy(g.edge_index)
        R, _, _ = model.resistance_matrix(s, edge_index, edge_feat, g.num_nodes)
        print(f"{g.name}: mean effective resistance {R.mean().item():.4f}")


if __name__ == "__main__":
    main()
