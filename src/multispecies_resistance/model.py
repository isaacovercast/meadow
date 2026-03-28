from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


torch.set_default_dtype(torch.float64)


class EdgeMLP(nn.Module):
    """Small MLP that maps edge features to a scalar edge logit."""

    def __init__(self, in_dim: int, hidden_dim: int = 32):
        """Initialize a three-layer ReLU MLP for edge scoring.

        Parameters
        ----------
        in_dim : int
            Number of input edge features.
        hidden_dim : int, optional
            Width of hidden layers.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute scalar edge logits for a batch of edge features.

        Parameters
        ----------
        x : torch.Tensor
            `E x F` edge-feature tensor.

        Returns
        -------
        torch.Tensor
            `E x 1` edge logits.
        """
        return self.net(x)


class MultiSpeciesResistanceModel(nn.Module):
    """Shared-plus-species neural resistance model over graph edges."""

    def __init__(self, num_species: int, edge_feat_dim: int, hidden_dim: int = 32):
        """Initialize shared/species edge networks and calibration parameters.

        Parameters
        ----------
        num_species : int
            Number of species modeled jointly.
        edge_feat_dim : int
            Number of edge features per edge.
        hidden_dim : int, optional
            Hidden width for all edge MLPs.
        """
        super().__init__()
        self.shared = EdgeMLP(edge_feat_dim, hidden_dim)
        self.species = nn.ModuleList(
            [EdgeMLP(edge_feat_dim, hidden_dim) for _ in range(num_species)]
        )
        self.alpha = nn.Parameter(torch.zeros(num_species))
        self.beta = nn.Parameter(torch.ones(num_species))

    def edge_logits(self, species_idx: int, edge_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute shared and species-specific edge logits.

        Parameters
        ----------
        species_idx : int
            Index of species head to apply.
        edge_feat : torch.Tensor
            `E x F` edge-feature tensor.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Shared logits and species logits, both length `E`.
        """
        shared = self.shared(edge_feat).squeeze(-1)
        species = self.species[species_idx](edge_feat).squeeze(-1)
        return shared, species

    def edge_resistance(self, species_idx: int, edge_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert edge logits into positive resistances for one species.

        Parameters
        ----------
        species_idx : int
            Index of species head to apply.
        edge_feat : torch.Tensor
            `E x F` edge-feature tensor.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Edge resistances, shared logits, and species logits.
        """
        shared, species = self.edge_logits(species_idx, edge_feat)
        resistance = torch.nn.functional.softplus(shared + species) + 1e-4
        return resistance, shared, species

    def resistance_matrix(
        self,
        species_idx: int,
        edge_index: torch.Tensor,
        edge_feat: torch.Tensor,
        num_nodes: int,
        edge_support_weight: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build a graph Laplacian and compute effective resistance between nodes.

        Parameters
        ----------
        species_idx : int
            Index of species head to apply.
        edge_index : torch.Tensor
            `E x 2` edge list indexing node rows.
        edge_feat : torch.Tensor
            `E x F` edge-feature tensor.
        num_nodes : int
            Number of graph nodes.
        edge_support_weight : torch.Tensor | None, optional
            Optional length-`E` attenuation factor applied to edge conductance.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Effective-resistance matrix `R` (`N x N`), shared logits, and species logits.
        """
        resistance, shared, species = self.edge_resistance(species_idx, edge_feat)
        conductance = 1.0 / resistance
        if edge_support_weight is not None:
            conductance = conductance * edge_support_weight.to(
                device=edge_feat.device,
                dtype=edge_feat.dtype,
            )

        i = edge_index[:, 0]
        j = edge_index[:, 1]

        L = torch.zeros((num_nodes, num_nodes), dtype=edge_feat.dtype, device=edge_feat.device)
        L.index_put_((i, i), conductance, accumulate=True)
        L.index_put_((j, j), conductance, accumulate=True)
        L.index_put_((i, j), -conductance, accumulate=True)
        L.index_put_((j, i), -conductance, accumulate=True)

        L_pinv = torch.linalg.pinv(L)
        diag = torch.diag(L_pinv)
        R = diag[:, None] + diag[None, :] - 2.0 * L_pinv
        return R, shared, species
