# Model

`multispecies_resistance.model` defines neural components that map edge features to resistance and then compute effective resistance between all node pairs via graph Laplacian algebra. The architecture combines a shared edge network with species-specific deviation networks.

![Model Architecture](model_architecture.svg)

## `EdgeMLP`
A small MLP that predicts one scalar edge logit from edge features.

### `EdgeMLP(in_dim, hidden_dim=32)`
Parameters:

- `in_dim`: number of edge-feature columns.
- `hidden_dim`: hidden width of the MLP.

### `forward(x)`
Parameters:

- `x`: `E x F` edge-feature tensor.

Returns:

- `logits`: `E x 1` edge logits.

## `MultiSpeciesResistanceModel`
Main multi-species model with shared and species-specific edge subnetworks.

### `MultiSpeciesResistanceModel(num_species, edge_feat_dim, hidden_dim=32)`
Parameters:

- `num_species`: number of species heads.
- `edge_feat_dim`: number of edge features.
- `hidden_dim`: hidden width for shared/species MLPs.

Learned parameters:
- `shared`: shared edge MLP.
- `species`: one edge MLP per species.
- `alpha`: species-specific intercepts for distance calibration.
- `beta`: species-specific slopes for distance calibration.

### `edge_logits(species_idx, edge_feat)`
Computes shared and species-specific edge logits.

Parameters:

- `species_idx`: species head index.
- `edge_feat`: `E x F` edge-feature tensor.

Returns:

- `shared_logits`: length-`E` shared logits.
- `species_logits`: length-`E` species logits.

### `edge_resistance(species_idx, edge_feat)`
Converts logits to strictly positive resistance values.

Parameters:

- `species_idx`: species head index.
- `edge_feat`: `E x F` edge-feature tensor.

Returns:

- `resistance`: length-`E` positive edge resistance.
- `shared_logits`: shared logits.
- `species_logits`: species logits.

### `resistance_matrix(species_idx, edge_index, edge_feat, num_nodes, edge_support_weight=None)`
Builds the graph Laplacian from conductances and computes effective resistance between all node pairs.

Parameters:

- `species_idx`: species head index.
- `edge_index`: `E x 2` edge list.
- `edge_feat`: `E x F` edge-feature tensor.
- `num_nodes`: number of nodes in the graph.
- `edge_support_weight`: optional length-`E` conductance attenuation vector.

Returns:

- `R`: `N x N` effective resistance matrix.
- `shared_logits`: shared logits.
- `species_logits`: species logits.
