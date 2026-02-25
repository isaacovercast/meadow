# Multi-Species Resistance Docs

This documentation covers the full workflow for loading multi-species genotype data, building spatial graphs, training shared/species resistance models, and visualizing inferred resistance structure.

## Documentation Map

| Module | Purpose |
| --- | --- |
| `data` | Sample-to-site aggregation and pseudo-site generation |
| `graph` | Spatial graph construction and edge feature generation |
| `train` | Graph dataset preparation and model optimization |
| `model` | Shared/species neural resistance architecture |
| `raster` | GeoTIFF path resolution and point sampling |
| `climate` | WorldClim/BioClim download, caching, and sampling |
| `io` | PEDIC FEEMS file loading into `SpeciesData` |
| `viz` | Static and interactive resistance visualizations |

## Quick Start

```python
from multispecies_resistance.io import load_pedic_species
from multispecies_resistance.train import build_species_graphs, train_model

species_list, env_names = load_pedic_species("/path/to/pedic")
graphs, stats = build_species_graphs(species_list, graph_type="dense_mesh")
model = train_model(graphs)
```

## Local Documentation Development

From `/Users/isaac/src/meems/meems-from-scratch`:

```bash
mkdocs serve
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000).

Build static docs with:

```bash
mkdocs build
```
