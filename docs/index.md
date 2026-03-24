# Multi-Species Resistance Docs

This documentation covers the sample-level workflow for loading species data, building graph training datasets (including node assignment and environmental extraction), training resistance models, and plotting outputs.

## Documentation Map

| Module | Purpose |
| --- | --- |
| `data` | Sample-level species container and genotype aggregation |
| `graph` | Spatial graph construction and edge feature utilities |
| `train` | Graph dataset construction and model optimization |
| `model` | Shared/species neural resistance architecture |
| `raster` | GeoTIFF path resolution and point sampling |
| `climate` | WorldClim/BioClim download and caching helpers |
| `io` | PEDIC sample-level loading |
| `viz` | Static and interactive resistance visualization |

## Quick Start

```python
from multispecies_resistance.io import load_pedic_species
from multispecies_resistance.climate import download_climate_layers
from multispecies_resistance.train import build_species_graphs, train_model

species_list = load_pedic_species("/path/to/pedic")
raster_paths = download_climate_layers(source="bioclim", variables=["bio1", "bio12"])

graphs, stats = build_species_graphs(
    species_list,
    raster_paths=raster_paths,
)
model = train_model(graphs)

# quick map for the first species graph
ax, gdf_edges = graphs[0].plot(edge_feature_idx=0, basemap=True)
```

## Local Documentation Development

From `/Users/isaac/src/meems/meems-from-scratch`:

```bash
mkdocs serve
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000).
