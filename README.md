# Multi-species resistance prototype

Minimal Python prototype for learning shared and species-specific migration resistance on spatial graphs from SNP data.

## What this does
- Keeps species data at the observed sample level (`name`, `genotypes`, `sample_coords`)
- Builds graph nodes/edges inside `build_species_graphs(...)` using either a shared dense mesh or a provided input graph
- Samples environmental rasters at the exact graph nodes used for training
- Learns shared and species-specific edge resistances and fits genetic distances via effective resistance

## Install (conda)

```bash
conda env create -f environment.yml
conda activate multispecies-resistance
```

## Run the synthetic example

```bash
python examples/minimal_prototype.py
```

## Core data model

`SpeciesData` now contains only sample-level information:
- `name`: species name
- `genotypes`: `N x M` genotype matrix
- `sample_coords`: `N x 2` coordinates (`lat, lon`)

Graph-node assignment (and any sample-to-node remapping) is performed in `build_species_graphs(...)`.

## Building graphs

```python
from multispecies_resistance.train import build_species_graphs

graphs, stats = build_species_graphs(
    species_list,
    mesh_grid_type="triangular",
    standardize=True,
)

# SpeciesGraph object with node/edge geometry + sample coordinates
ax, gdf_edges = graphs[0].plot(edge_feature_idx=0, basemap=True)
```

`build_species_graphs(...)` returns `SpeciesGraph` objects that include:
- graph geometry (`node_coords`, `edge_index`)
- edge covariates (`edge_features`)
- neighboring-edge pairs for optional smoothing penalties (`edge_nbr_i`, `edge_nbr_j`)
- original sample locations (`sample_coords`)
- pairwise training targets (`pair_i`, `pair_j`, `pair_dist`)

Training can optionally smooth neighboring predicted edge logits:

```python
from multispecies_resistance.train import train_model

model = train_model(graphs, edge_smoothing=0.5)
```

## GeoTIFF environmental sampling (inside graph build)

Pass raster inputs directly to `build_species_graphs(...)`:

```python
graphs, stats = build_species_graphs(
    species_list,
    raster_paths=["/path/to/env1.tif", "/path/to/env2.tif"],
    raster_fill_method="nearest",
    standardize=True,
)
```

You can also provide a raster directory:

```python
graphs, stats = build_species_graphs(
    species_list,
    raster_root="/path/to/rasters",
    raster_pattern="*.tif",
    raster_recursive=True,
)
```

## WorldClim / BioClim helper

```python
from multispecies_resistance.climate import download_climate_layers

raster_paths = download_climate_layers(
    source="bioclim",
    variables=["bio1", "bio12"],
    resolution="2.5m",
)

graphs, stats = build_species_graphs(
    species_list,
    raster_paths=raster_paths,
)
```

Pass `input_graph="/path/to/graph.gml"` when you want to use a provided shared graph instead of the default dense mesh.
When `input_graph` is omitted and `mesh_spacing_km=None` (the default), the mesh spacing is chosen automatically from nearest-neighbor sample distances.

## PEDIC FEEMS loader

`load_pedic_species(...)` now only loads sample-level inputs:

```python
from multispecies_resistance.io import load_pedic_species

species_list = load_pedic_species(
    "/Users/isaac/src/meems/pedic_feems_files",
    mmap_mode="r",
)
```

Then build graphs (and optionally sample rasters) in `build_species_graphs(...)`.

## Notebooks
- `notebooks/pedic_example.ipynb`: PEDIC FEEMS workflow with climate rasters sampled in graph build
- `notebooks/example_geotiff_pseudosites.ipynb`: sample-level workflow using synthetic raster input

## Key files
- `src/multispecies_resistance/data.py`: sample-level species container and genotype aggregation
- `src/multispecies_resistance/graph.py`: graph construction and edge feature utilities
- `src/multispecies_resistance/train.py`: graph dataset construction + training loop
- `src/multispecies_resistance/model.py`: neural resistance model
- `src/multispecies_resistance/io.py`: PEDIC sample-level loader
