# Overview

`multispecies-resistance` is a prototype for jointly modeling landscape resistance across multiple species from sample-level genotype data.

## End-to-End Workflow

1. **Load sample-level data:** Use `load_pedic_species(...)` or construct `SpeciesData(name, genotypes, sample_coords)` directly.
2. **Build graph training datasets:** Use `build_species_graphs(...)` to define nodes/edges and sample node-level environmental covariates.
3. **Train resistance model:** Fit with `train_model(...)`.
4. **Visualize outputs:** Use `SpeciesGraph.plot(...)` for direct per-species graph mapping, or the helpers in `viz.py` for multi-species overlays.

## Module Pages

- [Data Utilities](data.md)
- [Graph Utilities](graph.md)
- [Training Utilities](train.md)
- [Model](model.md)
- [Raster Utilities](raster.md)
- [Climate Utilities](climate.md)
- [I/O Utilities](io.md)
- [Visualization Utilities](viz.md)
