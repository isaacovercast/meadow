# Overview

`multispecies-resistance` is a compact prototype for jointly modeling landscape resistance across multiple species. The package combines graph-based effective resistance with neural edge models so you can estimate shared barriers/corridors and species-specific deviations from SNP-derived genetic distances.

## End-to-End Workflow

1. **Load data**
- Use `SpeciesData` directly or `load_pedic_species(...)` for PEDIC-style inputs.

2. **Create graph inputs**
- Build per-species graphs (`"delaunay"`, `"knn"`) or a shared dense mesh/global graph with `build_species_graphs(...)`.

3. **Train model**
- Fit shared/species resistance using `train_model(...)` with optional validation splits and early stopping.

4. **Visualize outputs**
- Plot species-specific edge resistance, multi-species overlays, shared resistance surfaces, and matrix summaries with functions in `viz`.

## Module Pages

- [Data Utilities](data.md)
- [Graph Utilities](graph.md)
- [Training Utilities](train.md)
- [Model](model.md)
- [Raster Utilities](raster.md)
- [Climate Utilities](climate.md)
- [I/O Utilities](io.md)
- [Visualization Utilities](viz.md)
