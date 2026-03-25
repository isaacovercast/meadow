"""Minimal multi-species resistance model prototype."""

from .climate import download_climate_layers, sample_climate_for_sites
from .cv import choose_edge_smoothing_cv
from .data import SpeciesData, aggregate_site_genotypes, pairwise_site_distance
from .graph import (
    SpeciesGraph,
    build_delaunay_graph,
    build_dense_mesh_graph,
    build_geodesic_mesh_graph,
    edge_features,
    project_coords,
    standardize_features,
)
from .io import list_pedic_species, load_pedic_species
from .raster import (
    RasterStack,
    open_raster_stack,
    resolve_raster_paths,
    sample_raster_at_points,
    sample_rasters_for_sites,
)
from .train import build_species_graphs, choose_mesh_spacing_km, train_model
from .viz import (
    plot_multi_edge_resistance,
    plot_resistance_matrix,
    plot_shared_resistance,
    plot_sites,
    plot_species_resistance,
)
