"""
Fitness evaluation function for gene optimization.

Wraps gene → Timeloop → metrics pipeline for use with NSGA-II.
"""
from typing import Dict, Tuple, Optional
import numpy as np

from .gene import Gene
from .onnx_parser import ConvLayerSpec
from .timeloop_integration import TimeloopRunner


def evaluate_gene(
    gene: Gene,
    layer_spec: ConvLayerSpec,
    timeloop_runner: Optional[TimeloopRunner] = None
) -> Tuple[float, float, float]:
    """
    Evaluate a single gene and return fitness objectives.
    
    This function is designed to be stateless and deterministic so it can
    be plugged directly into NSGA-II.
    
    Args:
        gene: The gene to evaluate
        layer_spec: The convolutional layer specification
        timeloop_runner: Optional TimeloopRunner instance (creates new one if None)
    
    Returns:
        Tuple of (latency, energy, -utilization) as objectives.
        Note: utilization is negated because NSGA-II minimizes objectives,
        and we want to maximize utilization.
    """
    if timeloop_runner is None:
        timeloop_runner = TimeloopRunner()
    
    # Run Timeloop
    metrics = timeloop_runner.run_timeloop(gene, layer_spec)
    
    # Extract objectives
    latency = metrics.get("latency", float('inf'))
    energy = metrics.get("energy", float('inf'))
    utilization = metrics.get("utilization", 0.0)
    
    # Return objectives: (latency, energy, -utilization)
    # We negate utilization because we want to maximize it (minimize -utilization)
    return (latency, energy, -utilization)


def validate_gene_constraints(gene: Gene, layer_spec: ConvLayerSpec) -> bool:
    """
    Validate that a gene's tiling factors are feasible for the given layer.
    
    Args:
        gene: The gene to validate
        layer_spec: The layer specification
    
    Returns:
        True if gene is valid, False otherwise
    """
    # Check that tiling factors don't exceed layer dimensions
    if gene.tile_n > layer_spec.N:
        return False
    if gene.tile_c > layer_spec.C:
        return False
    if gene.tile_k > layer_spec.K:
        return False
    if gene.tile_h > layer_spec.H:
        return False
    if gene.tile_w > layer_spec.W:
        return False
    
    # Check that spatial mapping doesn't exceed PE array capacity
    spatial_dims = [
        (gene.spatial_n, gene.tile_n),
        (gene.spatial_c, gene.tile_c),
        (gene.spatial_k, gene.tile_k),
        (gene.spatial_h, gene.tile_h),
        (gene.spatial_w, gene.tile_w),
    ]
    
    spatial_products = [
        tile for is_spatial, tile in spatial_dims if is_spatial
    ]
    
    if spatial_products:
        total_spatial = np.prod(spatial_products)
        pe_capacity = gene.pe_array_x * gene.pe_array_y
        if total_spatial > pe_capacity:
            return False
    
    return True


def repair_gene(gene: Gene, layer_spec: ConvLayerSpec) -> Gene:
    """
    Repair a gene to satisfy constraints.
    
    Clips tiling factors to valid ranges and adjusts spatial mapping.
    
    Args:
        gene: The gene to repair
        layer_spec: The layer specification
    
    Returns:
        Repaired gene
    """
    # Clip tiling factors
    tile_n = min(gene.tile_n, layer_spec.N)
    tile_c = min(gene.tile_c, layer_spec.C)
    tile_k = min(gene.tile_k, layer_spec.K)
    tile_h = min(gene.tile_h, layer_spec.H)
    tile_w = min(gene.tile_w, layer_spec.W)
    
    # Create repaired gene
    repaired = Gene(
        pe_array_x=gene.pe_array_x,
        pe_array_y=gene.pe_array_y,
        simd_width=gene.simd_width,
        buffer_budget_class=gene.buffer_budget_class,
        dataflow=gene.dataflow,
        tile_n=tile_n,
        tile_c=tile_c,
        tile_k=tile_k,
        tile_h=tile_h,
        tile_w=tile_w,
        spatial_n=gene.spatial_n,
        spatial_c=gene.spatial_c,
        spatial_k=gene.spatial_k,
        spatial_h=gene.spatial_h,
        spatial_w=gene.spatial_w,
        loop_permutation_id=gene.loop_permutation_id,
    )
    
    return repaired

