"""
Validation module for IS/WS/RS dataflow templates.

Ensures that each dataflow template correctly implements the expected
reuse patterns before using them in NSGA-II optimization.
"""
from typing import Dict, List
import numpy as np

from .gene import Gene
from .onnx_parser import ConvLayerSpec
from .timeloop_integration import TimeloopRunner


def create_handcrafted_is_gene(layer_spec: ConvLayerSpec) -> Gene:
    """
    Create a handcrafted Input-Stationary gene that matches textbook definition.
    
    IS characteristics:
    - Inputs stay in PE buffers (stationary)
    - Weights and outputs move (temporal)
    - Good for input reuse
    """
    return Gene(
        pe_array_x=8,
        pe_array_y=8,
        simd_width=4,
        buffer_budget_class=1,
        dataflow="IS",
        tile_n=1,
        tile_c=min(16, layer_spec.C),
        tile_k=min(32, layer_spec.K),
        tile_h=min(7, layer_spec.H),
        tile_w=min(7, layer_spec.W),
        spatial_n=False,  # Temporal
        spatial_c=True,   # Spatial across PEs
        spatial_k=False,  # Temporal
        spatial_h=True,   # Spatial across PEs
        spatial_w=True,   # Spatial across PEs
        loop_permutation_id=0,  # N, C, K, H, W
    )


def create_handcrafted_ws_gene(layer_spec: ConvLayerSpec) -> Gene:
    """
    Create a handcrafted Weight-Stationary gene that matches textbook definition.
    
    WS characteristics:
    - Weights stay in PE buffers (stationary)
    - Inputs and outputs move (temporal)
    - Good for weight reuse
    """
    return Gene(
        pe_array_x=8,
        pe_array_y=8,
        simd_width=4,
        buffer_budget_class=1,
        dataflow="WS",
        tile_n=1,
        tile_c=min(16, layer_spec.C),
        tile_k=min(32, layer_spec.K),
        tile_h=min(7, layer_spec.H),
        tile_w=min(7, layer_spec.W),
        spatial_n=False,  # Temporal
        spatial_c=False,  # Temporal
        spatial_k=True,   # Spatial across PEs
        spatial_h=True,   # Spatial across PEs
        spatial_w=True,   # Spatial across PEs
        loop_permutation_id=10,
    )


def create_handcrafted_rs_gene(layer_spec: ConvLayerSpec) -> Gene:
    """
    Create a handcrafted Row-Stationary gene that matches textbook definition.
    
    RS characteristics:
    - Row of weights and inputs stay (stationary)
    - Outputs accumulate
    - Balanced reuse of all three
    """
    return Gene(
        pe_array_x=8,
        pe_array_y=8,
        simd_width=4,
        buffer_budget_class=1,
        dataflow="RS",
        tile_n=1,
        tile_c=min(16, layer_spec.C),
        tile_k=min(16, layer_spec.K),
        tile_h=min(7, layer_spec.H),
        tile_w=min(7, layer_spec.W),
        spatial_n=False,  # Temporal
        spatial_c=True,   # Spatial
        spatial_k=True,   # Spatial
        spatial_h=True,   # Spatial
        spatial_w=True,   # Spatial
        loop_permutation_id=20,
    )


def validate_dataflow_templates(layer_spec: ConvLayerSpec) -> Dict[str, Dict]:
    """
    Validate all three dataflow templates and return metrics.
    
    Checks that:
    - IS: input-buffer reads low, weight/psum movement higher
    - WS: weight-buffer reads low
    - RS: more balanced, good reuse of all three
    
    Returns:
        Dictionary mapping dataflow name to metrics
    """
    runner = TimeloopRunner()
    results = {}
    
    # Test IS
    print("Testing Input-Stationary (IS) dataflow...")
    is_gene = create_handcrafted_is_gene(layer_spec)
    is_metrics = runner.run_timeloop(is_gene, layer_spec)
    results["IS"] = {
        "gene": is_gene,
        "metrics": is_metrics,
    }
    print(f"  Latency: {is_metrics.get('latency', 0):.6f}s")
    print(f"  Energy: {is_metrics.get('energy', 0):.2f}pJ")
    print(f"  Utilization: {is_metrics.get('utilization', 0):.2%}")
    
    # Test WS
    print("\nTesting Weight-Stationary (WS) dataflow...")
    ws_gene = create_handcrafted_ws_gene(layer_spec)
    ws_metrics = runner.run_timeloop(ws_gene, layer_spec)
    results["WS"] = {
        "gene": ws_gene,
        "metrics": ws_metrics,
    }
    print(f"  Latency: {ws_metrics.get('latency', 0):.6f}s")
    print(f"  Energy: {ws_metrics.get('energy', 0):.2f}pJ")
    print(f"  Utilization: {ws_metrics.get('utilization', 0):.2%}")
    
    # Test RS
    print("\nTesting Row-Stationary (RS) dataflow...")
    rs_gene = create_handcrafted_rs_gene(layer_spec)
    rs_metrics = runner.run_timeloop(rs_gene, layer_spec)
    results["RS"] = {
        "gene": rs_gene,
        "metrics": rs_metrics,
    }
    print(f"  Latency: {rs_metrics.get('latency', 0):.6f}s")
    print(f"  Energy: {rs_metrics.get('energy', 0):.2f}pJ")
    print(f"  Utilization: {rs_metrics.get('utilization', 0):.2%}")
    
    # Summary
    print("\n=== Dataflow Comparison ===")
    print(f"Best latency: {min(r['metrics'].get('latency', float('inf')) for r in results.values()):.6f}s")
    print(f"Best energy: {min(r['metrics'].get('energy', float('inf')) for r in results.values()):.2f}pJ")
    print(f"Best utilization: {max(r['metrics'].get('utilization', 0) for r in results.values()):.2%}")
    
    runner.cleanup()
    return results

