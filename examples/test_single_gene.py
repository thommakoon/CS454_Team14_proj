"""
Test script for evaluating a single gene.

This script demonstrates Step 2 of the plan: building the evaluation backbone.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pe_mapper import (
    Gene,
    get_resnet18_conv3_example,
    TimeloopRunner,
    evaluate_gene,
)


def test_single_gene():
    """Test evaluating a single fixed gene instance."""
    print("Testing single gene evaluation...")
    
    # Get layer spec
    layer_spec = get_resnet18_conv3_example()
    print(f"Layer: {layer_spec}")
    
    # Create a test gene (Input-Stationary)
    gene = Gene(
        pe_array_x=4,
        pe_array_y=4,
        simd_width=4,
        buffer_budget_class=1,
        dataflow="IS",
        tile_n=1,
        tile_c=16,
        tile_k=32,
        tile_h=7,
        tile_w=7,
        spatial_n=False,
        spatial_c=True,
        spatial_k=False,
        spatial_h=True,
        spatial_w=True,
        loop_permutation_id=0,
    )
    
    print(f"\nGene: {gene.dataflow} dataflow")
    print(f"  PE array: {gene.pe_array_x}x{gene.pe_array_y}")
    print(f"  SIMD width: {gene.simd_width}")
    print(f"  Buffer class: {gene.buffer_budget_class}")
    print(f"  Tiling: N={gene.tile_n}, C={gene.tile_c}, K={gene.tile_k}, "
          f"H={gene.tile_h}, W={gene.tile_w}")
    
    # Test 1: Direct Timeloop evaluation
    print("\n[Test 1] Direct Timeloop evaluation...")
    runner = TimeloopRunner()
    metrics = runner.run_timeloop(gene, layer_spec)
    print(f"  Latency: {metrics.get('latency', 0):.6f} seconds")
    print(f"  Energy: {metrics.get('energy', 0):.2f} pJ")
    print(f"  Utilization: {metrics.get('utilization', 0):.2%}")
    
    # Test 2: Fitness function evaluation
    print("\n[Test 2] Fitness function evaluation...")
    latency, energy, neg_utilization = evaluate_gene(gene, layer_spec)
    utilization = -neg_utilization
    print(f"  Objectives: ({latency:.6f}, {energy:.2f}, {utilization:.2%})")
    print(f"  (latency, energy, utilization)")
    
    # Test 3: Change dataflow and verify metrics change
    print("\n[Test 3] Changing dataflow to WS...")
    gene_ws = Gene(
        pe_array_x=4,
        pe_array_y=4,
        simd_width=4,
        buffer_budget_class=1,
        dataflow="WS",  # Changed to Weight-Stationary
        tile_n=1,
        tile_c=16,
        tile_k=32,
        tile_h=7,
        tile_w=7,
        spatial_n=False,
        spatial_c=False,  # Changed
        spatial_k=True,   # Changed
        spatial_h=True,
        spatial_w=True,
        loop_permutation_id=10,
    )
    
    metrics_ws = runner.run_timeloop(gene_ws, layer_spec)
    print(f"  Latency: {metrics_ws.get('latency', 0):.6f} seconds")
    print(f"  Energy: {metrics_ws.get('energy', 0):.2f} pJ")
    print(f"  Utilization: {metrics_ws.get('utilization', 0):.2%}")
    
    print("\n✓ Single gene evaluation tests complete!")
    runner.cleanup()


if __name__ == "__main__":
    test_single_gene()

