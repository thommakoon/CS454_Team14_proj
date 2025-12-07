"""
Main entry point for CS454 Team 14 Project.

Demonstrates the complete pipeline:
1. Gene definition and examples
2. ONNX layer extraction
3. Timeloop evaluation
4. Dataflow validation
5. NSGA-II optimization
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pe_mapper import (
    Gene,
    create_example_genes,
    ConvLayerSpec,
    get_resnet18_conv3_example,
    TimeloopRunner,
    evaluate_gene,
    validate_gene_constraints,
    optimize_genes,
)
from pe_mapper.dataflow_validation import validate_dataflow_templates


def main():
    print("=" * 60)
    print("CS454 Team 14: PE-focused CNN Accelerator Mapping Optimization")
    print("=" * 60)
    
    # Step 1: Create example genes
    print("\n[Step 1] Creating example genes...")
    example_genes = create_example_genes()
    for i, gene in enumerate(example_genes, 1):
        print(f"  Gene {i}: {gene.dataflow} dataflow, "
              f"PE array {gene.pe_array_x}x{gene.pe_array_y}, "
              f"SIMD={gene.simd_width}")
    
    # Step 2: Get layer specification
    print("\n[Step 2] Loading layer specification...")
    layer_spec = get_resnet18_conv3_example()
    print(f"  Layer: {layer_spec.name}")
    print(f"  Input shape: {layer_spec.input_shape}")
    print(f"  Output shape: {layer_spec.output_shape}")
    print(f"  Kernel size: {layer_spec.kernel_size}")
    
    # Step 3: Evaluate a single gene
    print("\n[Step 3] Evaluating a single gene with Timeloop...")
    test_gene = example_genes[0]
    runner = TimeloopRunner()
    metrics = runner.run_timeloop(test_gene, layer_spec)
    print(f"  Latency: {metrics.get('latency', 0):.6f} seconds")
    print(f"  Energy: {metrics.get('energy', 0):.2f} pJ")
    print(f"  Utilization: {metrics.get('utilization', 0):.2%}")
    print(f"  Area: {metrics.get('area', 0):.2f} mm²")
    
    # Step 4: Validate dataflow templates
    print("\n[Step 4] Validating IS/WS/RS dataflow templates...")
    try:
        dataflow_results = validate_dataflow_templates(layer_spec)
        print("  ✓ All dataflow templates validated")
    except Exception as e:
        print(f"  ⚠ Dataflow validation encountered an issue: {e}")
        print("  (This is expected if Timeloop is not installed)")
    
    # Step 5: Test fitness function
    print("\n[Step 5] Testing fitness evaluation function...")
    try:
        latency, energy, neg_utilization = evaluate_gene(test_gene, layer_spec)
        utilization = -neg_utilization
        print(f"  Objectives: latency={latency:.6f}, energy={energy:.2f}, utilization={utilization:.2%}")
    except Exception as e:
        print(f"  ⚠ Fitness evaluation encountered an issue: {e}")
    
    # Step 6: NSGA-II optimization (optional, can be slow)
    print("\n[Step 6] NSGA-II Optimization (optional)...")
    # print("  To run optimization, uncomment the following code:")
    # print("  pareto_genes, objectives = optimize_genes(")
    # print("      layer_spec,")
    # print("      population_size=20,")
    # print("      n_generations=10,")
    # print("  )")
    # print("  print(f'Found {len(pareto_genes)} Pareto-optimal solutions')")
    
    # Uncomment to actually run optimization:
    print("\n  Running NSGA-II optimization...")
    pareto_genes, objectives = optimize_genes(
        layer_spec,
        population_size=10,  # Small for quick test
        n_generations=5,
        seed=42,
    )
    print(f"  Found {len(pareto_genes)} Pareto-optimal solutions")
    for i, (gene, obj) in enumerate(zip(pareto_genes[:5], objectives[:5]), 1):
        print(f"    Solution {i}: {gene.dataflow}, "
              f"latency={obj[0]:.6f}, energy={obj[1]:.2f}, util={-obj[2]:.2%}")
    
    print("\n" + "=" * 60)
    print("Pipeline demonstration complete!")
    print("=" * 60)
    
    runner.cleanup()


if __name__ == "__main__":
    main()
