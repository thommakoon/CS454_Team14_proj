"""
NSGA-II optimization script.

This script implements Step 5 of the plan: running NSGA-II optimization
to find Pareto-optimal gene configurations.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pe_mapper import (
    get_resnet18_conv3_example,
    optimize_genes,
    evaluate_gene,
)


def main():
    """Run NSGA-II optimization."""
    print("=" * 60)
    print("NSGA-II Optimization")
    print("=" * 60)
    
    # Get layer spec
    layer_spec = get_resnet18_conv3_example()
    print(f"\nLayer: {layer_spec.name}")
    print(f"  Input: {layer_spec.input_shape}")
    print(f"  Output: {layer_spec.output_shape}")
    
    # Run optimization
    print("\nRunning NSGA-II optimization...")
    print("  Population size: 20")
    print("  Generations: 10")
    print("  Objectives: (latency, energy, -utilization)")
    print("\nThis may take a while...\n")
    
    try:
        pareto_genes, objectives = optimize_genes(
            layer_spec,
            population_size=20,
            n_generations=10,
            seed=42,
        )
        
        print(f"\n✓ Optimization complete!")
        print(f"  Found {len(pareto_genes)} Pareto-optimal solutions")
        
        # Display top solutions
        print("\nTop 10 Pareto-optimal solutions:")
        print("-" * 60)
        print(f"{'#':<4} {'Dataflow':<10} {'PE Array':<12} {'Latency (s)':<15} "
              f"{'Energy (pJ)':<15} {'Utilization':<12}")
        print("-" * 60)
        
        for i, (gene, obj) in enumerate(zip(pareto_genes[:10], objectives[:10]), 1):
            latency = obj[0]
            energy = obj[1]
            utilization = -obj[2]  # Negate back
            pe_array = f"{gene.pe_array_x}x{gene.pe_array_y}"
            
            print(f"{i:<4} {gene.dataflow:<10} {pe_array:<12} "
                  f"{latency:<15.6f} {energy:<15.2f} {utilization:<12.2%}")
        
        # Analyze dataflow distribution
        print("\n" + "-" * 60)
        print("Dataflow distribution in Pareto front:")
        dataflow_counts = {}
        for gene in pareto_genes:
            dataflow_counts[gene.dataflow] = dataflow_counts.get(gene.dataflow, 0) + 1
        
        for dataflow, count in sorted(dataflow_counts.items()):
            print(f"  {dataflow}: {count} solutions ({count/len(pareto_genes)*100:.1f}%)")
        
        print("\n" + "=" * 60)
        print("Optimization complete!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n⚠ Optimization encountered an error: {e}")
        print("  This may be due to:")
        print("  - pymoo not installed (pip install pymoo)")
        print("  - Timeloop not available (using mock metrics)")
        print("  - Other dependencies missing")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

