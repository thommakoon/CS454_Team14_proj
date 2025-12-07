"""
Example script showing how to use real ONNX models from the models/ directory.

This demonstrates loading a real CNN model and optimizing one of its layers.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pe_mapper import (
    Gene,
    parse_onnx_model,
    TimeloopRunner,
    evaluate_gene,
    optimize_genes,
)


def main():
    """Example of using a real ONNX model."""
    print("=" * 60)
    print("Using Real ONNX Model")
    print("=" * 60)
    
    # Path to your ONNX model
    model_path = "models/resnet18.onnx"
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"\n⚠ Model not found: {model_path}")
        print("\nTo download a model, run:")
        print("  uv run python scripts/download_model.py resnet18")
        print("\nOr place your .onnx file in the models/ directory.")
        return
    
    print(f"\n[Step 1] Loading ONNX model: {model_path}")
    try:
        layers = parse_onnx_model(model_path)
        print(f"  ✓ Found {len(layers)} convolutional layers")
        
        if len(layers) == 0:
            print("  ⚠ No convolutional layers found in model")
            return
        
        # Use the first conv layer (or you can select a specific one)
        layer_spec = layers[0]
        print(f"\n[Step 2] Using layer: {layer_spec.name}")
        print(f"  Input shape: {layer_spec.input_shape}")
        print(f"  Output shape: {layer_spec.output_shape}")
        print(f"  Kernel size: {layer_spec.kernel_size}")
        
        # Create a test gene
        print("\n[Step 3] Creating test gene...")
        gene = Gene(
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
            spatial_n=False,
            spatial_c=True,
            spatial_k=False,
            spatial_h=True,
            spatial_w=True,
            loop_permutation_id=0,
        )
        print(f"  Dataflow: {gene.dataflow}")
        print(f"  PE array: {gene.pe_array_x}x{gene.pe_array_y}")
        
        # Evaluate the gene
        print("\n[Step 4] Evaluating gene with Timeloop...")
        runner = TimeloopRunner()
        metrics = runner.run_timeloop(gene, layer_spec, keep_outputs=True)
        print(f"  Latency: {metrics.get('latency', 0):.6f} seconds")
        print(f"  Energy: {metrics.get('energy', 0):.2f} pJ")
        print(f"  Utilization: {metrics.get('utilization', 0):.2%}")
        
        # Optional: Run optimization
        print("\n[Step 5] NSGA-II Optimization (optional)...")
        print("  To run optimization, uncomment the code below:")
        print("  pareto_genes, objectives = optimize_genes(")
        print("      layer_spec,")
        print("      population_size=20,")
        print("      n_generations=10,")
        print("  )")
        
        # Uncomment to run optimization:
        # print("\n  Running optimization...")
        # pareto_genes, objectives = optimize_genes(
        #     layer_spec,
        #     population_size=10,  # Small for quick test
        #     n_generations=5,
        #     seed=42,
        # )
        # print(f"  Found {len(pareto_genes)} Pareto-optimal solutions")
        
        print("\n" + "=" * 60)
        print("✓ Successfully used real ONNX model!")
        print("=" * 60)
        
        runner.cleanup()
        
    except FileNotFoundError:
        print(f"  ✗ Model file not found: {model_path}")
        print("  Download a model first using scripts/download_model.py")
    except Exception as e:
        print(f"  ✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

