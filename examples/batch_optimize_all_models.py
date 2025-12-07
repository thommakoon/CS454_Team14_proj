"""
Batch optimization script - optimizes all models in models/ directory.

This script will:
1. Find all .onnx files in models/ directory
2. Extract all convolutional layers from each model
3. Run NSGA-II optimization on each layer
4. Summarize results
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pe_mapper import optimize_all_models, summarize_results


def main():
    """Run batch optimization on all models."""
    print("=" * 60)
    print("Batch Optimization: All Models")
    print("=" * 60)
    
    print("\nThis will optimize ALL models in the models/ directory.")
    print("This may take a very long time!")
    print("\nOptions:")
    print("  - Optimize all layers from all models")
    print("  - Or limit layers per model for faster testing")
    
    # Configuration
    models_dir = "models"
    population_size = 20
    n_generations = 10
    max_layers_per_model = 3  # Set to None to optimize all layers
    seed = 42
    
    print(f"\nConfiguration:")
    print(f"  Models directory: {models_dir}")
    print(f"  Population size: {population_size}")
    print(f"  Generations: {n_generations}")
    if max_layers_per_model:
        print(f"  Max layers per model: {max_layers_per_model}")
    else:
        print(f"  Max layers per model: ALL")
    
    # Check if models directory exists
    if not Path(models_dir).exists():
        print(f"\n⚠ Models directory not found: {models_dir}")
        print("  Create it and add .onnx files, or run:")
        print("  uv run python scripts/download_model.py resnet18")
        return
    
    # Run batch optimization
    print("\n" + "=" * 60)
    print("Starting batch optimization...")
    print("=" * 60)
    
    try:
        results = optimize_all_models(
            models_dir=models_dir,
            population_size=population_size,
            n_generations=n_generations,
            max_layers_per_model=max_layers_per_model,
            seed=seed,
        )
        
        # Summarize results
        if results:
            summarize_results(results)
            
            print("\n" + "=" * 60)
            print("✓ Batch optimization complete!")
            print("=" * 60)
            print("\nResults are stored in the 'results' dictionary.")
            print("You can access them programmatically or save to file.")
        else:
            print("\n⚠ No results generated. Check if models exist and have conv layers.")
            
    except KeyboardInterrupt:
        print("\n\n⚠ Optimization interrupted by user")
    except Exception as e:
        print(f"\n✗ Error during batch optimization: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

