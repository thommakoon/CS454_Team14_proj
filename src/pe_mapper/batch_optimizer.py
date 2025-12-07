"""
Batch optimization utilities for optimizing multiple models and layers.

Allows running NSGA-II optimization across multiple ONNX models and layers.
"""
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np

from .onnx_parser import parse_onnx_model, ConvLayerSpec
from .nsga2_optimizer import optimize_genes
from .gene import Gene


def optimize_all_models(
    models_dir: str = "models",
    population_size: int = 20,
    n_generations: int = 10,
    max_layers_per_model: Optional[int] = None,
    seed: Optional[int] = None,
) -> Dict[str, Dict[str, Tuple[List[Gene], np.ndarray]]]:
    """
    Optimize all ONNX models in a directory.
    
    Args:
        models_dir: Directory containing ONNX model files
        population_size: NSGA-II population size
        n_generations: Number of generations
        max_layers_per_model: Maximum layers to optimize per model (None = all)
        seed: Random seed
    
    Returns:
        Dictionary mapping model_name -> {layer_name: (pareto_genes, objectives)}
    """
    results = {}
    models_path = Path(models_dir)
    
    if not models_path.exists():
        print(f"⚠ Models directory not found: {models_dir}")
        return results
    
    # Find all ONNX files
    onnx_files = list(models_path.glob("*.onnx"))
    
    if len(onnx_files) == 0:
        print(f"⚠ No ONNX files found in {models_dir}")
        return results
    
    print(f"Found {len(onnx_files)} ONNX model(s)")
    
    for model_file in onnx_files:
        model_name = model_file.stem
        print(f"\n{'='*60}")
        print(f"Processing model: {model_name}")
        print(f"{'='*60}")
        
        try:
            # Parse model
            layers = parse_onnx_model(str(model_file))
            print(f"  Found {len(layers)} convolutional layers")
            
            if len(layers) == 0:
                print(f"  ⚠ No conv layers found, skipping")
                continue
            
            # Limit layers if specified
            if max_layers_per_model:
                layers = layers[:max_layers_per_model]
                print(f"  Optimizing first {len(layers)} layer(s)")
            
            model_results = {}
            
            # Optimize each layer
            for i, layer_spec in enumerate(layers, 1):
                print(f"\n  [{i}/{len(layers)}] Optimizing layer: {layer_spec.name}")
                print(f"      Input: {layer_spec.input_shape}")
                print(f"      Output: {layer_spec.output_shape}")
                
                try:
                    pareto_genes, objectives = optimize_genes(
                        layer_spec,
                        population_size=population_size,
                        n_generations=n_generations,
                        seed=seed,
                    )
                    
                    model_results[layer_spec.name] = (pareto_genes, objectives)
                    print(f"      ✓ Found {len(pareto_genes)} Pareto-optimal solutions")
                    
                except Exception as e:
                    print(f"      ✗ Error optimizing layer: {e}")
                    continue
            
            if model_results:
                results[model_name] = model_results
                print(f"\n  ✓ Completed {model_name}: {len(model_results)} layers optimized")
            else:
                print(f"\n  ⚠ No layers optimized for {model_name}")
                
        except Exception as e:
            print(f"  ✗ Error processing {model_name}: {e}")
            continue
    
    return results


def optimize_all_layers_from_model(
    model_path: str,
    population_size: int = 20,
    n_generations: int = 10,
    max_layers: Optional[int] = None,
    seed: Optional[int] = None,
) -> Dict[str, Tuple[List[Gene], np.ndarray]]:
    """
    Optimize all layers from a single ONNX model.
    
    Args:
        model_path: Path to ONNX model file
        population_size: NSGA-II population size
        n_generations: Number of generations
        max_layers: Maximum layers to optimize (None = all)
        seed: Random seed
    
    Returns:
        Dictionary mapping layer_name -> (pareto_genes, objectives)
    """
    print(f"Loading model: {model_path}")
    layers = parse_onnx_model(model_path)
    
    if len(layers) == 0:
        print("⚠ No convolutional layers found")
        return {}
    
    print(f"Found {len(layers)} convolutional layers")
    
    if max_layers:
        layers = layers[:max_layers]
        print(f"Optimizing first {len(layers)} layer(s)")
    
    results = {}
    
    for i, layer_spec in enumerate(layers, 1):
        print(f"\n[{i}/{len(layers)}] Optimizing: {layer_spec.name}")
        print(f"  Input: {layer_spec.input_shape}, Output: {layer_spec.output_shape}")
        
        try:
            pareto_genes, objectives = optimize_genes(
                layer_spec,
                population_size=population_size,
                n_generations=n_generations,
                seed=seed,
            )
            
            results[layer_spec.name] = (pareto_genes, objectives)
            print(f"  ✓ Found {len(pareto_genes)} Pareto-optimal solutions")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue
    
    return results


def summarize_results(results: Dict[str, Dict[str, Tuple[List[Gene], np.ndarray]]]) -> None:
    """
    Print a summary of batch optimization results.
    
    Args:
        results: Results dictionary from optimize_all_models()
    """
    print("\n" + "="*60)
    print("Batch Optimization Summary")
    print("="*60)
    
    total_models = len(results)
    total_layers = sum(len(layers) for layers in results.values())
    total_solutions = sum(
        len(genes) 
        for model_results in results.values() 
        for genes, _ in model_results.values()
    )
    
    print(f"\nModels processed: {total_models}")
    print(f"Layers optimized: {total_layers}")
    print(f"Total Pareto solutions: {total_solutions}")
    
    print("\nPer-model breakdown:")
    for model_name, model_results in results.items():
        num_layers = len(model_results)
        num_solutions = sum(len(genes) for genes, _ in model_results.values())
        print(f"  {model_name}: {num_layers} layers, {num_solutions} solutions")
        
        # Show dataflow distribution
        dataflow_counts = {}
        for genes, _ in model_results.values():
            for gene in genes:
                dataflow_counts[gene.dataflow] = dataflow_counts.get(gene.dataflow, 0) + 1
        
        if dataflow_counts:
            print(f"    Dataflows: {dict(dataflow_counts)}")

