"""
CS454 Team 14 Project: PE-focused CNN Accelerator Mapping Optimization

A GenCNN-like approach using Timeloop, ONNX, and NSGA-II for optimizing
convolutional layer mappings on PE arrays with IS/WS/RS dataflows.
"""

from .gene import Gene, create_example_genes
from .onnx_parser import ConvLayerSpec, parse_onnx_model, get_resnet18_conv3_example
from .timeloop_integration import TimeloopRunner
from .fitness import evaluate_gene, validate_gene_constraints, repair_gene
from .nsga2_optimizer import GeneOptimizationProblem, optimize_genes
from .batch_optimizer import optimize_all_models, optimize_all_layers_from_model, summarize_results

__version__ = "0.1.0"
__all__ = [
    "Gene",
    "create_example_genes",
    "ConvLayerSpec",
    "parse_onnx_model",
    "get_resnet18_conv3_example",
    "TimeloopRunner",
    "evaluate_gene",
    "validate_gene_constraints",
    "repair_gene",
    "GeneOptimizationProblem",
    "optimize_genes",
    "optimize_all_models",
    "optimize_all_layers_from_model",
    "summarize_results",
]

