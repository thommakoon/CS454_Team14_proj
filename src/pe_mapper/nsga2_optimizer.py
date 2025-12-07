"""
NSGA-II optimizer integration for multi-objective gene optimization.

Uses pymoo library to perform NSGA-II optimization over the gene space.
"""
import numpy as np
from typing import List, Tuple, Optional, Callable
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling

from .gene import Gene
from .onnx_parser import ConvLayerSpec
from .fitness import evaluate_gene, validate_gene_constraints, repair_gene


class GeneOptimizationProblem(Problem):
    """
    NSGA-II problem definition for gene optimization.
    
    Encodes gene as a decision variable vector and evaluates using fitness function.
    """
    
    def __init__(
        self,
        layer_spec: ConvLayerSpec,
        fitness_func: Callable[[Gene], Tuple[float, float, float]],
        pe_array_bounds: Tuple[int, int] = (2, 16),
        simd_bounds: Tuple[int, int] = (1, 16),
        tile_bounds: Tuple[int, int] = (1, 64),
    ):
        """
        Initialize optimization problem.
        
        Args:
            layer_spec: The convolutional layer specification
            fitness_func: Function that takes a Gene and returns (latency, energy, -utilization)
            pe_array_bounds: (min, max) for PE array dimensions
            simd_bounds: (min, max) for SIMD width
            tile_bounds: (min, max) for tiling factors
        """
        self.layer_spec = layer_spec
        self.fitness_func = fitness_func
        self.pe_array_bounds = pe_array_bounds
        self.simd_bounds = simd_bounds
        self.tile_bounds = tile_bounds
        
        # Decision variables:
        # 0: pe_array_x (int)
        # 1: pe_array_y (int)
        # 2: simd_width (int)
        # 3: buffer_budget_class (int: 0, 1, or 2)
        # 4: dataflow (int: 0=IS, 1=WS, 2=RS)
        # 5-9: tile_n, tile_c, tile_k, tile_h, tile_w (int)
        # 10-14: spatial_n, spatial_c, spatial_k, spatial_h, spatial_w (bool -> 0/1)
        # 15: loop_permutation_id (int: 0-119)
        
        n_vars = 16
        n_obj = 3  # latency, energy, -utilization
        n_constr = 0
        
        # Bounds for each variable
        xl = np.array([
            pe_array_bounds[0],      # pe_array_x
            pe_array_bounds[0],      # pe_array_y
            simd_bounds[0],          # simd_width
            0,                       # buffer_budget_class
            0,                       # dataflow
            tile_bounds[0],          # tile_n
            tile_bounds[0],          # tile_c
            tile_bounds[0],          # tile_k
            tile_bounds[0],          # tile_h
            tile_bounds[0],          # tile_w
            0,                       # spatial_n
            0,                       # spatial_c
            0,                       # spatial_k
            0,                       # spatial_h
            0,                       # spatial_w
            0,                       # loop_permutation_id
        ])
        
        xu = np.array([
            pe_array_bounds[1],      # pe_array_x
            pe_array_bounds[1],      # pe_array_y
            simd_bounds[1],          # simd_width
            2,                       # buffer_budget_class
            2,                       # dataflow
            min(tile_bounds[1], layer_spec.N),  # tile_n
            min(tile_bounds[1], layer_spec.C),  # tile_c
            min(tile_bounds[1], layer_spec.K),  # tile_k
            min(tile_bounds[1], layer_spec.H),  # tile_h
            min(tile_bounds[1], layer_spec.W),  # tile_w
            1,                       # spatial_n
            1,                       # spatial_c
            1,                       # spatial_k
            1,                       # spatial_h
            1,                       # spatial_w
            119,                     # loop_permutation_id
        ])
        
        super().__init__(
            n_var=n_vars,
            n_obj=n_obj,
            n_constr=n_constr,
            xl=xl,
            xu=xu,
        )
    
    def _decode_gene(self, x: np.ndarray) -> Gene:
        """Decode decision variable vector to Gene object."""
        # Round integer variables
        pe_array_x = int(round(x[0]))
        pe_array_y = int(round(x[1]))
        simd_width = int(round(x[2]))
        buffer_budget_class = int(round(x[3]))
        dataflow_int = int(round(x[4]))
        dataflow_map = {0: "IS", 1: "WS", 2: "RS"}
        dataflow = dataflow_map[dataflow_int]
        
        tile_n = int(round(x[5]))
        tile_c = int(round(x[6]))
        tile_k = int(round(x[7]))
        tile_h = int(round(x[8]))
        tile_w = int(round(x[9]))
        
        spatial_n = bool(round(x[10]))
        spatial_c = bool(round(x[11]))
        spatial_k = bool(round(x[12]))
        spatial_h = bool(round(x[13]))
        spatial_w = bool(round(x[14]))
        
        loop_permutation_id = int(round(x[15]))
        
        gene = Gene(
            pe_array_x=pe_array_x,
            pe_array_y=pe_array_y,
            simd_width=simd_width,
            buffer_budget_class=buffer_budget_class,
            dataflow=dataflow,
            tile_n=tile_n,
            tile_c=tile_c,
            tile_k=tile_k,
            tile_h=tile_h,
            tile_w=tile_w,
            spatial_n=spatial_n,
            spatial_c=spatial_c,
            spatial_k=spatial_k,
            spatial_h=spatial_h,
            spatial_w=spatial_w,
            loop_permutation_id=loop_permutation_id,
        )
        
        # Repair gene if needed
        if not validate_gene_constraints(gene, self.layer_spec):
            gene = repair_gene(gene, self.layer_spec)
        
        return gene
    
    def _evaluate(self, X, out, *args, **kwargs):
        """
        Evaluate population of genes.
        
        X: (n_pop, n_vars) array of decision variables
        out: dictionary to store objectives
        """
        n_pop = X.shape[0]
        objectives = np.zeros((n_pop, self.n_obj))
        
        for i in range(n_pop):
            try:
                gene = self._decode_gene(X[i])
                latency, energy, neg_utilization = self.fitness_func(gene)
                objectives[i] = [latency, energy, neg_utilization]
            except Exception as e:
                # If evaluation fails, assign worst-case objectives
                print(f"Warning: Gene evaluation failed: {e}")
                objectives[i] = [float('inf'), float('inf'), 0.0]
        
        out["F"] = objectives


def optimize_genes(
    layer_spec: ConvLayerSpec,
    fitness_func: Optional[Callable[[Gene], Tuple[float, float, float]]] = None,
    population_size: int = 20,
    n_generations: int = 10,
    seed: Optional[int] = None,
) -> Tuple[List[Gene], np.ndarray]:
    """
    Run NSGA-II optimization to find Pareto-optimal genes.
    
    Args:
        layer_spec: The convolutional layer specification
        fitness_func: Optional fitness function (uses default if None)
        population_size: Size of NSGA-II population
        n_generations: Number of generations to run
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (list of Pareto-optimal genes, array of objective values)
    """
    if fitness_func is None:
        from .fitness import evaluate_gene
        fitness_func = lambda g: evaluate_gene(g, layer_spec)
    
    # Create problem
    problem = GeneOptimizationProblem(layer_spec, fitness_func)
    
    # Create algorithm
    algorithm = NSGA2(
        pop_size=population_size,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True,
    )
    
    # Run optimization
    if seed is not None:
        np.random.seed(seed)
    
    res = minimize(
        problem,
        algorithm,
        ("n_gen", n_generations),
        verbose=True,
        seed=seed,
    )
    
    # Decode Pareto-optimal solutions
    pareto_genes = []
    for x in res.X:
        gene = problem._decode_gene(x)
        pareto_genes.append(gene)
    
    return pareto_genes, res.F

