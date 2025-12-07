"""
Gene definition for CNN accelerator mapping optimization.

A gene encodes a complete mapping strategy for a single convolutional layer,
including hardware configuration and dataflow mapping.
"""
from dataclasses import dataclass
from typing import Literal
import numpy as np


@dataclass
class Gene:
    """
    Minimal but complete gene for one conv layer.
    
    Hardware/mapping side:
    - PE array dimensions
    - SIMD width per PE
    - Buffer sizes (or buffer budget class)
    - Dataflow mode
    
    Mapping side:
    - Tiling factors for N, C, K, H, W
    - Input partitioning (spatial vs temporal)
    - Execution order (loop permutation)
    """
    # Hardware configuration
    pe_array_x: int  # PE array width
    pe_array_y: int  # PE array height
    simd_width: int  # SIMD width per PE
    buffer_budget_class: int  # 0=small, 1=medium, 2=large (maps to actual sizes)
    dataflow: Literal["IS", "WS", "RS"]  # Input-Stationary, Weight-Stationary, Row-Stationary
    
    # Tiling factors (for N, C, K, H, W dimensions)
    tile_n: int
    tile_c: int
    tile_k: int
    tile_h: int
    tile_w: int
    
    # Partitioning: which dimensions are spatially mapped vs temporally iterated
    # 1 = spatially mapped, 0 = temporally iterated
    spatial_n: bool
    spatial_c: bool
    spatial_k: bool
    spatial_h: bool
    spatial_w: bool
    
    # Execution order: permutation ID (0-119 for 5! permutations)
    # Maps to a specific loop order
    loop_permutation_id: int
    
    def __post_init__(self):
        """Validate gene constraints."""
        assert self.pe_array_x > 0 and self.pe_array_y > 0
        assert self.simd_width > 0
        assert self.buffer_budget_class in [0, 1, 2]
        assert self.dataflow in ["IS", "WS", "RS"]
        assert all(t > 0 for t in [self.tile_n, self.tile_c, self.tile_k, self.tile_h, self.tile_w])
        assert 0 <= self.loop_permutation_id < 120  # 5! = 120
    
    def to_dict(self):
        """Convert gene to dictionary for serialization."""
        return {
            "pe_array_x": self.pe_array_x,
            "pe_array_y": self.pe_array_y,
            "simd_width": self.simd_width,
            "buffer_budget_class": self.buffer_budget_class,
            "dataflow": self.dataflow,
            "tile_n": self.tile_n,
            "tile_c": self.tile_c,
            "tile_k": self.tile_k,
            "tile_h": self.tile_h,
            "tile_w": self.tile_w,
            "spatial_n": self.spatial_n,
            "spatial_c": self.spatial_c,
            "spatial_k": self.spatial_k,
            "spatial_h": self.spatial_h,
            "spatial_w": self.spatial_w,
            "loop_permutation_id": self.loop_permutation_id,
        }
    
    @classmethod
    def from_dict(cls, d):
        """Create gene from dictionary."""
        return cls(**d)
    
    def get_buffer_sizes(self):
        """
        Map buffer_budget_class to actual buffer sizes.
        Returns (input_buffer, weight_buffer, output_buffer) in bytes.
        """
        # Buffer sizes in KB (will be converted to bytes)
        buffer_configs = {
            0: (8, 8, 8),    # Small: 8KB each
            1: (32, 32, 32), # Medium: 32KB each
            2: (128, 128, 128), # Large: 128KB each
        }
        sizes_kb = buffer_configs[self.buffer_budget_class]
        return tuple(s * 1024 for s in sizes_kb)  # Convert to bytes
    
    def get_loop_order(self):
        """
        Convert loop_permutation_id to actual loop order.
        Dimensions: N, C, K, H, W (indexed 0-4)
        """
        dims = ['N', 'C', 'K', 'H', 'W']
        # Generate all permutations
        from itertools import permutations
        perms = list(permutations(range(5)))
        if self.loop_permutation_id < len(perms):
            perm = perms[self.loop_permutation_id]
            return [dims[i] for i in perm]
        return dims  # Default order


def create_example_genes():
    """
    Create a few handcrafted example genes for testing.
    """
    # Example 1: Input-Stationary, small PE array, medium buffers
    gene1 = Gene(
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
        loop_permutation_id=0,  # N, C, K, H, W
    )
    
    # Example 2: Weight-Stationary, larger PE array
    gene2 = Gene(
        pe_array_x=8,
        pe_array_y=8,
        simd_width=8,
        buffer_budget_class=2,
        dataflow="WS",
        tile_n=2,
        tile_c=32,
        tile_k=64,
        tile_h=14,
        tile_w=14,
        spatial_n=False,
        spatial_c=False,
        spatial_k=True,
        spatial_h=True,
        spatial_w=True,
        loop_permutation_id=10,
    )
    
    # Example 3: Row-Stationary, balanced
    gene3 = Gene(
        pe_array_x=6,
        pe_array_y=6,
        simd_width=4,
        buffer_budget_class=1,
        dataflow="RS",
        tile_n=1,
        tile_c=16,
        tile_k=16,
        tile_h=7,
        tile_w=7,
        spatial_n=False,
        spatial_c=True,
        spatial_k=True,
        spatial_h=True,
        spatial_w=True,
        loop_permutation_id=20,
    )
    
    return [gene1, gene2, gene3]

