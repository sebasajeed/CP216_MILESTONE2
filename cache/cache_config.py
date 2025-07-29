"""
Cache Configuration Module

This module defines the configuration parameters for different cache levels
in the ARM CPU simulator. It includes settings for cache size, associativity,
block size, and replacement policies.

Author: CPU Simulator Project
Date: 2025
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any


class ReplacementPolicy(Enum):
    """Cache replacement policies"""
    LRU = "lru"          # Least Recently Used
    FIFO = "fifo"        # First In First Out
    RANDOM = "random"    # Random replacement
    LFU = "lfu"          # Least Frequently Used


class WritePolicy(Enum):
    """Cache write policies"""
    WRITE_THROUGH = "write_through"
    WRITE_BACK = "write_back"


class CacheMappingType(Enum):
    """Cache mapping types"""
    DIRECT_MAPPED = "direct_mapped"
    FULLY_ASSOCIATIVE = "fully_associative"
    SET_ASSOCIATIVE = "set_associative"


@dataclass
class CacheConfig:
    """
    Configuration class for cache parameters
    
    Attributes:
        size: Cache size in bytes
        block_size: Size of each cache block in bytes
        associativity: Number of ways in set-associative cache (1 for direct-mapped)
        replacement_policy: Policy for replacing cache blocks
        write_policy: Policy for handling write operations
        write_allocate: Whether to allocate block on write miss
        hit_time: Time to access cache on hit (in cycles)
        miss_penalty: Additional cycles for cache miss
    """
    size: int
    block_size: int
    associativity: int
    replacement_policy: ReplacementPolicy
    write_policy: WritePolicy
    write_allocate: bool = True
    hit_time: int = 1
    miss_penalty: int = 10
    
    def __post_init__(self):
        """Validate cache configuration parameters"""
        if self.size <= 0:
            raise ValueError("Cache size must be positive")
        if self.block_size <= 0 or (self.block_size & (self.block_size - 1)) != 0:
            raise ValueError("Block size must be a positive power of 2")
        if self.associativity <= 0:
            raise ValueError("Associativity must be positive")
        if self.size % self.block_size != 0:
            raise ValueError("Cache size must be divisible by block size")
        
        # Calculate derived parameters
        self.num_blocks = self.size // self.block_size
        self.num_sets = self.num_blocks // self.associativity
        
        if self.num_sets <= 0:
            raise ValueError("Invalid cache configuration: no valid sets")
    
    @property
    def mapping_type(self) -> CacheMappingType:
        """Determine the cache mapping type based on associativity"""
        if self.associativity == 1:
            return CacheMappingType.DIRECT_MAPPED
        elif self.associativity == self.num_blocks:
            return CacheMappingType.FULLY_ASSOCIATIVE
        else:
            return CacheMappingType.SET_ASSOCIATIVE
    
    def get_cache_stats_template(self) -> Dict[str, int]:
        """Return template for cache statistics"""
        return {
            'hits': 0,
            'misses': 0,
            'reads': 0,
            'writes': 0,
            'writebacks': 0,
            'total_accesses': 0
        }


class CacheConfigPresets:
    """Predefined cache configurations for common scenarios"""
    
    @staticmethod
    def l1_data_cache() -> CacheConfig:
        """Typical L1 data cache configuration"""
        return CacheConfig(
            size=32 * 1024,  # 32KB
            block_size=64,   # 64 bytes
            associativity=8, # 8-way set associative
            replacement_policy=ReplacementPolicy.LRU,
            write_policy=WritePolicy.WRITE_BACK,
            write_allocate=True,
            hit_time=1,
            miss_penalty=12
        )
    
    @staticmethod
    def l1_instruction_cache() -> CacheConfig:
        """Typical L1 instruction cache configuration"""
        return CacheConfig(
            size=32 * 1024,  # 32KB
            block_size=64,   # 64 bytes
            associativity=8, # 8-way set associative
            replacement_policy=ReplacementPolicy.LRU,
            write_policy=WritePolicy.WRITE_THROUGH,  # Instructions rarely written
            write_allocate=False,
            hit_time=1,
            miss_penalty=12
        )
    
    @staticmethod
    def l2_unified_cache() -> CacheConfig:
        """Typical L2 unified cache configuration"""
        return CacheConfig(
            size=256 * 1024,  # 256KB
            block_size=64,    # 64 bytes
            associativity=16, # 16-way set associative
            replacement_policy=ReplacementPolicy.LRU,
            write_policy=WritePolicy.WRITE_BACK,
            write_allocate=True,
            hit_time=8,
            miss_penalty=100
        )
    
    @staticmethod
    def direct_mapped_cache(size: int = 16 * 1024) -> CacheConfig:
        """Simple direct-mapped cache for testing"""
        return CacheConfig(
            size=size,
            block_size=32,
            associativity=1,  # Direct mapped
            replacement_policy=ReplacementPolicy.FIFO,  # Not used in direct mapped
            write_policy=WritePolicy.WRITE_THROUGH,
            write_allocate=True,
            hit_time=1,
            miss_penalty=20
        )
    
    @staticmethod
    def fully_associative_cache(size: int = 8 * 1024) -> CacheConfig:
        """Fully associative cache configuration"""
        num_blocks = size // 32  # 32-byte blocks
        return CacheConfig(
            size=size,
            block_size=32,
            associativity=num_blocks,  # Fully associative
            replacement_policy=ReplacementPolicy.LRU,
            write_policy=WritePolicy.WRITE_BACK,
            write_allocate=True,
            hit_time=2,
            miss_penalty=25
        )


def validate_cache_hierarchy(configs: Dict[str, CacheConfig]) -> bool:
    """
    Validate a cache hierarchy configuration
    
    Args:
        configs: Dictionary mapping cache level names to configurations
        
    Returns:
        True if hierarchy is valid, False otherwise
    """
    if not configs:
        return False
    
    # Sort by cache size to ensure proper hierarchy
    sorted_caches = sorted(configs.items(), key=lambda x: x[1].size)
    
    # Check that higher levels have smaller sizes
    for i in range(len(sorted_caches) - 1):
        current_cache = sorted_caches[i][1]
        next_cache = sorted_caches[i + 1][1]
        
        if current_cache.size >= next_cache.size:
            return False
        if current_cache.hit_time >= next_cache.hit_time:
            return False
    
    return True


# Example usage and testing
if __name__ == "__main__":
    # Test cache configurations
    l1_cache = CacheConfigPresets.l1_data_cache()
    print(f"L1 Cache: {l1_cache.num_sets} sets, {l1_cache.mapping_type.value}")
    
    l2_cache = CacheConfigPresets.l2_unified_cache()
    print(f"L2 Cache: {l2_cache.num_sets} sets, {l2_cache.mapping_type.value}")
    
    # Test cache hierarchy
    hierarchy = {
        'l1': l1_cache,
        'l2': l2_cache
    }
    
    print(f"Cache hierarchy valid: {validate_cache_hierarchy(hierarchy)}")
