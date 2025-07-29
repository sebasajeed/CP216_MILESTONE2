"""
Cache Implementation Module

This module implements the main cache system for the ARM CPU simulator.
It supports various cache configurations including direct-mapped,
set-associative, and fully-associative caches with different replacement
policies and write strategies.

Author: CPU Simulator Project
Date: 2025
"""

import math
import time
from typing import Dict, List, Optional, Tuple, Any
from .cache_config import CacheConfig, ReplacementPolicy, WritePolicy
from .cache_block import CacheBlock, CacheSet, BlockState


class CacheAccessResult:
    """Result of a cache access operation"""
    
    def __init__(self, hit: bool, data: Optional[bytes] = None, 
                 cycles: int = 1, writeback_needed: bool = False,
                 writeback_address: Optional[int] = None,
                 writeback_data: Optional[bytes] = None):
        self.hit = hit
        self.data = data
        self.cycles = cycles
        self.writeback_needed = writeback_needed
        self.writeback_address = writeback_address
        self.writeback_data = writeback_data


class Cache:
    """
    Main cache implementation supporting various cache architectures
    
    This class implements a configurable cache system that can simulate
    different cache organizations including direct-mapped, set-associative,
    and fully-associative caches with various replacement policies.
    """
    
    def __init__(self, config: CacheConfig, name: str = "Cache"):
        """
        Initialize cache with given configuration
        
        Args:
            config: Cache configuration parameters
            name: Name identifier for this cache
        """
        self.config = config
        self.name = name
        
        # Calculate address bit allocations
        self.block_offset_bits = int(math.log2(config.block_size))
        self.set_index_bits = int(math.log2(config.num_sets))
        self.tag_bits = 32 - self.set_index_bits - self.block_offset_bits  # Assume 32-bit addresses
        
        # Create cache structure
        self.sets: List[CacheSet] = [
            CacheSet(i, config.associativity, config.block_size)
            for i in range(config.num_sets)
        ]
        
        # Statistics
        self.stats = config.get_cache_stats_template()
        self.access_history: List[Dict[str, Any]] = []
        
        # Performance counters
        self.total_cycles = 0
        self.start_time = time.time()
        
        print(f"Initialized {name}: {config.size//1024}KB, "
              f"{config.num_sets} sets, {config.associativity}-way, "
              f"{config.block_size}B blocks")
        print(f"Address bits: {self.tag_bits} tag, {self.set_index_bits} index, "
              f"{self.block_offset_bits} offset")
    
    def _decode_address(self, address: int) -> Tuple[int, int, int]:
        """
        Decode memory address into cache components
        
        Args:
            address: Memory address to decode
            
        Returns:
            Tuple of (tag, set_index, block_offset)
        """
        block_offset = address & ((1 << self.block_offset_bits) - 1)
        set_index = (address >> self.block_offset_bits) & ((1 << self.set_index_bits) - 1)
        tag = (address >> (self.block_offset_bits + self.set_index_bits)) & ((1 << self.tag_bits) - 1)
        
        return tag, set_index, block_offset
    
    def _get_block_address(self, address: int) -> int:
        """Get the starting address of the block containing the given address"""
        return address & ~((1 << self.block_offset_bits) - 1)
    
    def read(self, address: int, size: int = 4, memory_interface=None) -> CacheAccessResult:
        """
        Read data from cache
        
        Args:
            address: Memory address to read from
            size: Number of bytes to read
            memory_interface: Interface to main memory for misses
            
        Returns:
            CacheAccessResult with read data and access information
        """
        self.stats['reads'] += 1
        self.stats['total_accesses'] += 1
        
        # Record access for analysis
        access_record = {
            'type': 'read',
            'address': address,
            'size': size,
            'timestamp': time.time()
        }
        
        tag, set_index, block_offset = self._decode_address(address)
        cache_set = self.sets[set_index]
        
        # Try to find block in cache
        block = cache_set.find_block(address, self.tag_bits, self.block_offset_bits)
        
        if block and block.is_valid():
            # Cache hit
            self.stats['hits'] += 1
            access_record['result'] = 'hit'
            
            try:
                data = block.read_data(block_offset, size)
                result = CacheAccessResult(
                    hit=True,
                    data=data,
                    cycles=self.config.hit_time
                )
                
                self.total_cycles += self.config.hit_time
                access_record['cycles'] = self.config.hit_time
                
            except (ValueError, RuntimeError) as e:
                # Error in block access - treat as miss
                self.stats['hits'] -= 1  # Correct the hit count
                result = self._handle_cache_miss(address, size, cache_set, tag, 
                                               block_offset, memory_interface, 'read')
                access_record['result'] = 'miss_error'
                access_record['error'] = str(e)
        else:
            # Cache miss
            result = self._handle_cache_miss(address, size, cache_set, tag, 
                                           block_offset, memory_interface, 'read')
            access_record['result'] = 'miss'
            access_record['cycles'] = result.cycles
        
        self.access_history.append(access_record)
        return result
    
    def write(self, address: int, data: bytes, memory_interface=None) -> CacheAccessResult:
        """
        Write data to cache
        
        Args:
            address: Memory address to write to
            data: Data bytes to write
            memory_interface: Interface to main memory
            
        Returns:
            CacheAccessResult with write operation information
        """
        self.stats['writes'] += 1
        self.stats['total_accesses'] += 1
        
        # Record access for analysis
        access_record = {
            'type': 'write',
            'address': address,
            'size': len(data),
            'timestamp': time.time()
        }
        
        tag, set_index, block_offset = self._decode_address(address)
        cache_set = self.sets[set_index]
        
        # Try to find block in cache
        block = cache_set.find_block(address, self.tag_bits, self.block_offset_bits)
        
        if block and block.is_valid():
            # Write hit
            self.stats['hits'] += 1
            access_record['result'] = 'hit'
            
            # Write to cache block
            write_through = (self.config.write_policy == WritePolicy.WRITE_THROUGH)
            block.write_data(block_offset, data, write_through)
            
            cycles = self.config.hit_time
            writeback_needed = False
            writeback_address = None
            writeback_data = None
            
            # Handle write-through policy
            if write_through and memory_interface:
                # Write to memory immediately
                memory_interface.write(address, data)
                cycles += 1  # Additional cycle for memory write
            
            result = CacheAccessResult(
                hit=True,
                cycles=cycles,
                writeback_needed=writeback_needed,
                writeback_address=writeback_address,
                writeback_data=writeback_data
            )
            
            self.total_cycles += cycles
            access_record['cycles'] = cycles
            
        else:
            # Write miss
            if self.config.write_allocate:
                # Allocate block on write miss
                result = self._handle_cache_miss(address, len(data), cache_set, tag, 
                                               block_offset, memory_interface, 'write', data)
                access_record['result'] = 'miss_allocate'
            else:
                # Write directly to memory without allocation
                if memory_interface:
                    memory_interface.write(address, data)
                
                self.stats['misses'] += 1
                result = CacheAccessResult(
                    hit=False,
                    cycles=self.config.miss_penalty
                )
                
                self.total_cycles += self.config.miss_penalty
                access_record['result'] = 'miss_no_allocate'
                access_record['cycles'] = self.config.miss_penalty
        
        self.access_history.append(access_record)
        return result
    
    def _handle_cache_miss(self, address: int, size: int, cache_set: CacheSet, 
                          tag: int, block_offset: int, memory_interface, 
                          operation: str, write_data: Optional[bytes] = None) -> CacheAccessResult:
        """
        Handle cache miss by loading block from memory
        
        Args:
            address: Memory address that missed
            size: Size of the access
            cache_set: Cache set where block should be placed
            tag: Tag portion of address
            block_offset: Offset within the block
            memory_interface: Interface to main memory
            operation: 'read' or 'write'
            write_data: Data to write (for write misses with allocation)
            
        Returns:
            CacheAccessResult with miss handling information
        """
        self.stats['misses'] += 1
        
        # Find block to replace
        victim_block = cache_set.find_invalid_block()
        writeback_needed = False
        writeback_address = None
        writeback_data = None
        
        if victim_block is None:
            # Need to evict a block
            victim_block = cache_set.find_replacement_victim(self.config.replacement_policy.value)
            
            # Check if victim needs writeback
            if victim_block.is_dirty():
                writeback_needed = True
                writeback_address = self._reconstruct_address(victim_block.tag, 
                                                             cache_set.set_id, 0)
                writeback_data = victim_block.get_writeback_data()
                self.stats['writebacks'] += 1
                
                # Perform writeback if memory interface available
                if memory_interface and writeback_data:
                    memory_interface.write(writeback_address, writeback_data)
        
        # Load new block from memory
        block_address = self._get_block_address(address)
        
        if memory_interface:
            # Load full block from memory
            memory_data = memory_interface.read(block_address, self.config.block_size)
            
            # Load data into cache block
            victim_block.load_from_memory(address, memory_data, 
                                        self.tag_bits, self.block_offset_bits)
            
            # Handle the original operation
            if operation == 'read':
                data = victim_block.read_data(block_offset, size)
            else:  # write
                if write_data:
                    write_through = (self.config.write_policy == WritePolicy.WRITE_THROUGH)
                    victim_block.write_data(block_offset, write_data, write_through)
                data = None
        else:
            # No memory interface - simulate with zeros
            memory_data = bytes(self.config.block_size)
            victim_block.load_from_memory(address, memory_data, 
                                        self.tag_bits, self.block_offset_bits)
            
            if operation == 'read':
                data = bytes(size)  # Return zeros
            else:
                data = None
        
        # Calculate total cycles
        cycles = self.config.hit_time + self.config.miss_penalty
        if writeback_needed:
            cycles += self.config.miss_penalty // 2  # Additional writeback cost
        
        self.total_cycles += cycles
        
        return CacheAccessResult(
            hit=False,
            data=data,
            cycles=cycles,
            writeback_needed=writeback_needed,
            writeback_address=writeback_address,
            writeback_data=writeback_data
        )
    
    def _reconstruct_address(self, tag: int, set_index: int, block_offset: int) -> int:
        """Reconstruct memory address from cache components"""
        return ((tag << (self.set_index_bits + self.block_offset_bits)) |
                (set_index << self.block_offset_bits) |
                block_offset)
    
    def flush(self, memory_interface=None) -> int:
        """
        Flush all dirty blocks to memory
        
        Args:
            memory_interface: Interface to main memory
            
        Returns:
            Number of blocks written back
        """
        writebacks = 0
        
        for cache_set in self.sets:
            for block in cache_set.blocks:
                if block.is_dirty():
                    # Get writeback data
                    writeback_data = block.get_writeback_data()
                    if writeback_data and memory_interface:
                        # Reconstruct block address
                        block_address = self._reconstruct_address(block.tag, 
                                                                cache_set.set_id, 0)
                        memory_interface.write(block_address, writeback_data)
                    
                    # Mark block as clean
                    block.state = BlockState.VALID
                    writebacks += 1
        
        self.stats['writebacks'] += writebacks
        return writebacks
    
    def invalidate(self) -> int:
        """
        Invalidate all cache blocks
        
        Returns:
            Number of blocks that were dirty and needed writeback
        """
        dirty_blocks = 0
        
        for cache_set in self.sets:
            for block in cache_set.blocks:
                if block.invalidate():  # Returns True if was dirty
                    dirty_blocks += 1
        
        return dirty_blocks
    
    def get_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        if self.stats['total_accesses'] == 0:
            return 0.0
        return self.stats['hits'] / self.stats['total_accesses']
    
    def get_miss_rate(self) -> float:
        """Calculate cache miss rate"""
        return 1.0 - self.get_hit_rate()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        runtime = time.time() - self.start_time
        
        stats = self.stats.copy()
        stats.update({
            'name': self.name,
            'config': {
                'size': self.config.size,
                'block_size': self.config.block_size,
                'associativity': self.config.associativity,
                'num_sets': self.config.num_sets,
                'replacement_policy': self.config.replacement_policy.value,
                'write_policy': self.config.write_policy.value
            },
            'hit_rate': self.get_hit_rate(),
            'miss_rate': self.get_miss_rate(),
            'total_cycles': self.total_cycles,
            'average_access_time': (self.total_cycles / max(1, self.stats['total_accesses'])),
            'runtime_seconds': runtime
        })
        
        return stats
    
    def get_detailed_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics including per-set information"""
        basic_stats = self.get_statistics()
        
        # Add per-set statistics
        set_stats = []
        for cache_set in self.sets:
            set_stats.append(cache_set.get_statistics())
        
        basic_stats['set_statistics'] = set_stats
        basic_stats['access_history'] = self.access_history[-100:]  # Last 100 accesses
        
        return basic_stats
    
    def reset_statistics(self):
        """Reset all cache statistics"""
        self.stats = self.config.get_cache_stats_template()
        self.access_history.clear()
        self.total_cycles = 0
        self.start_time = time.time()
        
        # Reset block statistics
        for cache_set in self.sets:
            for block in cache_set.blocks:
                block.hit_count = 0
                block.miss_count = 0
                block.access_count = 0
    
    def print_status(self):
        """Print current cache status"""
        stats = self.get_statistics()
        
        print(f"\n=== {self.name} Status ===")
        print(f"Size: {self.config.size // 1024}KB")
        print(f"Hits: {stats['hits']}, Misses: {stats['misses']}")
        print(f"Hit Rate: {stats['hit_rate']:.3f}")
        print(f"Total Accesses: {stats['total_accesses']}")
        print(f"Writebacks: {stats['writebacks']}")
        print(f"Average Access Time: {stats['average_access_time']:.2f} cycles")
        print(f"Total Cycles: {stats['total_cycles']}")
    
    def __str__(self) -> str:
        return (f"{self.name}({self.config.size//1024}KB, "
                f"{self.config.associativity}-way, "
                f"hit_rate={self.get_hit_rate():.3f})")


# Example usage and testing
if __name__ == "__main__":
    from cache_config import CacheConfigPresets
    
    # Test different cache configurations
    print("Testing Cache Implementation\n")
    
    # Create a simple direct-mapped cache
    config = CacheConfigPresets.direct_mapped_cache(8192)  # 8KB
    cache = Cache(config, "L1-Data")
    
    # Simulate some memory accesses
    print("Simulating memory accesses...")
    
    # Sequential access pattern
    for addr in range(0x1000, 0x1100, 4):
        result = cache.read(addr, 4)
        if addr == 0x1000:
            print(f"First access (miss): {result.hit}, cycles: {result.cycles}")
    
    # Repeated access (should hit)
    result = cache.read(0x1000, 4)
    print(f"Repeated access (hit): {result.hit}, cycles: {result.cycles}")
    
    # Print final statistics
    cache.print_status()
