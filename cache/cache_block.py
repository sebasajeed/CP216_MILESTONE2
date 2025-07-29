"""
Cache Block Module

This module implements the cache block (cache line) data structure used in the
ARM CPU simulator's cache system. Each cache block contains data, metadata,
and state information necessary for cache operations.

Author: CPU Simulator Project
Date: 2025
"""

from enum import Enum
from typing import Optional, List, Any
import time


class BlockState(Enum):
    """Cache block states for coherency protocol"""
    INVALID = "invalid"
    VALID = "valid"
    DIRTY = "dirty"     # Modified but not written back (write-back policy)
    SHARED = "shared"   # For multi-core coherency (future extension)


class CacheBlock:
    """
    Represents a single cache block (cache line)
    
    A cache block stores a contiguous chunk of memory data along with
    metadata required for cache management including tags, validity,
    and replacement policy information.
    """
    
    def __init__(self, block_size: int, block_id: int = 0):
        """
        Initialize a cache block
        
        Args:
            block_size: Size of the cache block in bytes
            block_id: Unique identifier for this block within the cache
        """
        self.block_size = block_size
        self.block_id = block_id
        
        # Data storage - initialized to zeros
        self.data: bytearray = bytearray(block_size)
        
        # Metadata
        self.tag: Optional[int] = None          # Tag portion of address
        self.state: BlockState = BlockState.INVALID
        self.valid: bool = False                # Quick validity check
        
        # Replacement policy metadata
        self.last_access_time: float = 0.0      # For LRU
        self.access_count: int = 0              # For LFU
        self.insertion_time: float = 0.0        # For FIFO
        
        # Statistics
        self.hit_count: int = 0
        self.miss_count: int = 0
    
    def is_valid(self) -> bool:
        """Check if block contains valid data"""
        return self.valid and self.state != BlockState.INVALID
    
    def is_dirty(self) -> bool:
        """Check if block has been modified (needs writeback)"""
        return self.state == BlockState.DIRTY
    
    def matches_address(self, address: int, tag_bits: int, block_offset_bits: int) -> bool:
        """
        Check if this block matches the given address
        
        Args:
            address: Memory address to check
            tag_bits: Number of bits used for tag
            block_offset_bits: Number of bits for block offset
            
        Returns:
            True if block matches the address tag
        """
        if not self.is_valid():
            return False
        
        # Extract tag from address
        address_tag = address >> (block_offset_bits)
        address_tag = address_tag & ((1 << tag_bits) - 1)
        
        return self.tag == address_tag
    
    def load_from_memory(self, address: int, memory_data: bytes, 
                        tag_bits: int, block_offset_bits: int) -> None:
        """
        Load data from memory into this cache block
        
        Args:
            address: Memory address being loaded
            memory_data: Data from memory to load
            tag_bits: Number of bits used for tag
            block_offset_bits: Number of bits for block offset
        """
        if len(memory_data) != self.block_size:
            raise ValueError(f"Memory data size {len(memory_data)} doesn't match block size {self.block_size}")
        
        # Extract and store tag
        address_tag = address >> block_offset_bits
        self.tag = address_tag & ((1 << tag_bits) - 1)
        
        # Copy data
        self.data[:] = memory_data
        
        # Update metadata
        self.state = BlockState.VALID
        self.valid = True
        self.last_access_time = time.time()
        self.insertion_time = time.time()
        self.access_count = 1
    
    def read_data(self, offset: int, size: int) -> bytes:
        """
        Read data from the cache block
        
        Args:
            offset: Byte offset within the block
            size: Number of bytes to read
            
        Returns:
            Data bytes read from the block
            
        Raises:
            ValueError: If offset or size is invalid
            RuntimeError: If block is invalid
        """
        if not self.is_valid():
            raise RuntimeError("Cannot read from invalid cache block")
        
        if offset < 0 or offset >= self.block_size:
            raise ValueError(f"Invalid offset {offset} for block size {self.block_size}")
        
        if size <= 0 or offset + size > self.block_size:
            raise ValueError(f"Invalid read size {size} at offset {offset}")
        
        # Update access statistics
        self.last_access_time = time.time()
        self.access_count += 1
        self.hit_count += 1
        
        return bytes(self.data[offset:offset + size])
    
    def write_data(self, offset: int, data: bytes, write_through: bool = False) -> None:
        """
        Write data to the cache block
        
        Args:
            offset: Byte offset within the block
            data: Data to write
            write_through: If True, don't mark as dirty (write-through policy)
            
        Raises:
            ValueError: If offset is invalid or data doesn't fit
            RuntimeError: If block is invalid
        """
        if not self.is_valid():
            raise RuntimeError("Cannot write to invalid cache block")
        
        if offset < 0 or offset >= self.block_size:
            raise ValueError(f"Invalid offset {offset} for block size {self.block_size}")
        
        if len(data) == 0 or offset + len(data) > self.block_size:
            raise ValueError(f"Data size {len(data)} doesn't fit at offset {offset}")
        
        # Write data
        self.data[offset:offset + len(data)] = data
        
        # Update metadata
        if not write_through:
            self.state = BlockState.DIRTY
        
        self.last_access_time = time.time()
        self.access_count += 1
    
    def invalidate(self) -> bool:
        """
        Invalidate this cache block
        
        Returns:
            True if block was dirty and needs writeback
        """
        was_dirty = self.is_dirty()
        
        self.state = BlockState.INVALID
        self.valid = False
        self.tag = None
        
        # Clear data for security (optional)
        self.data = bytearray(self.block_size)
        
        return was_dirty
    
    def get_writeback_data(self) -> Optional[bytes]:
        """
        Get data that needs to be written back to memory
        
        Returns:
            Data bytes if block is dirty, None otherwise
        """
        if self.is_dirty():
            return bytes(self.data)
        return None
    
    def get_replacement_priority(self, policy: str) -> float:
        """
        Get priority value for replacement algorithms
        
        Args:
            policy: Replacement policy name ("lru", "lfu", "fifo")
            
        Returns:
            Priority value (lower means higher priority for replacement)
        """
        if policy.lower() == "lru":
            return self.last_access_time
        elif policy.lower() == "lfu":
            return self.access_count
        elif policy.lower() == "fifo":
            return self.insertion_time
        else:
            # Random or unknown policy
            return hash(self.block_id) % 1000
    
    def clone(self) -> 'CacheBlock':
        """Create a deep copy of this cache block"""
        new_block = CacheBlock(self.block_size, self.block_id)
        new_block.data = self.data.copy()
        new_block.tag = self.tag
        new_block.state = self.state
        new_block.valid = self.valid
        new_block.last_access_time = self.last_access_time
        new_block.access_count = self.access_count
        new_block.insertion_time = self.insertion_time
        new_block.hit_count = self.hit_count
        new_block.miss_count = self.miss_count
        return new_block
    
    def get_statistics(self) -> dict:
        """Get statistics for this cache block"""
        return {
            'block_id': self.block_id,
            'valid': self.valid,
            'dirty': self.is_dirty(),
            'tag': self.tag,
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'access_count': self.access_count,
            'last_access_time': self.last_access_time
        }
    
    def __str__(self) -> str:
        """String representation of cache block"""
        return (f"CacheBlock(id={self.block_id}, tag={self.tag}, "
                f"state={self.state.value}, valid={self.valid}, "
                f"accesses={self.access_count})")
    
    def __repr__(self) -> str:
        return self.__str__()


class CacheSet:
    """
    Represents a set in a set-associative cache
    
    A cache set contains multiple cache blocks and manages them according
    to the cache's associativity and replacement policy.
    """
    
    def __init__(self, set_id: int, associativity: int, block_size: int):
        """
        Initialize a cache set
        
        Args:
            set_id: Unique identifier for this set
            associativity: Number of blocks in this set
            block_size: Size of each block in bytes
        """
        self.set_id = set_id
        self.associativity = associativity
        self.block_size = block_size
        
        # Create blocks for this set
        self.blocks: List[CacheBlock] = [
            CacheBlock(block_size, set_id * associativity + i)
            for i in range(associativity)
        ]
    
    def find_block(self, address: int, tag_bits: int, block_offset_bits: int) -> Optional[CacheBlock]:
        """
        Find a block in this set that matches the given address
        
        Args:
            address: Memory address to search for
            tag_bits: Number of bits used for tag
            block_offset_bits: Number of bits for block offset
            
        Returns:
            Matching cache block or None if not found
        """
        for block in self.blocks:
            if block.matches_address(address, tag_bits, block_offset_bits):
                return block
        return None
    
    def find_invalid_block(self) -> Optional[CacheBlock]:
        """Find an invalid block for allocation"""
        for block in self.blocks:
            if not block.is_valid():
                return block
        return None
    
    def find_replacement_victim(self, policy: str) -> CacheBlock:
        """
        Find block to replace according to replacement policy
        
        Args:
            policy: Replacement policy name
            
        Returns:
            Block to be replaced
        """
        if policy.lower() == "random":
            import random
            return random.choice(self.blocks)
        
        # For LRU, LFU, FIFO - find block with minimum priority
        victim = min(self.blocks, key=lambda b: b.get_replacement_priority(policy))
        return victim
    
    def get_statistics(self) -> dict:
        """Get statistics for this cache set"""
        total_hits = sum(block.hit_count for block in self.blocks)
        total_accesses = sum(block.access_count for block in self.blocks)
        valid_blocks = sum(1 for block in self.blocks if block.is_valid())
        dirty_blocks = sum(1 for block in self.blocks if block.is_dirty())
        
        return {
            'set_id': self.set_id,
            'total_hits': total_hits,
            'total_accesses': total_accesses,
            'valid_blocks': valid_blocks,
            'dirty_blocks': dirty_blocks,
            'blocks': [block.get_statistics() for block in self.blocks]
        }
    
    def __str__(self) -> str:
        valid_count = sum(1 for block in self.blocks if block.is_valid())
        return f"CacheSet(id={self.set_id}, valid_blocks={valid_count}/{self.associativity})"


# Example usage and testing
if __name__ == "__main__":
    # Test cache block operations
    block = CacheBlock(64, 0)
    
    # Simulate loading from memory
    memory_data = b"Hello, Cache!" + b"\x00" * 51  # 64-byte block
    block.load_from_memory(0x1000, memory_data, 20, 6)  # Example bit allocations
    
    print(f"Block after load: {block}")
    
    # Test read operation
    data = block.read_data(0, 13)
    print(f"Read data: {data}")
    
    # Test write operation
    block.write_data(0, b"Hi, Cache!!!")
    print(f"Block after write: {block}")
    print(f"Is dirty: {block.is_dirty()}")
    
    # Test cache set
    cache_set = CacheSet(0, 4, 64)  # 4-way set associative
    print(f"Cache set: {cache_set}")
