"""
Memory System Module

This module implements the main memory system for the ARM CPU simulator.
It provides a realistic memory interface with configurable latency,
bandwidth, and capacity constraints.

Author: CPU Simulator Project  
Date: 2025
"""

import time
import threading
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum


class MemoryAccessType(Enum):
    """Types of memory access operations"""
    READ = "read"
    WRITE = "write"
    INSTRUCTION_FETCH = "instruction_fetch"


class MemoryRegion:
    """Represents a memory region with specific properties"""
    
    def __init__(self, start_address: int, size: int, name: str,
                 readable: bool = True, writable: bool = True, 
                 executable: bool = False, latency_cycles: int = 100):
        self.start_address = start_address
        self.size = size
        self.end_address = start_address + size - 1
        self.name = name
        self.readable = readable
        self.writable = writable
        self.executable = executable
        self.latency_cycles = latency_cycles
        
        # Initialize memory content
        self.data = bytearray(size)
        
        # Statistics
        self.read_count = 0
        self.write_count = 0
        self.total_bytes_read = 0
        self.total_bytes_written = 0
    
    def contains_address(self, address: int) -> bool:
        """Check if address falls within this memory region"""
        return self.start_address <= address <= self.end_address
    
    def get_offset(self, address: int) -> int:
        """Get offset within this region for the given address"""
        if not self.contains_address(address):
            raise ValueError(f"Address 0x{address:08x} not in region {self.name}")
        return address - self.start_address
    
    def read_bytes(self, address: int, size: int) -> bytes:
        """Read bytes from this memory region"""
        if not self.readable:
            raise PermissionError(f"Memory region {self.name} is not readable")
        
        offset = self.get_offset(address)
        if offset + size > self.size:
            raise ValueError(f"Read beyond region boundary in {self.name}")
        
        self.read_count += 1
        self.total_bytes_read += size
        
        return bytes(self.data[offset:offset + size])
    
    def write_bytes(self, address: int, data: bytes) -> None:
        """Write bytes to this memory region"""
        if not self.writable:
            raise PermissionError(f"Memory region {self.name} is not writable")
        
        offset = self.get_offset(address)
        if offset + len(data) > self.size:
            raise ValueError(f"Write beyond region boundary in {self.name}")
        
        self.data[offset:offset + len(data)] = data
        self.write_count += 1
        self.total_bytes_written += len(data)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory region statistics"""
        return {
            'name': self.name,
            'start_address': f"0x{self.start_address:08x}",
            'end_address': f"0x{self.end_address:08x}",
            'size': self.size,
            'read_count': self.read_count,
            'write_count': self.write_count,
            'total_bytes_read': self.total_bytes_read,
            'total_bytes_written': self.total_bytes_written,
            'utilization': (self.total_bytes_read + self.total_bytes_written) / self.size
        }


class MainMemory:
    """
    Main memory system with realistic timing and memory regions
    
    This class simulates the main memory system with configurable
    memory regions, access latencies, and bandwidth limitations.
    """
    
    def __init__(self, total_size: int = 64 * 1024 * 1024):  # 64MB default
        """
        Initialize main memory system
        
        Args:
            total_size: Total memory size in bytes
        """
        self.total_size = total_size
        self.regions: List[MemoryRegion] = []
        
        # Memory timing parameters
        self.base_latency_cycles = 100
        self.bandwidth_bytes_per_cycle = 8  # 8 bytes per cycle
        
        # Statistics
        self.total_accesses = 0
        self.total_read_accesses = 0
        self.total_write_accesses = 0
        self.total_cycles_spent = 0
        self.access_history: List[Dict[str, Any]] = []
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Initialize default memory regions
        self._initialize_default_regions()
        
        print(f"Initialized Main Memory: {total_size // (1024*1024)}MB")
    
    def _initialize_default_regions(self):
        """Initialize default memory regions for ARM system"""
        # ROM region (0x00000000 - 0x00100000) - 1MB
        self.add_region(0x00000000, 1024*1024, "ROM", 
                       readable=True, writable=False, executable=True, 
                       latency_cycles=80)
        
        # RAM region (0x20000000 - 0x24000000) - 64MB  
        remaining_size = min(self.total_size - 1024*1024, 63*1024*1024)
        self.add_region(0x20000000, remaining_size, "RAM",
                       readable=True, writable=True, executable=True,
                       latency_cycles=100)
        
        # I/O region (0x40000000 - 0x40001000) - 4KB
        self.add_region(0x40000000, 4096, "IO",
                       readable=True, writable=True, executable=False,
                       latency_cycles=200)
        
        # Stack region (0x60000000 - 0x60100000) - 1MB
        self.add_region(0x60000000, 1024*1024, "STACK",
                       readable=True, writable=True, executable=False,
                       latency_cycles=100)
    
    def add_region(self, start_address: int, size: int, name: str,
                   readable: bool = True, writable: bool = True,
                   executable: bool = False, latency_cycles: int = 100) -> MemoryRegion:
        """
        Add a memory region
        
        Args:
            start_address: Starting address of the region
            size: Size of the region in bytes
            name: Name identifier for the region
            readable: Whether region is readable
            writable: Whether region is writable
            executable: Whether region is executable
            latency_cycles: Access latency in cycles
            
        Returns:
            Created memory region
        """
        with self.lock:
            # Check for overlaps with existing regions
            for region in self.regions:
                if (start_address < region.end_address and 
                    start_address + size > region.start_address):
                    raise ValueError(f"Memory region {name} overlaps with {region.name}")
            
            region = MemoryRegion(start_address, size, name, readable, 
                                writable, executable, latency_cycles)
            self.regions.append(region)
            
            # Sort regions by start address
            self.regions.sort(key=lambda r: r.start_address)
            
            return region
    
    def _find_region(self, address: int) -> Optional[MemoryRegion]:
        """Find memory region containing the given address"""
        for region in self.regions:
            if region.contains_address(address):
                return region
        return None
    
    def read(self, address: int, size: int) -> bytes:
        """
        Read data from memory
        
        Args:
            address: Memory address to read from
            size: Number of bytes to read
            
        Returns:
            Data bytes read from memory
            
        Raises:
            ValueError: If address is invalid or read spans regions
            PermissionError: If region is not readable
        """
        with self.lock:
            if size <= 0:
                raise ValueError("Read size must be positive")
            
            # Record access
            start_time = time.time()
            self.total_accesses += 1
            self.total_read_accesses += 1
            
            # Find memory region
            region = self._find_region(address)
            if not region:
                raise ValueError(f"Invalid memory address: 0x{address:08x}")
            
            # Check if read spans multiple regions
            if not region.contains_address(address + size - 1):
                raise ValueError(f"Read spans multiple memory regions at 0x{address:08x}")
            
            # Perform read
            data = region.read_bytes(address, size)
            
            # Calculate timing
            cycles = self._calculate_access_cycles(size, region.latency_cycles)
            self.total_cycles_spent += cycles
            
            # Record access history
            access_record = {
                'type': MemoryAccessType.READ.value,
                'address': f"0x{address:08x}",
                'size': size,
                'region': region.name,
                'cycles': cycles,
                'timestamp': start_time
            }
            self.access_history.append(access_record)
            
            # Keep history manageable
            if len(self.access_history) > 1000:
                self.access_history = self.access_history[-500:]
            
            return data
    
    def write(self, address: int, data: bytes) -> None:
        """
        Write data to memory
        
        Args:
            address: Memory address to write to
            data: Data bytes to write
            
        Raises:
            ValueError: If address is invalid or write spans regions
            PermissionError: If region is not writable
        """
        with self.lock:
            if len(data) == 0:
                raise ValueError("Write data cannot be empty")
            
            # Record access
            start_time = time.time()
            self.total_accesses += 1
            self.total_write_accesses += 1
            
            # Find memory region
            region = self._find_region(address)
            if not region:
                raise ValueError(f"Invalid memory address: 0x{address:08x}")
            
            # Check if write spans multiple regions
            if not region.contains_address(address + len(data) - 1):
                raise ValueError(f"Write spans multiple memory regions at 0x{address:08x}")
            
            # Perform write
            region.write_bytes(address, data)
            
            # Calculate timing
            cycles = self._calculate_access_cycles(len(data), region.latency_cycles)
            self.total_cycles_spent += cycles
            
            # Record access history
            access_record = {
                'type': MemoryAccessType.WRITE.value,
                'address': f"0x{address:08x}",
                'size': len(data),
                'region': region.name,
                'cycles': cycles,
                'timestamp': start_time
            }
            self.access_history.append(access_record)
            
            # Keep history manageable
            if len(self.access_history) > 1000:
                self.access_history = self.access_history[-500:]
    
    def read_word(self, address: int) -> int:
        """Read a 32-bit word from memory (little-endian)"""
        data = self.read(address, 4)
        return int.from_bytes(data, byteorder='little')
    
    def write_word(self, address: int, value: int) -> None:
        """Write a 32-bit word to memory (little-endian)"""
        data = value.to_bytes(4, byteorder='little')
        self.write(address, data)
    
    def read_halfword(self, address: int) -> int:
        """Read a 16-bit halfword from memory (little-endian)"""
        data = self.read(address, 2)
        return int.from_bytes(data, byteorder='little')
    
    def write_halfword(self, address: int, value: int) -> None:
        """Write a 16-bit halfword to memory (little-endian)"""
        data = value.to_bytes(2, byteorder='little')
        self.write(address, data)
    
    def read_byte(self, address: int) -> int:
        """Read a single byte from memory"""
        data = self.read(address, 1)
        return data[0]
    
    def write_byte(self, address: int, value: int) -> None:
        """Write a single byte to memory"""
        data = bytes([value & 0xFF])
        self.write(address, data)
    
    def _calculate_access_cycles(self, size: int, base_latency: int) -> int:
        """Calculate memory access cycles based on size and latency"""
        # Base latency + bandwidth-limited transfer time
        transfer_cycles = max(1, (size + self.bandwidth_bytes_per_cycle - 1) // self.bandwidth_bytes_per_cycle)
        return base_latency + transfer_cycles
    
    def load_program(self, program_data: bytes, start_address: int = 0x20000000) -> None:
        """
        Load a program into memory
        
        Args:
            program_data: Program binary data
            start_address: Address to load program at
        """
        if len(program_data) == 0:
            return
        
        print(f"Loading program: {len(program_data)} bytes at 0x{start_address:08x}")
        self.write(start_address, program_data)
    
    def dump_region(self, region_name: str, start_offset: int = 0, 
                   size: int = 256) -> str:
        """
        Dump memory region contents as hex
        
        Args:
            region_name: Name of region to dump
            start_offset: Offset within region to start dump
            size: Number of bytes to dump
            
        Returns:
            Formatted hex dump string
        """
        # Find region
        region = None
        for r in self.regions:
            if r.name == region_name:
                region = r
                break
        
        if not region:
            return f"Region '{region_name}' not found"
        
        # Limit size to region bounds
        actual_size = min(size, region.size - start_offset)
        if actual_size <= 0:
            return f"Invalid offset or size for region '{region_name}'"
        
        # Get data
        address = region.start_address + start_offset
        try:
            data = self.read(address, actual_size)
        except Exception as e:
            return f"Error reading region '{region_name}': {e}"
        
        # Format as hex dump
        lines = []
        lines.append(f"Memory dump: {region_name} (0x{address:08x} - 0x{address + actual_size - 1:08x})")
        lines.append("-" * 60)
        
        for i in range(0, len(data), 16):
            addr = address + i
            chunk = data[i:i+16]
            
            # Hex bytes
            hex_str = ' '.join(f"{b:02x}" for b in chunk)
            hex_str = hex_str.ljust(48)  # Pad to align ASCII
            
            # ASCII representation
            ascii_str = ''.join(chr(b) if 32 <= b <= 126 else '.' for b in chunk)
            
            lines.append(f"0x{addr:08x}: {hex_str} |{ascii_str}|")
        
        return '\n'.join(lines)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics"""
        with self.lock:
            region_stats = [region.get_statistics() for region in self.regions]
            
            return {
                'total_size': self.total_size,
                'total_accesses': self.total_accesses,
                'read_accesses': self.total_read_accesses,
                'write_accesses': self.total_write_accesses,
                'total_cycles': self.total_cycles_spent,
                'average_cycles_per_access': (self.total_cycles_spent / max(1, self.total_accesses)),
                'regions': region_stats,
                'recent_accesses': self.access_history[-20:]  # Last 20 accesses
            }
    
    def reset_statistics(self):
        """Reset all memory statistics"""
        with self.lock:
            self.total_accesses = 0
            self.total_read_accesses = 0
            self.total_write_accesses = 0
            self.total_cycles_spent = 0
            self.access_history.clear()
            
            # Reset region statistics
            for region in self.regions:
                region.read_count = 0
                region.write_count = 0
                region.total_bytes_read = 0
                region.total_bytes_written = 0
    
    def print_status(self):
        """Print current memory system status"""
        stats = self.get_statistics()
        
        print(f"\n=== Memory System Status ===")
        print(f"Total Size: {self.total_size // (1024*1024)}MB")
        print(f"Total Accesses: {stats['total_accesses']}")
        print(f"Read/Write Ratio: {stats['read_accesses']}/{stats['write_accesses']}")
        print(f"Average Access Time: {stats['average_cycles_per_access']:.2f} cycles")
        print(f"Total Cycles: {stats['total_cycles']}")
        
        print(f"\nMemory Regions:")
        for region_stat in stats['regions']:
            print(f"  {region_stat['name']}: {region_stat['start_address']} - {region_stat['end_address']}")
            print(f"    Reads: {region_stat['read_count']}, Writes: {region_stat['write_count']}")
            print(f"    Utilization: {region_stat['utilization']:.4f}")
    
    def __str__(self) -> str:
        return f"MainMemory({self.total_size // (1024*1024)}MB, {len(self.regions)} regions)"


# Example usage and testing
if __name__ == "__main__":
    print("Testing Memory System\n")
    
    # Create memory system
    memory = MainMemory(16 * 1024 * 1024)  # 16MB
    
    # Test basic operations
    print("Testing basic memory operations...")
    
    # Write some data
    test_data = b"Hello, ARM CPU Simulator!"
    memory.write(0x20000000, test_data)
    
    # Read it back
    read_data = memory.read(0x20000000, len(test_data))
    print(f"Write/Read test: {'PASS' if read_data == test_data else 'FAIL'}")
    
    # Test word operations
    memory.write_word(0x20001000, 0xDEADBEEF)
    word_value = memory.read_word(0x20001000)
    print(f"Word test: {'PASS' if word_value == 0xDEADBEEF else 'FAIL'}")
    
    # Load a simple program
    program = b"\x01\x02\x03\x04" * 100  # Simple test program
    memory.load_program(program, 0x20002000)
    
    # Dump a small portion of memory
    print("\nMemory dump:")
    print(memory.dump_region("RAM", 0x2000, 64))
    
    # Print statistics
    memory.print_status()
