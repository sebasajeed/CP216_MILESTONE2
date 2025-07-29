"""
ARM CPU Simulator Main Interface

This module provides the main interface for the ARM CPU simulator,
integrating all components including CPU, memory, cache, and instruction
set architecture. It serves as the entry point for running simulations.

Author: CPU Simulator Project
Date: 2025
"""

import sys
import os
import time
import argparse
from typing import Dict, List, Optional, Any

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cpu.cpu import ARMCPU, CPUState
from cpu.register_file import RegisterFile, ProcessorMode
from cpu.alu import ALU
from memory.memory import MainMemory
from cache.cache import Cache
from cache.cache_config import CacheConfigPresets
from isa.arm_instructions import ARMAssembler, ARMInstructionDecoder, ARMCondition


class CPUSimulator:
    """
    Main ARM CPU Simulator class
    
    Integrates all components and provides a high-level interface
    for running ARM programs and analyzing performance.
    """
    
    def __init__(self, memory_size: int = 64 * 1024 * 1024,
                 enable_cache: bool = True,
                 cache_config: str = "l1_data"):
        """
        Initialize CPU simulator
        
        Args:
            memory_size: Main memory size in bytes
            enable_cache: Whether to enable cache simulation
            cache_config: Cache configuration preset name
        """
        print("Initializing ARM CPU Simulator...")
        
        # Initialize memory system
        self.memory = MainMemory(memory_size)
        print(f"Memory: {memory_size // (1024*1024)}MB")
        
        # Initialize cache system
        self.cache = None
        if enable_cache:
            if cache_config == "l1_data":
                config = CacheConfigPresets.l1_data_cache()
            elif cache_config == "l1_instruction":
                config = CacheConfigPresets.l1_instruction_cache()
            elif cache_config == "l2_unified":
                config = CacheConfigPresets.l2_unified_cache()
            elif cache_config == "direct_mapped":
                config = CacheConfigPresets.direct_mapped_cache()
            else:
                config = CacheConfigPresets.l1_data_cache()  # Default
            
            self.cache = Cache(config, "L1-Cache")
            print(f"Cache: {config.size // 1024}KB {config.associativity}-way")
        
        # Initialize CPU
        self.cpu = ARMCPU(self.memory, self.cache)
        
        # Simulation state
        self.program_loaded = False
        self.program_start_address = 0x20000000
        self.simulation_stats = {}
        
        print("ARM CPU Simulator initialized successfully\n")
    
    def load_program_binary(self, binary_data: bytes, start_address: int = None) -> None:
        """
        Load binary program into memory
        
        Args:
            binary_data: Program binary data
            start_address: Address to load program at (default: 0x20000000)
        """
        if start_address is None:
            start_address = self.program_start_address
        
        print(f"Loading program: {len(binary_data)} bytes at 0x{start_address:08x}")
        
        # Load program into memory
        self.memory.load_program(binary_data, start_address)
        
        # Set CPU initial state
        self.cpu.registers.set_pc(start_address)
        self.cpu.registers.set_sp(0x60100000)  # Set stack pointer
        
        self.program_loaded = True
        self.program_start_address = start_address
        
        print("Program loaded successfully")
    
    def load_program_assembly(self, assembly_code: str) -> None:
        """
        Load assembly program
        
        Args:
            assembly_code: ARM assembly code as string
        """
        # This is a simplified assembly loader
        # In a real implementation, you'd have a full assembler
        
        asm = ARMAssembler()
        asm.set_origin(self.program_start_address)
        
        # Parse and assemble (very basic implementation)
        lines = assembly_code.strip().split('\n')
        for line in lines:
            line = line.strip()
            if not line or line.startswith(';'):
                continue
            
            # Very basic instruction parsing
            parts = line.replace(',', ' ').split()
            if not parts:
                continue
            
            instr = parts[0].lower()
            
            # Skip labels (lines ending with ':')
            if line.strip().endswith(':'):
                continue
            
            try:
                if instr == 'mov' and len(parts) >= 3:
                    rd = self._parse_register(parts[1])
                    if parts[2].startswith('#'):
                        imm_str = parts[2][1:]  # Remove #
                        # Handle hex values
                        if imm_str.startswith('0x') or imm_str.startswith('0X'):
                            imm = int(imm_str, 16)
                        else:
                            imm = int(imm_str)
                        asm.mov(rd, imm)
                    else:
                        rs = self._parse_register(parts[2])
                        # For register operands, use tuple format (reg, shift_type, shift_amount)
                        asm.mov(rd, (rs, 0, 0))
                
                elif instr == 'add' and len(parts) >= 4:
                    rd = self._parse_register(parts[1])
                    rn = self._parse_register(parts[2])
                    if parts[3].startswith('#'):
                        imm_str = parts[3][1:]
                        if imm_str.startswith('0x') or imm_str.startswith('0X'):
                            imm = int(imm_str, 16)
                        else:
                            imm = int(imm_str)
                        asm.add(rd, rn, imm)
                    else:
                        rs = self._parse_register(parts[3])
                        # For register operands, use tuple format (reg, shift_type, shift_amount)
                        asm.add(rd, rn, (rs, 0, 0))
                
                elif instr == 'sub' and len(parts) >= 4:
                    rd = self._parse_register(parts[1])
                    rn = self._parse_register(parts[2])
                    if parts[3].startswith('#'):
                        imm_str = parts[3][1:]
                        if imm_str.startswith('0x') or imm_str.startswith('0X'):
                            imm = int(imm_str, 16)
                        else:
                            imm = int(imm_str)
                        asm.sub(rd, rn, imm)
                    else:
                        rs = self._parse_register(parts[3])
                        # For register operands, use tuple format (reg, shift_type, shift_amount)
                        asm.sub(rd, rn, (rs, 0, 0))
                
                elif instr == 'b':
                    # Simple infinite loop for now
                    asm.b("loop")
                
                elif instr == 'nop':
                    asm.nop()
                
                elif instr in ['cmp', 'str', 'ldr', 'blt', 'bgt', 'beq', 'bne', 'ble']:
                    # Unsupported instructions - insert NOP for now
                    print(f"Note: {instr} instruction simplified to NOP")
                    asm.nop()
                
                else:
                    if instr not in ['b']:  # Don't warn for 'b' as it's handled above
                        print(f"Warning: Unsupported instruction: {line}")
                    asm.nop()  # Insert NOP for unsupported instructions
            
            except Exception as e:
                print(f"Error assembling '{line}': {e}")
                asm.nop()  # Insert NOP on error
        
        # Assemble and load
        binary_code = asm.assemble()
        self.load_program_binary(binary_code)
        
        # Print program info
        info = asm.get_program_info()
        print(f"Assembled program: {info['instructions']} instructions, {info['size_bytes']} bytes")
    
    def _parse_register(self, reg_str: str) -> int:
        """Parse register string (e.g., 'R0', 'r1') to register number"""
        reg_str = reg_str.upper().replace('R', '')
        return int(reg_str)
    
    def run_simulation(self, max_cycles: Optional[int] = None,
                      max_instructions: Optional[int] = None,
                      print_progress: bool = True) -> Dict[str, Any]:
        """
        Run CPU simulation
        
        Args:
            max_cycles: Maximum cycles to simulate
            max_instructions: Maximum instructions to execute
            print_progress: Whether to print progress updates
            
        Returns:
            Simulation statistics
        """
        if not self.program_loaded:
            raise RuntimeError("No program loaded. Use load_program_binary() or load_program_assembly() first.")
        
        print("Starting simulation...")
        if print_progress:
            print(f"Initial PC: 0x{self.cpu.registers.get_pc():08x}")
            print(f"Initial SP: 0x{self.cpu.registers.get_sp():08x}")
        
        # Reset statistics
        self.cpu.alu.reset_statistics()
        self.memory.reset_statistics()
        if self.cache:
            self.cache.reset_statistics()
        
        # Run simulation
        start_time = time.time()
        execution_stats = self.cpu.run(max_cycles, max_instructions)
        end_time = time.time()
        
        # Collect comprehensive statistics
        cpu_stats = self.cpu.get_statistics()
        memory_stats = self.memory.get_statistics()
        cache_stats = self.cache.get_statistics() if self.cache else None
        
        self.simulation_stats = {
            'execution_time': end_time - start_time,
            'cpu': cpu_stats,
            'memory': memory_stats,
            'cache': cache_stats,
            'execution': execution_stats
        }
        
        print(f"\\nSimulation completed in {end_time - start_time:.3f} seconds")
        
        return self.simulation_stats
    
    def step_simulation(self, steps: int = 1) -> bool:
        """
        Step simulation by specified number of cycles
        
        Args:
            steps: Number of cycles to step
            
        Returns:
            True if CPU is still running
        """
        if not self.program_loaded:
            raise RuntimeError("No program loaded")
        
        for _ in range(steps):
            if not self.cpu.step():
                return False
        return True
    
    def reset_simulation(self):
        """Reset simulation to initial state"""
        print("Resetting simulation...")
        
        # Reset CPU
        self.cpu.reset()
        
        # Reset memory statistics (but keep program data)
        self.memory.reset_statistics()
        
        # Reset cache
        if self.cache:
            self.cache.reset_statistics()
        
        # Restore program counter
        if self.program_loaded:
            self.cpu.registers.set_pc(self.program_start_address)
        
        print("Simulation reset completed")
    
    def print_status(self):
        """Print comprehensive simulator status"""
        print("\\n" + "="*60)
        print("ARM CPU SIMULATOR STATUS")
        print("="*60)
        
        # CPU status
        self.cpu.print_status()
        
        # Memory status
        self.memory.print_status()
        
        # Cache status
        if self.cache:
            self.cache.print_status()
        
        print("="*60)
    
    def dump_memory(self, start_address: int, size: int = 256) -> str:
        """
        Dump memory contents
        
        Args:
            start_address: Starting address
            size: Number of bytes to dump
            
        Returns:
            Formatted memory dump string
        """
        # Find appropriate memory region
        region_name = "RAM"  # Default
        for region in self.memory.regions:
            if region.contains_address(start_address):
                region_name = region.name
                break
        
        offset = start_address - self.memory.regions[0].start_address
        return self.memory.dump_region(region_name, offset, size)
    
    def get_instruction_trace(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent instruction execution trace
        
        Args:
            count: Number of recent instructions to return
            
        Returns:
            List of instruction trace entries
        """
        return self.cpu.instruction_trace[-count:] if self.cpu.instruction_trace else []
    
    def analyze_performance(self) -> Dict[str, Any]:
        """
        Analyze simulation performance
        
        Returns:
            Performance analysis results
        """
        if not self.simulation_stats:
            return {"error": "No simulation data available"}
        
        cpu_stats = self.simulation_stats['cpu']
        memory_stats = self.simulation_stats['memory']
        cache_stats = self.simulation_stats.get('cache')
        
        analysis = {
            'performance_metrics': {
                'cpi': cpu_stats['cpi'],
                'instruction_throughput': cpu_stats['instruction_count'] / self.simulation_stats['execution_time'],
                'cycle_throughput': cpu_stats['cycle_count'] / self.simulation_stats['execution_time'],
                'memory_efficiency': memory_stats['total_accesses'] / max(cpu_stats['cycle_count'], 1)
            },
            'bottlenecks': [],
            'recommendations': []
        }
        
        # Identify bottlenecks
        if cpu_stats['cpi'] > 2.0:
            analysis['bottlenecks'].append("High CPI indicates CPU stalls")
        
        if cache_stats and cache_stats['hit_rate'] < 0.8:
            analysis['bottlenecks'].append(f"Low cache hit rate: {cache_stats['hit_rate']:.3f}")
        
        if cpu_stats['branch_mispredict_rate'] > 0.1:
            analysis['bottlenecks'].append(f"High branch mispredict rate: {cpu_stats['branch_mispredict_rate']:.3f}")
        
        # Generate recommendations
        if cache_stats and cache_stats['hit_rate'] < 0.9:
            analysis['recommendations'].append("Consider larger cache or better replacement policy")
        
        if cpu_stats['cpi'] > 1.5:
            analysis['recommendations'].append("Optimize code to reduce memory accesses and branches")
        
        return analysis
    
    def save_statistics(self, filename: str):
        """Save simulation statistics to file"""
        import json
        
        if not self.simulation_stats:
            print("No statistics to save")
            return
        
        # Convert any non-serializable objects to strings
        def make_serializable(obj):
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            elif hasattr(obj, '__dict__'):
                return str(obj)
            else:
                return obj
        
        serializable_stats = make_serializable(self.simulation_stats)
        
        with open(filename, 'w') as f:
            json.dump(serializable_stats, f, indent=2)
        
        print(f"Statistics saved to {filename}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current simulation statistics"""
        if not self.simulation_stats:
            # Return basic stats if no simulation has been run
            return {
                'cycles_executed': 0,
                'instructions_executed': 0,
                'cache_stats': {
                    'l1_data': {'hits': 0, 'misses': 0, 'hit_rate': 0.0}
                },
                'cpu_stats': {},
                'memory_stats': {},
                'execution_time': 0.0
            }
        
        # Format stats for easy access
        stats = {
            'cycles_executed': self.simulation_stats.get('execution', {}).get('cycles_executed', 0),
            'instructions_executed': self.simulation_stats.get('execution', {}).get('instructions_executed', 0),
            'execution_time': self.simulation_stats.get('execution_time', 0.0),
            'cpu_stats': self.simulation_stats.get('cpu', {}),
            'memory_stats': self.simulation_stats.get('memory', {}),
            'cache_stats': self.simulation_stats.get('cache', {})
        }
        
        return stats


def create_sample_program() -> str:
    """Create a sample ARM assembly program for testing"""
    return """
        ; Sample ARM assembly program
        ; Calculates factorial of 5
        
        MOV R0, #5      ; Load 5 into R0
        MOV R1, #1      ; Initialize result to 1
        
        ; Factorial loop
        MOV R2, R0      ; Copy counter
        ADD R1, R1, R2  ; Simplified: just add instead of multiply
        SUB R0, R0, #1  ; Decrement counter
        
        ; Simple termination
        MOV R3, #0      ; Set R3 to 0
        ADD R3, R3, #1  ; Increment R3
        B .             ; Infinite loop (branch to self)
    """


def main():
    """Main function for command-line interface"""
    parser = argparse.ArgumentParser(description="ARM CPU Simulator")
    parser.add_argument("--memory-size", type=int, default=64*1024*1024,
                       help="Memory size in bytes (default: 64MB)")
    parser.add_argument("--no-cache", action="store_true",
                       help="Disable cache simulation")
    parser.add_argument("--cache-config", default="l1_data",
                       choices=["l1_data", "l1_instruction", "l2_unified", "direct_mapped"],
                       help="Cache configuration preset")
    parser.add_argument("--max-cycles", type=int,
                       help="Maximum cycles to simulate")
    parser.add_argument("--max-instructions", type=int,
                       help="Maximum instructions to execute")
    parser.add_argument("--program", type=str,
                       help="Assembly program file to load")
    parser.add_argument("--stats-file", type=str,
                       help="File to save simulation statistics")
    
    args = parser.parse_args()
    
    # Create simulator
    simulator = CPUSimulator(
        memory_size=args.memory_size,
        enable_cache=not args.no_cache,
        cache_config=args.cache_config
    )
    
    # Load program
    if args.program:
        try:
            with open(args.program, 'r') as f:
                program_code = f.read()
            simulator.load_program_assembly(program_code)
        except FileNotFoundError:
            print("Program file not found, using sample program")
            simulator.load_program_assembly(create_sample_program())
    else:
        print("No program specified, using sample program")
        simulator.load_program_assembly(create_sample_program())
    
    # Run simulation
    try:
        stats = simulator.run_simulation(
            max_cycles=args.max_cycles,
            max_instructions=args.max_instructions
        )
        
        # Print results
        simulator.print_status()
        
        # Performance analysis
        analysis = simulator.analyze_performance()
        print("\\n" + "="*60)
        print("PERFORMANCE ANALYSIS")
        print("="*60)
        print(f"CPI: {analysis['performance_metrics']['cpi']:.3f}")
        print(f"Instruction Throughput: {analysis['performance_metrics']['instruction_throughput']:.0f} inst/sec")
        print(f"Cycle Throughput: {analysis['performance_metrics']['cycle_throughput']:.0f} cycles/sec")
        
        if analysis['bottlenecks']:
            print("\\nBottlenecks identified:")
            for bottleneck in analysis['bottlenecks']:
                print(f"  - {bottleneck}")
        
        if analysis['recommendations']:
            print("\\nRecommendations:")
            for rec in analysis['recommendations']:
                print(f"  - {rec}")
        
        # Save statistics if requested
        if args.stats_file:
            simulator.save_statistics(args.stats_file)
    
    except KeyboardInterrupt:
        print("\\nSimulation interrupted by user")
        simulator.print_status()
    
    except Exception as e:
        print(f"Simulation error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
