import time
import csv
import os
import sys
from typing import Dict, List, Any, Tuple
import statistics

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cpu_simulator.main import CPUSimulator
from cache.cache_config import CacheConfigPresets


class BenchmarkSuite:
    """Comprehensive benchmarking suite for CPU simulator"""
    
    def __init__(self):
        self.results = []
        self.test_programs = {
            'arithmetic': self._create_arithmetic_program(),
            'memory_intensive': self._create_memory_intensive_program(),
            'branch_intensive': self._create_branch_intensive_program(),
            'mixed_workload': self._create_mixed_workload_program(),
            'bubble_sort': self._create_bubble_sort_program()
        }
    
    def _create_arithmetic_program(self) -> str:
        """Create arithmetic-intensive test program"""
        return """
        ; Arithmetic benchmark - compute factorial-like operation
        MOV R0, #10     ; Counter
        MOV R1, #1      ; Accumulator
        MOV R2, #2      ; Multiplier base
        
        ; Simplified arithmetic loop (no branches for now)
        ADD R1, R1, R0  ; Add counter to accumulator
        ADD R1, R1, R2  ; Add multiplier
        SUB R0, R0, #1  ; Decrement counter
        ADD R1, R1, R0  ; More arithmetic
        SUB R0, R0, #1  ; More decrement
        ADD R1, R1, R2  ; More addition
        
        ; Final computation
        MOV R4, #100
        ADD R1, R1, R4
        SUB R1, R1, #50
        
        ; End with infinite loop
        MOV R5, #0
        ADD R5, R5, #1
        """
    
    def _create_memory_intensive_program(self) -> str:
        """Create memory-intensive test program"""
        return """
        ; Memory benchmark - simplified without complex addressing
        MOV R0, #0x2100 ; Array base address
        MOV R1, #20     ; Array size
        MOV R2, #0      ; Index
        MOV R3, #1      ; Value to store
        
        ; Simulate array operations with simple arithmetic
        ; (STR/LDR not fully implemented, so use arithmetic instead)
        ADD R4, R0, R2  ; Calculate address
        ADD R3, R3, #1  ; Increment value
        ADD R2, R2, #1  ; Increment index
        ADD R4, R0, R2  ; Calculate next address
        ADD R3, R3, #1  ; Increment value
        ADD R2, R2, #1  ; Increment index
        
        ; More memory-like operations
        MOV R5, #0x2200
        ADD R6, R5, R2
        SUB R6, R6, R1
        ADD R4, R4, R6
        
        ; Sum simulation
        MOV R2, #0      ; Reset index
        MOV R4, #0      ; Sum accumulator
        ADD R4, R4, R3  ; Add to sum
        ADD R2, R2, #1  ; Increment index
        ADD R4, R4, R3  ; Add to sum
        ADD R2, R2, #1  ; Increment index
        """
    
    def _create_branch_intensive_program(self) -> str:
        """Create branch-intensive test program"""
        return """
        ; Branch benchmark - nested loops with conditions
        MOV R0, #5      ; Outer loop counter
        
        outer_loop:
        MOV R1, #8      ; Inner loop counter
        
        inner_loop:
        CMP R1, #4      ; Compare with middle value
        BEQ equal_case  ; Branch if equal
        BGT greater_case; Branch if greater
        
        less_case:
        ADD R2, R2, #1  ; Increment R2
        B continue_inner
        
        equal_case:
        ADD R3, R3, #1  ; Increment R3
        B continue_inner
        
        greater_case:
        ADD R4, R4, #1  ; Increment R4
        
        continue_inner:
        SUB R1, R1, #1  ; Decrement inner counter
        CMP R1, #0      ; Check if done
        BNE inner_loop  ; Continue inner loop
        
        SUB R0, R0, #1  ; Decrement outer counter
        CMP R0, #0      ; Check if done
        BNE outer_loop  ; Continue outer loop
        
        B .             ; End
        """
    
    def _create_mixed_workload_program(self) -> str:
        """Create mixed workload test program"""
        return """
        ; Mixed workload - arithmetic, memory simulation, and simple control
        MOV R0, #0x2200 ; Data base address
        MOV R1, #10     ; Loop counter
        MOV R2, #0      ; Array index
        
        ; Arithmetic operations
        MOV R3, R1      ; Copy counter
        ADD R3, R3, R3  ; Double it
        SUB R4, R3, R1  ; Subtract original
        
        ; Memory address calculations (simulating memory ops)
        ADD R5, R0, R2  ; Calculate address
        ADD R5, R5, R4  ; Add offset
        
        ; More arithmetic to simulate processing
        ADD R6, R4, R1  ; Add values
        SUB R6, R6, #5  ; Subtract constant
        ADD R4, R4, R6  ; Combine results
        
        ; Simulate loop iterations with arithmetic
        ADD R2, R2, #1  ; Increment index
        SUB R1, R1, #1  ; Decrement counter
        ADD R7, R1, R2  ; Combine loop variables
        SUB R7, R7, #3  ; More processing
        
        ; Final calculations
        MOV R8, #100
        ADD R4, R4, R8
        SUB R4, R4, R2
        """
    
    def _create_bubble_sort_program(self) -> str:
        """Create bubble sort test program"""
        return """
        ; Bubble sort benchmark
        MOV R0, #0x2300 ; Array base address
        MOV R1, #8      ; Array size
        MOV R2, #0      ; Index i
        
        ; Initialize array with reverse order
        MOV R3, R1      ; Start with size
        init_array:
        STR R3, [R0, R2, LSL #2]  ; Store decreasing values
        SUB R3, R3, #1            ; Decrement value
        ADD R2, R2, #1            ; Increment index
        CMP R2, R1                ; Check if done
        BLT init_array            ; Continue if not done
        
        ; Bubble sort algorithm
        SUB R1, R1, #1  ; n-1 for outer loop
        MOV R2, #0      ; i = 0
        
        outer_bubble:
        SUB R6, R1, R2  ; n-1-i for inner loop
        MOV R3, #0      ; j = 0
        
        inner_bubble:
        LDR R4, [R0, R3, LSL #2]      ; Load arr[j]
        ADD R7, R3, #1                ; j+1
        LDR R5, [R0, R7, LSL #2]      ; Load arr[j+1]
        
        CMP R4, R5                    ; Compare arr[j] with arr[j+1]
        BLE no_swap                   ; Branch if arr[j] <= arr[j+1]
        
        ; Swap elements
        STR R5, [R0, R3, LSL #2]      ; arr[j] = arr[j+1]
        STR R4, [R0, R7, LSL #2]      ; arr[j+1] = arr[j]
        
        no_swap:
        ADD R3, R3, #1                ; j++
        CMP R3, R6                    ; Compare j with n-1-i
        BLT inner_bubble              ; Continue inner loop
        
        ADD R2, R2, #1                ; i++
        CMP R2, R1                    ; Compare i with n-1
        BLT outer_bubble              ; Continue outer loop
        
        B .                           ; End
        """
    
    def run_benchmark(self, program_name: str, config_name: str, 
                     cache_enabled: bool = True, max_cycles: int = 10000) -> Dict[str, Any]:
        """
        Run a single benchmark
        
        Args:
            program_name: Name of test program
            config_name: Cache configuration name
            cache_enabled: Whether to enable cache
            max_cycles: Maximum cycles to simulate
            
        Returns:
            Benchmark results dictionary
        """
        print(f"Running benchmark: {program_name} with {config_name}")
        
        # Create simulator
        simulator = CPUSimulator(
            memory_size=32*1024*1024,  # 32MB
            enable_cache=cache_enabled,
            cache_config=config_name
        )
        
        # Load program
        if program_name not in self.test_programs:
            raise ValueError(f"Unknown program: {program_name}")
        
        program_code = self.test_programs[program_name]
        simulator.load_program_assembly(program_code)
        
        # Run simulation
        start_time = time.time()
        stats = simulator.run_simulation(max_cycles=max_cycles, print_progress=False)
        end_time = time.time()
        
        # Extract key metrics
        cpu_stats = stats['cpu']
        memory_stats = stats['memory']
        cache_stats = stats.get('cache')
        
        result = {
            'program': program_name,
            'config': config_name,
            'cache_enabled': cache_enabled,
            'simulation_time': end_time - start_time,
            'cycles': cpu_stats['cycle_count'],
            'instructions': cpu_stats['instruction_count'],
            'cpi': cpu_stats['cpi'],
            'memory_accesses': memory_stats['total_accesses'],
            'cache_hit_rate': cache_stats['hit_rate'] if cache_stats else 0.0,
            'cache_miss_rate': 1.0 - cache_stats['hit_rate'] if cache_stats else 1.0,
            'branch_count': cpu_stats['branch_count'],
            'branch_mispredict_rate': cpu_stats['branch_mispredict_rate'],
            'instruction_mix': cpu_stats['instruction_mix']
        }
        
        return result
    
    def run_configuration_comparison(self, program_name: str = 'mixed_workload',
                                   max_cycles: int = 5000) -> List[Dict[str, Any]]:
        """
        Compare different cache configurations on the same program
        
        Args:
            program_name: Test program to use
            max_cycles: Maximum cycles per test
            
        Returns:
            List of benchmark results
        """
        print(f"\\nRunning configuration comparison for {program_name}")
        print("="*60)
        
        configurations = [
            ('no_cache', False),
            ('direct_mapped', True),
            ('l1_data', True),
            ('l2_unified', True)
        ]
        
        results = []
        
        for config_name, cache_enabled in configurations:
            try:
                result = self.run_benchmark(program_name, config_name if cache_enabled else 'l1_data', 
                                          cache_enabled, max_cycles)
                results.append(result)
                
                # Print summary
                print(f"{config_name:15} | CPI: {result['cpi']:5.2f} | "
                      f"Hit Rate: {result['cache_hit_rate']:5.3f} | "
                      f"Cycles: {result['cycles']:6d}")
                
            except Exception as e:
                print(f"Error running {config_name}: {e}")
        
        return results
    
    def run_program_comparison(self, config_name: str = 'l1_data',
                             max_cycles: int = 5000) -> List[Dict[str, Any]]:
        """
        Compare different programs on the same configuration
        
        Args:
            config_name: Cache configuration to use
            max_cycles: Maximum cycles per test
            
        Returns:
            List of benchmark results
        """
        print(f"\\nRunning program comparison with {config_name}")
        print("="*60)
        
        results = []
        
        for program_name in self.test_programs.keys():
            try:
                result = self.run_benchmark(program_name, config_name, True, max_cycles)
                results.append(result)
                
                # Print summary
                print(f"{program_name:15} | CPI: {result['cpi']:5.2f} | "
                      f"Instructions: {result['instructions']:5d} | "
                      f"Hit Rate: {result['cache_hit_rate']:5.3f}")
                
            except Exception as e:
                print(f"Error running {program_name}: {e}")
        
        return results
    
    def run_full_benchmark_suite(self, max_cycles: int = 3000) -> Dict[str, List[Dict[str, Any]]]:
        """
        Run complete benchmark suite
        
        Args:
            max_cycles: Maximum cycles per test
            
        Returns:
            Dictionary of all benchmark results
        """
        print("\\n" + "="*80)
        print("ARM CPU SIMULATOR BENCHMARK SUITE")
        print("="*80)
        
        all_results = {}
        
        # Configuration comparison
        all_results['config_comparison'] = self.run_configuration_comparison(
            'mixed_workload', max_cycles
        )
        
        # Program comparison
        all_results['program_comparison'] = self.run_program_comparison(
            'l1_data', max_cycles
        )
        
        # Individual program tests with different configs
        for program in ['arithmetic', 'memory_intensive', 'branch_intensive']:
            results = []
            for config in ['direct_mapped', 'l1_data']:
                try:
                    result = self.run_benchmark(program, config, True, max_cycles)
                    results.append(result)
                except Exception as e:
                    print(f"Error in {program} with {config}: {e}")
            
            all_results[f'{program}_comparison'] = results
        
        return all_results
    
    def generate_report(self, results: Dict[str, List[Dict[str, Any]]], 
                       output_file: str = 'benchmark_report.txt') -> None:
        """
        Generate comprehensive benchmark report
        
        Args:
            results: Benchmark results dictionary
            output_file: Output file path
        """
        print(f"\\nGenerating benchmark report: {output_file}")
        
        with open(output_file, 'w') as f:
            f.write("ARM CPU SIMULATOR BENCHMARK REPORT\\n")
            f.write("="*50 + "\\n\\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\\n\\n")
            
            # Configuration comparison analysis
            if 'config_comparison' in results:
                f.write("CACHE CONFIGURATION COMPARISON\\n")
                f.write("-"*30 + "\\n")
                
                config_results = results['config_comparison']
                
                f.write(f"{'Configuration':<15} {'CPI':<8} {'Hit Rate':<10} {'Cycles':<8} {'Instrs':<8}\\n")
                f.write("-"*55 + "\\n")
                
                for result in config_results:
                    f.write(f"{result['config']:<15} "
                           f"{result['cpi']:<8.2f} "
                           f"{result['cache_hit_rate']:<10.3f} "
                           f"{result['cycles']:<8d} "
                           f"{result['instructions']:<8d}\\n")
                
                # Find best configuration
                best_config = min(config_results, key=lambda x: x['cpi'])
                f.write(f"\\nBest Configuration: {best_config['config']} (CPI: {best_config['cpi']:.2f})\\n\\n")
            
            # Program comparison analysis
            if 'program_comparison' in results:
                f.write("PROGRAM WORKLOAD COMPARISON\\n")
                f.write("-"*28 + "\\n")
                
                program_results = results['program_comparison']
                
                f.write(f"{'Program':<18} {'CPI':<8} {'Hit Rate':<10} {'Branches':<10}\\n")
                f.write("-"*50 + "\\n")
                
                for result in program_results:
                    f.write(f"{result['program']:<18} "
                           f"{result['cpi']:<8.2f} "
                           f"{result['cache_hit_rate']:<10.3f} "
                           f"{result['branch_count']:<10d}\\n")
                
                # Program characteristics analysis
                f.write("\\nProgram Characteristics:\\n")
                for result in program_results:
                    mem_intensity = result['memory_accesses'] / result['instructions']
                    branch_intensity = result['branch_count'] / result['instructions']
                    
                    f.write(f"{result['program']}:\\n")
                    f.write(f"  Memory Intensity: {mem_intensity:.3f} accesses/instruction\\n")
                    f.write(f"  Branch Intensity: {branch_intensity:.3f} branches/instruction\\n")
                    f.write(f"  Cache Friendliness: {'High' if result['cache_hit_rate'] > 0.8 else 'Medium' if result['cache_hit_rate'] > 0.6 else 'Low'}\\n\\n")
            
            # Performance insights
            f.write("PERFORMANCE INSIGHTS\\n")
            f.write("-"*20 + "\\n")
            
            all_results = []
            for category in results.values():
                if isinstance(category, list):
                    all_results.extend(category)
            
            if all_results:
                avg_cpi = statistics.mean([r['cpi'] for r in all_results])
                avg_hit_rate = statistics.mean([r['cache_hit_rate'] for r in all_results])
                
                f.write(f"Average CPI across all tests: {avg_cpi:.3f}\\n")
                f.write(f"Average Cache Hit Rate: {avg_hit_rate:.3f}\\n\\n")
                
                # Identify patterns
                cache_enabled_results = [r for r in all_results if r['cache_enabled']]
                no_cache_results = [r for r in all_results if not r['cache_enabled']]
                
                if cache_enabled_results and no_cache_results:
                    cache_cpi = statistics.mean([r['cpi'] for r in cache_enabled_results])
                    no_cache_cpi = statistics.mean([r['cpi'] for r in no_cache_results])
                    improvement = ((no_cache_cpi - cache_cpi) / no_cache_cpi) * 100
                    
                    f.write(f"Cache Performance Improvement: {improvement:.1f}%\\n")
                    f.write(f"  With Cache: {cache_cpi:.3f} CPI\\n")
                    f.write(f"  Without Cache: {no_cache_cpi:.3f} CPI\\n\\n")
        
        print(f"Report saved to {output_file}")
    
    def save_csv_results(self, results: Dict[str, List[Dict[str, Any]]], 
                        output_file: str = 'benchmark_results.csv') -> None:
        """
        Save benchmark results to CSV file
        
        Args:
            results: Benchmark results dictionary
            output_file: Output CSV file path
        """
        print(f"Saving CSV results: {output_file}")
        
        # Flatten all results
        all_results = []
        for category, category_results in results.items():
            if isinstance(category_results, list):
                for result in category_results:
                    result['category'] = category
                    all_results.append(result)
        
        if not all_results:
            print("No results to save")
            return
        
        # Write CSV
        fieldnames = [
            'category', 'program', 'config', 'cache_enabled',
            'cycles', 'instructions', 'cpi', 'memory_accesses',
            'cache_hit_rate', 'cache_miss_rate', 'branch_count',
            'branch_mispredict_rate', 'simulation_time'
        ]
        
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in all_results:
                # Filter to only include CSV fields
                csv_result = {field: result.get(field, '') for field in fieldnames}
                writer.writerow(csv_result)
        
        print(f"CSV results saved to {output_file}")


def main():
    """Main function for running benchmarks"""
    print("ARM CPU Simulator Benchmarking Suite")
    print("="*40)
    
    # Create benchmark suite
    benchmark = BenchmarkSuite()
    
    # Run quick benchmark by default, full suite with --full flag
    import sys
    if '--full' in sys.argv:
        print("Running full benchmark suite (this may take a while)...")
        results = benchmark.run_full_benchmark_suite(max_cycles=5000)
    else:
        print("Running quick benchmark (use --full for complete suite)")
        results = {}
        results['config_comparison'] = benchmark.run_configuration_comparison(
            'mixed_workload', max_cycles=2000
        )
        results['program_comparison'] = benchmark.run_program_comparison(
            'l1_data', max_cycles=2000
        )
    
    # Generate outputs
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    report_file = os.path.join(output_dir, 'benchmark_report.txt')
    csv_file = os.path.join(output_dir, 'benchmark_results.csv')
    
    benchmark.generate_report(results, report_file)
    benchmark.save_csv_results(results, csv_file)
    
    print(f"\\nBenchmarking completed!")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
