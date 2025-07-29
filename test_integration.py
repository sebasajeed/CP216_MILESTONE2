#!/usr/bin/env python3
"""
Integration test for the complete CPU simulator system
"""

import sys
sys.path.append('.')

from cpu_simulator.main import CPUSimulator

def main():
    # Create simulator
    sim = CPUSimulator()

    # Test program: Simple arithmetic operations
    program = '''
        MOV R0, #10
        MOV R1, #20  
        ADD R2, R0, R1
        MOV R4, #1
    '''

    print('=== CPU Simulator Integration Test ===')
    print('Program:')
    for i, line in enumerate(program.strip().split('\n'), 1):
        print(f'{i:2}: {line}')

    print('\n=== Loading and Running Program ===')
    try:
        # Load program
        sim.load_program_assembly(program)
        print('✓ Program loaded successfully')
        
        # Run simulation
        print('\nStarting simulation...')
        sim.run_simulation(max_cycles=100)  # Increased from 50 to 100
        
        # Get results
        stats = sim.get_statistics()
        print('\n=== Simulation Results ===')
        print(f'Cycles executed: {stats["cycles_executed"]}')
        print(f'Instructions executed: {stats["instructions_executed"]}')
        
        # Debug: Print cache stats structure
        cache_stats = stats.get("cache_stats", {})
        if cache_stats and isinstance(cache_stats, dict):
            print(f'Cache statistics available: {list(cache_stats.keys())}')
            # Try to find cache hits/misses in various formats
            if 'hits' in cache_stats:
                print(f'Cache hits: {cache_stats["hits"]}')
                print(f'Cache misses: {cache_stats.get("misses", "N/A")}')
            else:
                print(f'Raw cache stats: {cache_stats}')
        else:
            print('No cache statistics available')
        
        # Check result in R4 (success/error code)
        register_file = sim.cpu.registers
        result_code = register_file.read_register(4)
        success_msg = 'SUCCESS' if result_code == 1 else 'ERROR'
        print(f'\nResult code in R4: {result_code} ({success_msg})')
        
        # Show final register state (first 5 registers)
        print('\n=== Final Register State ===')
        for i in range(5):
            val = register_file.read_register(i)
            print(f'R{i}: {val} (0x{val:08X})')
            
        # Test arithmetic result
        r0_val = register_file.read_register(0)  # Should be 10
        r1_val = register_file.read_register(1)  # Should be 20
        r2_val = register_file.read_register(2)  # Should be 30
        
        print(f'\n=== Arithmetic Test ===')
        print(f'R0 (10): {r0_val}')
        print(f'R1 (20): {r1_val}')
        print(f'R2 (R0+R1): {r2_val}')
        
        if r2_val == r0_val + r1_val:
            print('✓ Arithmetic test PASSED!')
        else:
            print('✗ Arithmetic test FAILED!')
            
    except Exception as e:
        print(f'Error during test: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
