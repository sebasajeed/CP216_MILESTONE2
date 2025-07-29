#!/usr/bin/env python3
"""
Debug instruction execution in detail
"""

import sys
sys.path.append('.')

from cpu_simulator.main import CPUSimulator

def detailed_debug():
    print("=== Detailed Instruction Execution Debug ===")
    
    sim = CPUSimulator()
    
    # Enable debug mode
    sim.cpu.debug_mode = True
    
    program = '''
        MOV R0, #10
    '''
    
    sim.load_program_assembly(program)
    
    print("\nBefore execution:")
    print(f"PC: 0x{sim.cpu.registers.get_pc():08X}")
    print(f"R0: {sim.cpu.registers.read_register(0)}")
    print(f"CPU State: {sim.cpu.state}")
    print(f"Pipeline enabled: {sim.cpu.pipeline_enabled}")
    
    # Read instruction manually
    pc = sim.cpu.registers.get_pc()
    print(f"\nReading instruction at PC 0x{pc:08X}")
    
    # Try to read from memory directly
    try:
        instruction_word = sim.memory.read_word(pc)
        print(f"Raw instruction word: 0x{instruction_word:08X}")
    except Exception as e:
        print(f"Error reading from memory: {e}")
    
    # Step multiple cycles to complete pipeline
    print(f"\nStepping multiple cycles to complete pipeline...")
    for i in range(5):
        result = sim.step_simulation(1)
        print(f"Step {i+2}: result={result}, PC=0x{sim.cpu.registers.get_pc():08X}, R0={sim.cpu.registers.read_register(0)}, instructions={sim.cpu.instruction_count}, cycles={sim.cpu.cycle_count}")
        if not result:
            break

if __name__ == "__main__":
    detailed_debug()
