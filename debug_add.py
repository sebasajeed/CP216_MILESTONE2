#!/usr/bin/env python3

import sys
sys.path.append('.')

from cpu_simulator.main import CPUSimulator
from isa.arm_instructions import ARMAssembler, DataProcessingOpcode

def test_add_instruction():
    print("=== Debugging ADD Instruction ===")
    
    # Test instruction encoding
    asm = ARMAssembler()
    asm.set_origin(0x20000000)
    
    # Manually test MOV and ADD
    print("\n1. Testing instruction encoding:")
    mov_r0 = asm.mov(0, 10)  # MOV R0, #10
    print(f"MOV R0, #10 encoded as: 0x{mov_r0:08X}")
    
    mov_r1 = asm.mov(1, 20)  # MOV R1, #20
    print(f"MOV R1, #20 encoded as: 0x{mov_r1:08X}")
    
    add_r2 = asm.add(2, 0, 1)  # ADD R2, R0, R1
    print(f"ADD R2, R0, R1 encoded as: 0x{add_r2:08X}")
    
    # Test ALU directly
    print("\n2. Testing ALU directly:")
    from cpu.alu import ALU, ALUOperation
    alu = ALU()
    result = alu.execute(ALUOperation.ADD, 10, 20, False, False)
    print(f"ALU ADD(10, 20) = {result.result}")
    
    # Test with simulator
    print("\n3. Testing with simulator:")
    sim = CPUSimulator()
    
    program = '''
        MOV R0, #10
        MOV R1, #20
        ADD R2, R0, R1
    '''
    
    sim.load_program_assembly(program)
    
    # Step through each instruction
    print("Before execution:")
    print(f"R0: {sim.cpu.registers.read_register(0)}")
    print(f"R1: {sim.cpu.registers.read_register(1)}")
    print(f"R2: {sim.cpu.registers.read_register(2)}")
    
    # Execute one instruction at a time
    for i in range(3):
        print(f"\nExecuting instruction {i+1}:")
        pc_before = sim.cpu.registers.get_pc()
        print(f"PC before: 0x{pc_before:08X}")
        
        # Read instruction at PC
        instruction_word = sim.memory.read_word(pc_before)
        print(f"Instruction: 0x{instruction_word:08X}")
        
        # Step one instruction
        sim.step_simulation(1)
        
        pc_after = sim.cpu.registers.get_pc()
        print(f"PC after: 0x{pc_after:08X}")
        print(f"R0: {sim.cpu.registers.read_register(0)}")
        print(f"R1: {sim.cpu.registers.read_register(1)}")
        print(f"R2: {sim.cpu.registers.read_register(2)}")

if __name__ == "__main__":
    test_add_instruction()
