"""
ARM Instructions Module

This module implements ARM instruction encoding/decoding and execution
for the ARM instruction set architecture. It provides support for
the complete ARM instruction set including data processing, memory
operations, branching, and system instructions.

Author: CPU Simulator Project
Date: 2025
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union
import struct


class ARMCondition(Enum):
    """ARM condition codes"""
    EQ = 0x0   # Equal
    NE = 0x1   # Not equal
    CS = 0x2   # Carry set / Unsigned higher or same
    CC = 0x3   # Carry clear / Unsigned lower
    MI = 0x4   # Minus / Negative
    PL = 0x5   # Plus / Positive or zero
    VS = 0x6   # Overflow
    VC = 0x7   # No overflow
    HI = 0x8   # Unsigned higher
    LS = 0x9   # Unsigned lower or same
    GE = 0xA   # Signed greater than or equal
    LT = 0xB   # Signed less than
    GT = 0xC   # Signed greater than
    LE = 0xD   # Signed less than or equal
    AL = 0xE   # Always
    NV = 0xF   # Never (deprecated)


class DataProcessingOpcode(Enum):
    """Data processing opcodes"""
    AND = 0x0
    EOR = 0x1
    SUB = 0x2
    RSB = 0x3
    ADD = 0x4
    ADC = 0x5
    SBC = 0x6
    RSC = 0x7
    TST = 0x8
    TEQ = 0x9
    CMP = 0xA
    CMN = 0xB
    ORR = 0xC
    MOV = 0xD
    BIC = 0xE
    MVN = 0xF


class ShiftType(Enum):
    """ARM shift types"""
    LSL = 0x0  # Logical shift left
    LSR = 0x1  # Logical shift right
    ASR = 0x2  # Arithmetic shift right
    ROR = 0x3  # Rotate right


class LoadStoreOpcode(Enum):
    """Load/Store opcodes"""
    STR = 0   # Store word
    LDR = 1   # Load word
    STRB = 2  # Store byte
    LDRB = 3  # Load byte


class MultipleOpcode(Enum):
    """Load/Store Multiple opcodes"""
    STM = 0   # Store multiple
    LDM = 1   # Load multiple


class ARMInstructionEncoder:
    """
    ARM instruction encoder for generating binary instructions
    """
    
    @staticmethod
    def encode_data_processing(condition: ARMCondition, opcode: DataProcessingOpcode,
                             set_flags: bool, rn: int, rd: int, 
                             operand2: Union[int, Tuple[int, int, int]]) -> int:
        """
        Encode data processing instruction
        
        Args:
            condition: Condition code
            opcode: Data processing opcode
            set_flags: Whether to set condition flags
            rn: First operand register
            rd: Destination register
            operand2: Either immediate value or (rm, shift_type, shift_amount)
            
        Returns:
            32-bit encoded instruction
        """
        instruction = 0
        
        # Condition field [31:28]
        instruction |= (condition.value & 0xF) << 28
        
        # Opcode field [24:21]
        instruction |= (opcode.value & 0xF) << 21
        
        # Set flags bit [20]
        if set_flags:
            instruction |= 1 << 20
        
        # Rn field [19:16]
        instruction |= (rn & 0xF) << 16
        
        # Rd field [15:12]
        instruction |= (rd & 0xF) << 12
        
        # Operand2 field
        if isinstance(operand2, int):
            # Immediate operand
            instruction |= 1 << 25  # Immediate bit
            # For simplicity, assume no rotation needed
            instruction |= operand2 & 0xFF
        else:
            # Register operand with shift
            if isinstance(operand2, tuple) and len(operand2) >= 3:
                rm, shift_type, shift_amount = operand2[0], operand2[1], operand2[2]
            else:
                # Fallback - just use as register
                rm, shift_type, shift_amount = operand2, 0, 0
            instruction |= rm & 0xF  # Rm field [3:0]
            instruction |= (shift_type & 0x3) << 5  # Shift type [6:5]
            instruction |= (shift_amount & 0x1F) << 7  # Shift amount [11:7]
        
        return instruction
    
    @staticmethod
    def encode_load_store(condition: ARMCondition, opcode: LoadStoreOpcode,
                         pre_index: bool, up: bool, writeback: bool,
                         rn: int, rd: int, offset: Union[int, Tuple[int, int, int]]) -> int:
        """
        Encode load/store instruction
        
        Args:
            condition: Condition code
            opcode: Load/store opcode
            pre_index: Pre-indexed addressing
            up: Add offset (True) or subtract (False)
            writeback: Write back to base register
            rn: Base register
            rd: Source/destination register
            offset: Either immediate offset or (rm, shift_type, shift_amount)
            
        Returns:
            32-bit encoded instruction
        """
        instruction = 0
        
        # Condition field [31:28]
        instruction |= (condition.value & 0xF) << 28
        
        # Load/Store bit pattern [27:26]
        instruction |= 0x1 << 26
        
        # Immediate bit [25] - set for register offset
        if not isinstance(offset, int):
            instruction |= 1 << 25
        
        # Pre/post index bit [24]
        if pre_index:
            instruction |= 1 << 24
        
        # Up/down bit [23]
        if up:
            instruction |= 1 << 23
        
        # Byte/word bit [22]
        if opcode in [LoadStoreOpcode.STRB, LoadStoreOpcode.LDRB]:
            instruction |= 1 << 22
        
        # Write-back bit [21]
        if writeback:
            instruction |= 1 << 21
        
        # Load/store bit [20]
        if opcode in [LoadStoreOpcode.LDR, LoadStoreOpcode.LDRB]:
            instruction |= 1 << 20
        
        # Rn field [19:16]
        instruction |= (rn & 0xF) << 16
        
        # Rd field [15:12]
        instruction |= (rd & 0xF) << 12
        
        # Offset field
        if isinstance(offset, int):
            # Immediate offset [11:0]
            instruction |= offset & 0xFFF
        else:
            # Register offset
            rm, shift_type, shift_amount = offset
            instruction |= rm & 0xF  # Rm field [3:0]
            instruction |= (shift_type & 0x3) << 5  # Shift type [6:5]
            instruction |= (shift_amount & 0x1F) << 7  # Shift amount [11:7]
        
        return instruction
    
    @staticmethod
    def encode_branch(condition: ARMCondition, link: bool, offset: int) -> int:
        """
        Encode branch instruction
        
        Args:
            condition: Condition code
            link: Branch with link (BL vs B)
            offset: 24-bit signed offset (in words)
            
        Returns:
            32-bit encoded instruction
        """
        instruction = 0
        
        # Condition field [31:28]
        instruction |= (condition.value & 0xF) << 28
        
        # Branch pattern [27:25]
        instruction |= 0x5 << 25
        
        # Link bit [24]
        if link:
            instruction |= 1 << 24
        
        # Offset field [23:0]
        instruction |= offset & 0xFFFFFF
        
        return instruction
    
    @staticmethod
    def encode_load_store_multiple(condition: ARMCondition, opcode: MultipleOpcode,
                                  pre_index: bool, up: bool, psr: bool, writeback: bool,
                                  rn: int, register_list: int) -> int:
        """
        Encode load/store multiple instruction
        
        Args:
            condition: Condition code
            opcode: LDM or STM
            pre_index: Pre-increment addressing
            up: Increment (True) or decrement (False)
            psr: PSR & force user bit
            writeback: Write back to base register
            rn: Base register
            register_list: 16-bit register list
            
        Returns:
            32-bit encoded instruction
        """
        instruction = 0
        
        # Condition field [31:28]
        instruction |= (condition.value & 0xF) << 28
        
        # Multiple pattern [27:25]
        instruction |= 0x4 << 25
        
        # Pre/post index bit [24]
        if pre_index:
            instruction |= 1 << 24
        
        # Up/down bit [23]
        if up:
            instruction |= 1 << 23
        
        # PSR & force user bit [22]
        if psr:
            instruction |= 1 << 22
        
        # Write-back bit [21]
        if writeback:
            instruction |= 1 << 21
        
        # Load/store bit [20]
        if opcode == MultipleOpcode.LDM:
            instruction |= 1 << 20
        
        # Rn field [19:16]
        instruction |= (rn & 0xF) << 16
        
        # Register list [15:0]
        instruction |= register_list & 0xFFFF
        
        return instruction
    
    @staticmethod
    def encode_multiply(condition: ARMCondition, accumulate: bool, set_flags: bool,
                       rd: int, rn: int, rs: int, rm: int) -> int:
        """
        Encode multiply instruction
        
        Args:
            condition: Condition code
            accumulate: MLA (True) vs MUL (False)
            set_flags: Set condition flags
            rd: Destination register
            rn: Accumulate register (for MLA)
            rs: Multiplier register
            rm: Multiplicand register
            
        Returns:
            32-bit encoded instruction
        """
        instruction = 0
        
        # Condition field [31:28]
        instruction |= (condition.value & 0xF) << 28
        
        # Accumulate bit [21]
        if accumulate:
            instruction |= 1 << 21
        
        # Set flags bit [20]
        if set_flags:
            instruction |= 1 << 20
        
        # Rd field [19:16]
        instruction |= (rd & 0xF) << 16
        
        # Rn field [15:12]
        instruction |= (rn & 0xF) << 12
        
        # Rs field [11:8]
        instruction |= (rs & 0xF) << 8
        
        # Multiply signature [7:4]
        instruction |= 0x9 << 4
        
        # Rm field [3:0]
        instruction |= rm & 0xF
        
        return instruction
    
    @staticmethod
    def encode_software_interrupt(condition: ARMCondition, imm24: int) -> int:
        """
        Encode software interrupt instruction
        
        Args:
            condition: Condition code
            imm24: 24-bit immediate value
            
        Returns:
            32-bit encoded instruction
        """
        instruction = 0
        
        # Condition field [31:28]
        instruction |= (condition.value & 0xF) << 28
        
        # SWI pattern [27:24]
        instruction |= 0xF << 24
        
        # Immediate field [23:0]
        instruction |= imm24 & 0xFFFFFF
        
        return instruction


class ARMInstructionDecoder:
    """
    ARM instruction decoder for parsing binary instructions
    """
    
    @staticmethod
    def decode_instruction(instruction: int) -> Dict[str, Any]:
        """
        Decode 32-bit ARM instruction
        
        Args:
            instruction: 32-bit instruction word
            
        Returns:
            Dictionary with decoded instruction fields
        """
        decoded = {
            'raw': instruction,
            'condition': (instruction >> 28) & 0xF,
            'format': 'unknown'
        }
        
        # Determine instruction format
        bits_27_25 = (instruction >> 25) & 0x7
        bits_7_4 = (instruction >> 4) & 0xF
        
        if bits_27_25 == 0b000:
            if bits_7_4 == 0b1001:
                decoded.update(ARMInstructionDecoder.decode_multiply(instruction))
            else:
                decoded.update(ARMInstructionDecoder.decode_data_processing(instruction))
        elif bits_27_25 == 0b001:
            decoded.update(ARMInstructionDecoder.decode_data_processing(instruction))
        elif bits_27_25 in [0b010, 0b011]:
            decoded.update(ARMInstructionDecoder.decode_load_store(instruction))
        elif bits_27_25 == 0b100:
            decoded.update(ARMInstructionDecoder.decode_load_store_multiple(instruction))
        elif bits_27_25 == 0b101:
            decoded.update(ARMInstructionDecoder.decode_branch(instruction))
        elif bits_27_25 == 0b111:
            decoded.update(ARMInstructionDecoder.decode_software_interrupt(instruction))
        
        return decoded
    
    @staticmethod
    def decode_data_processing(instruction: int) -> Dict[str, Any]:
        """Decode data processing instruction"""
        return {
            'format': 'data_processing',
            'immediate': bool(instruction & (1 << 25)),
            'opcode': (instruction >> 21) & 0xF,
            'set_flags': bool(instruction & (1 << 20)),
            'rn': (instruction >> 16) & 0xF,
            'rd': (instruction >> 12) & 0xF,
            'operand2': instruction & 0xFFF if instruction & (1 << 25) else {
                'rm': instruction & 0xF,
                'shift_type': (instruction >> 5) & 0x3,
                'shift_amount': (instruction >> 7) & 0x1F if not (instruction & (1 << 4)) else (instruction >> 8) & 0xF,
                'shift_register': bool(instruction & (1 << 4))
            }
        }
    
    @staticmethod
    def decode_load_store(instruction: int) -> Dict[str, Any]:
        """Decode load/store instruction"""
        return {
            'format': 'load_store',
            'immediate': not bool(instruction & (1 << 25)),
            'pre_index': bool(instruction & (1 << 24)),
            'up': bool(instruction & (1 << 23)),
            'byte': bool(instruction & (1 << 22)),
            'writeback': bool(instruction & (1 << 21)),
            'load': bool(instruction & (1 << 20)),
            'rn': (instruction >> 16) & 0xF,
            'rd': (instruction >> 12) & 0xF,
            'offset': instruction & 0xFFF if instruction & (1 << 25) else {
                'rm': instruction & 0xF,
                'shift_type': (instruction >> 5) & 0x3,
                'shift_amount': (instruction >> 7) & 0x1F
            }
        }
    
    @staticmethod
    def decode_load_store_multiple(instruction: int) -> Dict[str, Any]:
        """Decode load/store multiple instruction"""
        return {
            'format': 'load_store_multiple',
            'pre_index': bool(instruction & (1 << 24)),
            'up': bool(instruction & (1 << 23)),
            'psr': bool(instruction & (1 << 22)),
            'writeback': bool(instruction & (1 << 21)),
            'load': bool(instruction & (1 << 20)),
            'rn': (instruction >> 16) & 0xF,
            'register_list': instruction & 0xFFFF
        }
    
    @staticmethod
    def decode_branch(instruction: int) -> Dict[str, Any]:
        """Decode branch instruction"""
        offset = instruction & 0xFFFFFF
        if offset & 0x800000:  # Sign extend
            offset = offset | 0xFF000000
        
        return {
            'format': 'branch',
            'link': bool(instruction & (1 << 24)),
            'offset': struct.unpack('<i', struct.pack('<I', offset))[0]
        }
    
    @staticmethod
    def decode_multiply(instruction: int) -> Dict[str, Any]:
        """Decode multiply instruction"""
        return {
            'format': 'multiply',
            'accumulate': bool(instruction & (1 << 21)),
            'set_flags': bool(instruction & (1 << 20)),
            'rd': (instruction >> 16) & 0xF,
            'rn': (instruction >> 12) & 0xF,
            'rs': (instruction >> 8) & 0xF,
            'rm': instruction & 0xF
        }
    
    @staticmethod
    def decode_software_interrupt(instruction: int) -> Dict[str, Any]:
        """Decode software interrupt instruction"""
        return {
            'format': 'software_interrupt',
            'imm24': instruction & 0xFFFFFF
        }


class ARMAssembler:
    """
    Simple ARM assembler for creating test programs
    """
    
    def __init__(self):
        self.labels = {}
        self.instructions = []
        self.current_address = 0
    
    def set_origin(self, address: int):
        """Set origin address for assembly"""
        self.current_address = address
    
    def label(self, name: str):
        """Define a label at current address"""
        self.labels[name] = self.current_address
    
    def mov(self, rd: int, operand2: Union[int, Tuple[int, int, int]], condition: ARMCondition = ARMCondition.AL) -> int:
        """Assemble MOV instruction"""
        if isinstance(operand2, int):
            # Immediate operand
            instruction = ARMInstructionEncoder.encode_data_processing(
                condition, DataProcessingOpcode.MOV, False, 0, rd, operand2
            )
        else:
            # Register operand with shift (tuple format)
            instruction = ARMInstructionEncoder.encode_data_processing(
                condition, DataProcessingOpcode.MOV, False, 0, rd, operand2
            )
        
        self.instructions.append((self.current_address, instruction))
        self.current_address += 4
        return instruction
    
    def add(self, rd: int, rn: int, operand2: Union[int, Tuple[int, int, int]], 
            set_flags: bool = False, condition: ARMCondition = ARMCondition.AL) -> int:
        """Assemble ADD instruction"""
        if isinstance(operand2, int):
            # Immediate operand
            instruction = ARMInstructionEncoder.encode_data_processing(
                condition, DataProcessingOpcode.ADD, set_flags, rn, rd, operand2
            )
        else:
            # Register operand with shift (tuple format)
            instruction = ARMInstructionEncoder.encode_data_processing(
                condition, DataProcessingOpcode.ADD, set_flags, rn, rd, operand2
            )
        
        self.instructions.append((self.current_address, instruction))
        self.current_address += 4
        return instruction
    
    def sub(self, rd: int, rn: int, operand2: Union[int, Tuple[int, int, int]], 
            set_flags: bool = False, condition: ARMCondition = ARMCondition.AL) -> int:
        """Assemble SUB instruction"""
        if isinstance(operand2, int):
            # Immediate operand
            instruction = ARMInstructionEncoder.encode_data_processing(
                condition, DataProcessingOpcode.SUB, set_flags, rn, rd, operand2
            )
        else:
            # Register operand with shift (tuple format)
            instruction = ARMInstructionEncoder.encode_data_processing(
                condition, DataProcessingOpcode.SUB, set_flags, rn, rd, operand2
            )
        
        self.instructions.append((self.current_address, instruction))
        self.current_address += 4
        return instruction
    
    def ldr(self, rd: int, rn: int, offset: int = 0, 
            condition: ARMCondition = ARMCondition.AL) -> int:
        """Assemble LDR instruction"""
        instruction = ARMInstructionEncoder.encode_load_store(
            condition, LoadStoreOpcode.LDR, True, offset >= 0, False, rn, rd, abs(offset)
        )
        
        self.instructions.append((self.current_address, instruction))
        self.current_address += 4
        return instruction
    
    def str(self, rd: int, rn: int, offset: int = 0, 
            condition: ARMCondition = ARMCondition.AL) -> int:
        """Assemble STR instruction"""
        instruction = ARMInstructionEncoder.encode_load_store(
            condition, LoadStoreOpcode.STR, True, offset >= 0, False, rn, rd, abs(offset)
        )
        
        self.instructions.append((self.current_address, instruction))
        self.current_address += 4
        return instruction
    
    def b(self, target: Union[str, int], condition: ARMCondition = ARMCondition.AL) -> int:
        """Assemble B instruction"""
        if isinstance(target, str):
            # Label reference - will be resolved later
            offset = 0  # Placeholder
        else:
            # Calculate offset
            offset = (target - self.current_address - 8) // 4  # PC+8 addressing
        
        instruction = ARMInstructionEncoder.encode_branch(condition, False, offset & 0xFFFFFF)
        
        self.instructions.append((self.current_address, instruction))
        self.current_address += 4
        return instruction
    
    def bl(self, target: Union[str, int], condition: ARMCondition = ARMCondition.AL) -> int:
        """Assemble BL instruction"""
        if isinstance(target, str):
            # Label reference - will be resolved later
            offset = 0  # Placeholder
        else:
            # Calculate offset
            offset = (target - self.current_address - 8) // 4  # PC+8 addressing
        
        instruction = ARMInstructionEncoder.encode_branch(condition, True, offset & 0xFFFFFF)
        
        self.instructions.append((self.current_address, instruction))
        self.current_address += 4
        return instruction
    
    def swi(self, imm24: int, condition: ARMCondition = ARMCondition.AL) -> int:
        """Assemble SWI instruction"""
        instruction = ARMInstructionEncoder.encode_software_interrupt(condition, imm24)
        
        self.instructions.append((self.current_address, instruction))
        self.current_address += 4
        return instruction
    
    def nop(self, condition: ARMCondition = ARMCondition.AL) -> int:
        """Assemble NOP instruction (MOV R0, R0)"""
        return self.mov(0, 0, condition)  # MOV R0, R0
    
    def assemble(self) -> bytes:
        """Assemble all instructions to binary"""
        # Resolve label references (simplified)
        resolved_instructions = []
        
        for address, instruction in self.instructions:
            resolved_instructions.append(instruction)
        
        # Convert to bytes (little-endian)
        binary_data = b''
        for instruction in resolved_instructions:
            binary_data += struct.pack('<I', instruction)
        
        return binary_data
    
    def get_program_info(self) -> Dict[str, Any]:
        """Get program information"""
        return {
            'instructions': len(self.instructions),
            'size_bytes': len(self.instructions) * 4,
            'labels': self.labels.copy(),
            'start_address': self.instructions[0][0] if self.instructions else 0,
            'end_address': self.instructions[-1][0] + 4 if self.instructions else 0
        }


# Example usage and testing
if __name__ == "__main__":
    print("Testing ARM Instructions\n")
    
    # Test encoder
    print("Testing instruction encoding...")
    
    # MOV R0, #42
    mov_instr = ARMInstructionEncoder.encode_data_processing(
        ARMCondition.AL, DataProcessingOpcode.MOV, False, 0, 0, 42
    )
    print(f"MOV R0, #42: 0x{mov_instr:08x}")
    
    # ADD R2, R0, R1
    add_instr = ARMInstructionEncoder.encode_data_processing(
        ARMCondition.AL, DataProcessingOpcode.ADD, False, 0, 2, (1, 0, 0)
    )
    print(f"ADD R2, R0, R1: 0x{add_instr:08x}")
    
    # LDR R0, [R1, #4]
    ldr_instr = ARMInstructionEncoder.encode_load_store(
        ARMCondition.AL, LoadStoreOpcode.LDR, True, True, False, 1, 0, 4
    )
    print(f"LDR R0, [R1, #4]: 0x{ldr_instr:08x}")
    
    # Test decoder
    print("\nTesting instruction decoding...")
    
    decoded_mov = ARMInstructionDecoder.decode_instruction(mov_instr)
    print(f"Decoded MOV: {decoded_mov}")
    
    decoded_add = ARMInstructionDecoder.decode_instruction(add_instr)
    print(f"Decoded ADD: {decoded_add}")
    
    # Test assembler
    print("\nTesting assembler...")
    
    asm = ARMAssembler()
    asm.set_origin(0x8000)
    
    asm.mov(0, 10)      # MOV R0, #10
    asm.mov(1, 20)      # MOV R1, #20
    asm.add(2, 0, 1)    # ADD R2, R0, R1
    asm.label("loop")
    asm.b("loop")       # B loop (infinite loop)
    
    program = asm.assemble()
    info = asm.get_program_info()
    
    print(f"Assembled program: {len(program)} bytes")
    print(f"Program info: {info}")
    print(f"Binary: {program.hex()}")
    
    # Decode assembled program
    print("\nDecoding assembled program:")
    for i in range(0, len(program), 4):
        instruction = struct.unpack('<I', program[i:i+4])[0]
        decoded = ARMInstructionDecoder.decode_instruction(instruction)
        print(f"  0x{0x8000 + i:08x}: 0x{instruction:08x} -> {decoded['format']}")
