from typing import Dict, Any, Optional
from enum import Enum


class ThumbInstructionType(Enum):
    """Thumb instruction types"""
    FORMAT_1 = "move_shifted_register"      # LSL, LSR, ASR
    FORMAT_2 = "add_subtract"               # ADD, SUB (immediate/register)
    FORMAT_3 = "move_compare_immediate"     # MOV, CMP, ADD, SUB (immediate)
    FORMAT_4 = "alu_operations"             # AND, EOR, LSL, LSR, etc.
    FORMAT_5 = "hi_register_ops"            # Hi register operations
    FORMAT_6 = "pc_relative_load"           # LDR (PC-relative)
    FORMAT_7 = "load_store_register"        # LDR, STR (register offset)
    FORMAT_8 = "load_store_sign_extend"     # LDRH, STRH, LDSB, LDSH
    FORMAT_9 = "load_store_immediate"       # LDR, STR (immediate offset)
    FORMAT_10 = "load_store_halfword"       # LDRH, STRH (immediate)
    FORMAT_11 = "sp_relative_load_store"    # LDR, STR (SP-relative)
    FORMAT_12 = "load_address"              # ADD (PC/SP relative)
    FORMAT_13 = "add_offset_sp"             # ADD, SUB (SP immediate)
    FORMAT_14 = "push_pop"                  # PUSH, POP
    FORMAT_15 = "multiple_load_store"       # LDMIA, STMIA
    FORMAT_16 = "conditional_branch"        # B (conditional)
    FORMAT_17 = "software_interrupt"        # SWI
    FORMAT_18 = "unconditional_branch"      # B (unconditional)
    FORMAT_19 = "long_branch_link"          # BL (long branch with link)


class ThumbInstructionDecoder:
    """
    Decoder for Thumb (16-bit) instructions
    
    Provides decoding functionality for the compressed Thumb instruction
    set used in ARM processors.
    """
    
    def __init__(self):
        """Initialize Thumb decoder"""
        self.format_decoders = {
            ThumbInstructionType.FORMAT_1: self._decode_format_1,
            ThumbInstructionType.FORMAT_2: self._decode_format_2,
            ThumbInstructionType.FORMAT_3: self._decode_format_3,
            ThumbInstructionType.FORMAT_4: self._decode_format_4,
            ThumbInstructionType.FORMAT_5: self._decode_format_5,
            ThumbInstructionType.FORMAT_6: self._decode_format_6,
            ThumbInstructionType.FORMAT_7: self._decode_format_7,
            ThumbInstructionType.FORMAT_8: self._decode_format_8,
            ThumbInstructionType.FORMAT_9: self._decode_format_9,
            ThumbInstructionType.FORMAT_10: self._decode_format_10,
            ThumbInstructionType.FORMAT_11: self._decode_format_11,
            ThumbInstructionType.FORMAT_12: self._decode_format_12,
            ThumbInstructionType.FORMAT_13: self._decode_format_13,
            ThumbInstructionType.FORMAT_14: self._decode_format_14,
            ThumbInstructionType.FORMAT_15: self._decode_format_15,
            ThumbInstructionType.FORMAT_16: self._decode_format_16,
            ThumbInstructionType.FORMAT_17: self._decode_format_17,
            ThumbInstructionType.FORMAT_18: self._decode_format_18,
            ThumbInstructionType.FORMAT_19: self._decode_format_19,
        }
        
        # Statistics
        self.decoded_count = 0
    
    def decode_instruction(self, instruction: int) -> Dict[str, Any]:
        """
        Decode 16-bit Thumb instruction
        
        Args:
            instruction: 16-bit Thumb instruction
            
        Returns:
            Decoded instruction dictionary
        """
        self.decoded_count += 1
        
        # Identify instruction format
        format_type = self._identify_format(instruction)
        
        if format_type in self.format_decoders:
            return self.format_decoders[format_type](instruction)
        else:
            return {
                'type': 'UNDEFINED',
                'instruction': instruction,
                'format': 'unknown',
                'cycles': 1
            }
    
    def _identify_format(self, instruction: int) -> ThumbInstructionType:
        """Identify Thumb instruction format from bit pattern"""
        # Extract relevant bits for format identification
        bits_15_13 = (instruction >> 13) & 0x7
        bits_15_12 = (instruction >> 12) & 0xF
        bits_15_11 = (instruction >> 11) & 0x1F
        bits_15_10 = (instruction >> 10) & 0x3F
        bits_15_8 = (instruction >> 8) & 0xFF
        
        # Format identification logic
        if bits_15_13 == 0b000:
            if bits_15_11 == 0b00011:
                return ThumbInstructionType.FORMAT_2  # ADD/SUB
            else:
                return ThumbInstructionType.FORMAT_1  # Move shifted register
        elif bits_15_13 == 0b001:
            return ThumbInstructionType.FORMAT_3  # Move/compare/add/subtract immediate
        elif bits_15_10 == 0b010000:
            return ThumbInstructionType.FORMAT_4  # ALU operations
        elif bits_15_10 == 0b010001:
            return ThumbInstructionType.FORMAT_5  # Hi register operations/branch exchange
        elif bits_15_11 == 0b01001:
            return ThumbInstructionType.FORMAT_6  # PC-relative load
        elif bits_15_12 == 0b0101:
            if (instruction >> 9) & 0x1:
                return ThumbInstructionType.FORMAT_8  # Load/store sign-extended byte/halfword
            else:
                return ThumbInstructionType.FORMAT_7  # Load/store with register offset
        elif bits_15_13 == 0b011:
            return ThumbInstructionType.FORMAT_9  # Load/store with immediate offset
        elif bits_15_12 == 0b1000:
            return ThumbInstructionType.FORMAT_10  # Load/store halfword
        elif bits_15_12 == 0b1001:
            return ThumbInstructionType.FORMAT_11  # SP-relative load/store
        elif bits_15_12 == 0b1010:
            return ThumbInstructionType.FORMAT_12  # Load address
        elif bits_15_12 == 0b1011:
            if bits_15_8 == 0b10110000:
                return ThumbInstructionType.FORMAT_13  # Add offset to stack pointer
            else:
                return ThumbInstructionType.FORMAT_14  # Push/pop registers
        elif bits_15_12 == 0b1100:
            return ThumbInstructionType.FORMAT_15  # Multiple load/store
        elif bits_15_12 == 0b1101:
            if bits_15_8 == 0b11011111:
                return ThumbInstructionType.FORMAT_17  # Software interrupt
            else:
                return ThumbInstructionType.FORMAT_16  # Conditional branch
        elif bits_15_11 == 0b11100:
            return ThumbInstructionType.FORMAT_18  # Unconditional branch
        elif bits_15_11 == 0b11110 or bits_15_11 == 0b11111:
            return ThumbInstructionType.FORMAT_19  # Long branch with link
        
        return ThumbInstructionType.FORMAT_1  # Default
    
    # Format decoders (simplified implementations)
    def _decode_format_1(self, instruction: int) -> Dict[str, Any]:
        """Format 1: Move shifted register"""
        op = (instruction >> 11) & 0x3
        offset = (instruction >> 6) & 0x1F
        rs = (instruction >> 3) & 0x7
        rd = instruction & 0x7
        
        operations = {0: 'LSL', 1: 'LSR', 2: 'ASR'}
        
        return {
            'type': 'SHIFT',
            'operation': operations.get(op, 'LSL'),
            'rd': rd,
            'rs': rs,
            'shift_amount': offset,
            'cycles': 1,
            'format': 'Format 1'
        }
    
    def _decode_format_2(self, instruction: int) -> Dict[str, Any]:
        """Format 2: Add/subtract"""
        i = (instruction >> 10) & 0x1  # Immediate flag
        op = (instruction >> 9) & 0x1  # Operation (0=ADD, 1=SUB)
        rn_offset = (instruction >> 6) & 0x7
        rs = (instruction >> 3) & 0x7
        rd = instruction & 0x7
        
        return {
            'type': 'ADD_SUB',
            'operation': 'SUB' if op else 'ADD',
            'rd': rd,
            'rs': rs,
            'operand': rn_offset,
            'immediate': bool(i),
            'cycles': 1,
            'format': 'Format 2'
        }
    
    def _decode_format_3(self, instruction: int) -> Dict[str, Any]:
        """Format 3: Move/compare/add/subtract immediate"""
        op = (instruction >> 11) & 0x3
        rd = (instruction >> 8) & 0x7
        offset = instruction & 0xFF
        
        operations = {0: 'MOV', 1: 'CMP', 2: 'ADD', 3: 'SUB'}
        
        return {
            'type': 'IMM_OP',
            'operation': operations.get(op, 'MOV'),
            'rd': rd,
            'immediate': offset,
            'cycles': 1,
            'format': 'Format 3'
        }
    
    def _decode_format_4(self, instruction: int) -> Dict[str, Any]:
        """Format 4: ALU operations"""
        op = (instruction >> 6) & 0xF
        rs = (instruction >> 3) & 0x7
        rd = instruction & 0x7
        
        operations = {
            0: 'AND', 1: 'EOR', 2: 'LSL', 3: 'LSR',
            4: 'ASR', 5: 'ADC', 6: 'SBC', 7: 'ROR',
            8: 'TST', 9: 'NEG', 10: 'CMP', 11: 'CMN',
            12: 'ORR', 13: 'MUL', 14: 'BIC', 15: 'MVN'
        }
        
        return {
            'type': 'ALU',
            'operation': operations.get(op, 'AND'),
            'rd': rd,
            'rs': rs,
            'cycles': 1 if op != 13 else 3,  # MUL takes 3 cycles
            'format': 'Format 4'
        }
    
    def _decode_format_5(self, instruction: int) -> Dict[str, Any]:
        """Format 5: Hi register operations/branch exchange"""
        op = (instruction >> 8) & 0x3
        h1 = (instruction >> 7) & 0x1
        h2 = (instruction >> 6) & 0x1
        rs = ((instruction >> 3) & 0x7) | (h2 << 3)
        rd = (instruction & 0x7) | (h1 << 3)
        
        operations = {0: 'ADD', 1: 'CMP', 2: 'MOV', 3: 'BX'}
        
        return {
            'type': 'HI_REG',
            'operation': operations.get(op, 'ADD'),
            'rd': rd,
            'rs': rs,
            'cycles': 1 if op != 3 else 3,  # BX takes 3 cycles
            'format': 'Format 5'
        }
    
    # Simplified implementations for other formats
    def _decode_format_6(self, instruction: int) -> Dict[str, Any]:
        """Format 6: PC-relative load"""
        return {'type': 'LDR_PC', 'cycles': 3, 'format': 'Format 6'}
    
    def _decode_format_7(self, instruction: int) -> Dict[str, Any]:
        """Format 7: Load/store with register offset"""
        return {'type': 'LDR_STR_REG', 'cycles': 3, 'format': 'Format 7'}
    
    def _decode_format_8(self, instruction: int) -> Dict[str, Any]:
        """Format 8: Load/store sign-extended byte/halfword"""
        return {'type': 'LDR_STR_SIGN', 'cycles': 3, 'format': 'Format 8'}
    
    def _decode_format_9(self, instruction: int) -> Dict[str, Any]:
        """Format 9: Load/store with immediate offset"""
        return {'type': 'LDR_STR_IMM', 'cycles': 3, 'format': 'Format 9'}
    
    def _decode_format_10(self, instruction: int) -> Dict[str, Any]:
        """Format 10: Load/store halfword"""
        return {'type': 'LDRH_STRH', 'cycles': 3, 'format': 'Format 10'}
    
    def _decode_format_11(self, instruction: int) -> Dict[str, Any]:
        """Format 11: SP-relative load/store"""
        return {'type': 'LDR_STR_SP', 'cycles': 3, 'format': 'Format 11'}
    
    def _decode_format_12(self, instruction: int) -> Dict[str, Any]:
        """Format 12: Load address"""
        return {'type': 'LOAD_ADDR', 'cycles': 1, 'format': 'Format 12'}
    
    def _decode_format_13(self, instruction: int) -> Dict[str, Any]:
        """Format 13: Add offset to stack pointer"""
        return {'type': 'ADD_SP', 'cycles': 1, 'format': 'Format 13'}
    
    def _decode_format_14(self, instruction: int) -> Dict[str, Any]:
        """Format 14: Push/pop registers"""
        return {'type': 'PUSH_POP', 'cycles': 2, 'format': 'Format 14'}
    
    def _decode_format_15(self, instruction: int) -> Dict[str, Any]:
        """Format 15: Multiple load/store"""
        return {'type': 'LDMIA_STMIA', 'cycles': 2, 'format': 'Format 15'}
    
    def _decode_format_16(self, instruction: int) -> Dict[str, Any]:
        """Format 16: Conditional branch"""
        cond = (instruction >> 8) & 0xF
        offset = instruction & 0xFF
        
        return {
            'type': 'BRANCH_COND',
            'condition': cond,
            'offset': offset,
            'cycles': 3 if self._branch_taken(cond) else 1,
            'format': 'Format 16'
        }
    
    def _decode_format_17(self, instruction: int) -> Dict[str, Any]:
        """Format 17: Software interrupt"""
        return {'type': 'SWI', 'cycles': 3, 'format': 'Format 17'}
    
    def _decode_format_18(self, instruction: int) -> Dict[str, Any]:
        """Format 18: Unconditional branch"""
        return {'type': 'BRANCH', 'cycles': 3, 'format': 'Format 18'}
    
    def _decode_format_19(self, instruction: int) -> Dict[str, Any]:
        """Format 19: Long branch with link"""
        return {'type': 'BL', 'cycles': 4, 'format': 'Format 19'}
    
    def _branch_taken(self, condition: int) -> bool:
        """Simplified branch condition evaluation"""
        # In real implementation, this would check CPSR flags
        return condition != 0xE  # Always branch except for "never"
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get decoder statistics"""
        return {
            'instructions_decoded': self.decoded_count,
            'instruction_size': 16,  # bits
            'supported_formats': len(self.format_decoders)
        }


# Example usage
if __name__ == "__main__":
    decoder = ThumbInstructionDecoder()
    
    # Test Thumb instruction decoding
    # MOV R0, #42 (Thumb format)
    mov_thumb = 0x2A00  # Format 3: MOV R0, #42
    decoded = decoder.decode_instruction(mov_thumb)
    print(f"Decoded Thumb MOV: {decoded}")
    
    # ADD R1, R2 (Thumb format)
    add_thumb = 0x1889  # Format 2: ADD R1, R2
    decoded = decoder.decode_instruction(add_thumb)
    print(f"Decoded Thumb ADD: {decoded}")
    
    # Statistics
    stats = decoder.get_statistics()
    print(f"Decoder statistics: {stats}")
