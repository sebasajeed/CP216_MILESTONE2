from enum import Enum
from typing import Dict, List, Optional, Any
import struct


class ProcessorMode(Enum):
    """ARM processor operating modes"""
    USER = 0b10000         # User mode
    FIQ = 0b10001          # Fast Interrupt Request
    IRQ = 0b10010          # Interrupt Request  
    SUPERVISOR = 0b10011   # Supervisor mode
    ABORT = 0b10111        # Abort mode
    UNDEFINED = 0b11011    # Undefined instruction mode
    SYSTEM = 0b11111       # System mode


class ConditionFlags:
    """ARM processor condition flags (CPSR/SPSR bits)"""
    
    def __init__(self, value: int = 0):
        self.value = value
    
    @property
    def N(self) -> bool:
        """Negative flag"""
        return bool(self.value & (1 << 31))
    
    @N.setter
    def N(self, val: bool):
        if val:
            self.value |= (1 << 31)
        else:
            self.value &= ~(1 << 31)
    
    @property
    def Z(self) -> bool:
        """Zero flag"""
        return bool(self.value & (1 << 30))
    
    @Z.setter
    def Z(self, val: bool):
        if val:
            self.value |= (1 << 30)
        else:
            self.value &= ~(1 << 30)
    
    @property
    def C(self) -> bool:
        """Carry flag"""
        return bool(self.value & (1 << 29))
    
    @C.setter
    def C(self, val: bool):
        if val:
            self.value |= (1 << 29)
        else:
            self.value &= ~(1 << 29)
    
    @property
    def V(self) -> bool:
        """Overflow flag"""
        return bool(self.value & (1 << 28))
    
    @V.setter
    def V(self, val: bool):
        if val:
            self.value |= (1 << 28)
        else:
            self.value &= ~(1 << 28)
    
    @property
    def Q(self) -> bool:
        """Saturation flag (ARMv5TE+)"""
        return bool(self.value & (1 << 27))
    
    @Q.setter
    def Q(self, val: bool):
        if val:
            self.value |= (1 << 27)
        else:
            self.value &= ~(1 << 27)
    
    @property
    def I(self) -> bool:
        """IRQ disable flag"""
        return bool(self.value & (1 << 7))
    
    @I.setter
    def I(self, val: bool):
        if val:
            self.value |= (1 << 7)
        else:
            self.value &= ~(1 << 7)
    
    @property
    def F(self) -> bool:
        """FIQ disable flag"""
        return bool(self.value & (1 << 6))
    
    @F.setter
    def F(self, val: bool):
        if val:
            self.value |= (1 << 6)
        else:
            self.value &= ~(1 << 6)
    
    @property
    def T(self) -> bool:
        """Thumb state flag"""
        return bool(self.value & (1 << 5))
    
    @T.setter
    def T(self, val: bool):
        if val:
            self.value |= (1 << 5)
        else:
            self.value &= ~(1 << 5)
    
    @property
    def mode(self) -> ProcessorMode:
        """Processor mode bits [4:0]"""
        mode_bits = self.value & 0x1F
        try:
            return ProcessorMode(mode_bits)
        except ValueError:
            return ProcessorMode.USER  # Default to user mode
    
    @mode.setter
    def mode(self, val: ProcessorMode):
        self.value = (self.value & ~0x1F) | val.value
    
    def __str__(self) -> str:
        flags = []
        if self.N: flags.append('N')
        if self.Z: flags.append('Z')
        if self.C: flags.append('C')
        if self.V: flags.append('V')
        if self.Q: flags.append('Q')
        if self.I: flags.append('I')
        if self.F: flags.append('F')
        if self.T: flags.append('T')
        
        return f"CPSR(0x{self.value:08x}, {self.mode.name}, {'|'.join(flags)})"


class RegisterFile:
    """
    ARM processor register file implementation
    
    Implements the complete ARM register file including:
    - 16 general-purpose registers (R0-R15)
    - Mode-specific banked registers
    - Program Status Registers (CPSR/SPSR)
    - Special register aliases (PC, SP, LR)
    """
    
    # Register aliases
    PC = 15  # Program Counter
    LR = 14  # Link Register
    SP = 13  # Stack Pointer
    
    def __init__(self):
        """Initialize ARM register file"""
        
        # General-purpose registers (32-bit each)
        # R0-R12 are shared across most modes
        # R13-R15 have mode-specific banking
        self.registers: List[int] = [0] * 16
        
        # Banked registers for different processor modes
        # Each mode has its own R13 (SP), R14 (LR), and some have SPSR
        self.banked_registers: Dict[ProcessorMode, Dict[str, int]] = {
            ProcessorMode.FIQ: {
                'R8': 0, 'R9': 0, 'R10': 0, 'R11': 0, 'R12': 0,  # Fast interrupt has more banked regs
                'R13': 0x10000000,  # FIQ SP
                'R14': 0,           # FIQ LR
                'SPSR': 0           # Saved Program Status Register
            },
            ProcessorMode.IRQ: {
                'R13': 0x08000000,  # IRQ SP
                'R14': 0,           # IRQ LR
                'SPSR': 0
            },
            ProcessorMode.SUPERVISOR: {
                'R13': 0x06000000,  # SVC SP
                'R14': 0,           # SVC LR
                'SPSR': 0
            },
            ProcessorMode.ABORT: {
                'R13': 0x04000000,  # ABT SP
                'R14': 0,           # ABT LR
                'SPSR': 0
            },
            ProcessorMode.UNDEFINED: {
                'R13': 0x02000000,  # UND SP
                'R14': 0,           # UND LR
                'SPSR': 0
            },
            ProcessorMode.SYSTEM: {
                'R13': 0x60000000,  # SYS SP (shared with user)
                'R14': 0            # SYS LR (shared with user)
            }
        }
        
        # Current Program Status Register
        self.cpsr = ConditionFlags(ProcessorMode.SUPERVISOR.value)  # Start in supervisor mode
        
        # Current processor mode
        self.current_mode = ProcessorMode.SUPERVISOR
        
        # Initialize stack pointers for different modes
        self.registers[13] = 0x60000000  # User/System SP
        
        # Statistics
        self.read_count = [0] * 16
        self.write_count = [0] * 16
        self.total_accesses = 0
        
        print("Initialized ARM RegisterFile")
        print(f"Initial mode: {self.current_mode.name}")
        print(f"Initial CPSR: {self.cpsr}")
    
    def read_register(self, reg_num: int) -> int:
        """
        Read value from register
        
        Args:
            reg_num: Register number (0-15)
            
        Returns:
            32-bit register value
        """
        if not (0 <= reg_num <= 15):
            raise ValueError(f"Invalid register number: {reg_num}")
        
        self.read_count[reg_num] += 1
        self.total_accesses += 1
        
        # Handle banked registers
        if self._is_banked_register(reg_num):
            return self._read_banked_register(reg_num)
        else:
            return self.registers[reg_num] & 0xFFFFFFFF
    
    def write_register(self, reg_num: int, value: int) -> None:
        """
        Write value to register
        
        Args:
            reg_num: Register number (0-15)
            value: 32-bit value to write
        """
        if not (0 <= reg_num <= 15):
            raise ValueError(f"Invalid register number: {reg_num}")
        
        self.write_count[reg_num] += 1
        self.total_accesses += 1
        
        # Ensure 32-bit value
        value = value & 0xFFFFFFFF
        
        # Handle banked registers
        if self._is_banked_register(reg_num):
            self._write_banked_register(reg_num, value)
        else:
            self.registers[reg_num] = value
        
        # Special handling for PC writes (clear bottom bits based on mode)
        if reg_num == self.PC:
            if self.cpsr.T:  # Thumb mode
                self.registers[reg_num] = value & 0xFFFFFFFE  # Clear bit 0
            else:  # ARM mode
                self.registers[reg_num] = value & 0xFFFFFFFC  # Clear bits 1:0
    
    def _is_banked_register(self, reg_num: int) -> bool:
        """Check if register is banked in current mode"""
        if self.current_mode == ProcessorMode.FIQ:
            return reg_num >= 8  # R8-R15 are banked in FIQ
        elif self.current_mode in [ProcessorMode.IRQ, ProcessorMode.SUPERVISOR, 
                                   ProcessorMode.ABORT, ProcessorMode.UNDEFINED]:
            return reg_num >= 13  # Only R13-R14 are banked
        else:
            return False  # User/System modes share registers
    
    def _read_banked_register(self, reg_num: int) -> int:
        """Read from banked register"""
        mode_regs = self.banked_registers.get(self.current_mode, {})
        reg_name = f'R{reg_num}'
        
        if reg_name in mode_regs:
            return mode_regs[reg_name] & 0xFFFFFFFF
        else:
            return self.registers[reg_num] & 0xFFFFFFFF
    
    def _write_banked_register(self, reg_num: int, value: int) -> None:
        """Write to banked register"""
        mode_regs = self.banked_registers.get(self.current_mode, {})
        reg_name = f'R{reg_num}'
        
        if reg_name in mode_regs:
            mode_regs[reg_name] = value & 0xFFFFFFFF
        else:
            self.registers[reg_num] = value & 0xFFFFFFFF
    
    def get_pc(self) -> int:
        """Get Program Counter value"""
        return self.read_register(self.PC)
    
    def set_pc(self, value: int) -> None:
        """Set Program Counter value"""
        self.write_register(self.PC, value)
    
    def get_sp(self) -> int:
        """Get Stack Pointer value"""
        return self.read_register(self.SP)
    
    def set_sp(self, value: int) -> None:
        """Set Stack Pointer value"""
        self.write_register(self.SP, value)
    
    def get_lr(self) -> int:
        """Get Link Register value"""
        return self.read_register(self.LR)
    
    def set_lr(self, value: int) -> None:
        """Set Link Register value"""
        self.write_register(self.LR, value)
    
    def get_cpsr(self) -> int:
        """Get Current Program Status Register"""
        return self.cpsr.value
    
    def set_cpsr(self, value: int) -> None:
        """Set Current Program Status Register"""
        old_mode = self.current_mode
        self.cpsr.value = value & 0xFFFFFFFF
        
        # Check if mode changed
        new_mode = self.cpsr.mode
        if new_mode != old_mode:
            self._change_mode(new_mode)
    
    def get_spsr(self) -> Optional[int]:
        """Get Saved Program Status Register for current mode"""
        if self.current_mode in [ProcessorMode.USER, ProcessorMode.SYSTEM]:
            return None  # User/System modes don't have SPSR
        
        mode_regs = self.banked_registers.get(self.current_mode, {})
        return mode_regs.get('SPSR', 0)
    
    def set_spsr(self, value: int) -> None:
        """Set Saved Program Status Register for current mode"""
        if self.current_mode in [ProcessorMode.USER, ProcessorMode.SYSTEM]:
            return  # User/System modes don't have SPSR
        
        mode_regs = self.banked_registers.get(self.current_mode, {})
        if mode_regs is not None:
            mode_regs['SPSR'] = value & 0xFFFFFFFF
    
    def _change_mode(self, new_mode: ProcessorMode) -> None:
        """Change processor mode and handle register banking"""
        if new_mode == self.current_mode:
            return
        
        old_mode = self.current_mode
        self.current_mode = new_mode
        
        print(f"Mode change: {old_mode.name} -> {new_mode.name}")
        
        # Handle stack pointer switching
        # This is a simplified implementation - real ARM has more complex banking
        if new_mode in self.banked_registers:
            mode_regs = self.banked_registers[new_mode]
            if 'R13' in mode_regs:
                self.registers[13] = mode_regs['R13']
    
    def update_flags_arithmetic(self, result: int, operand1: int, operand2: int, 
                              carry_in: bool = False, subtract: bool = False) -> None:
        """
        Update CPSR flags based on arithmetic operation result
        
        Args:
            result: Operation result (can be > 32 bits)
            operand1: First operand
            operand2: Second operand
            carry_in: Input carry flag
            subtract: True for subtraction operations
        """
        # Truncate result to 32 bits for flag calculation
        result_32 = result & 0xFFFFFFFF
        
        # Zero flag
        self.cpsr.Z = (result_32 == 0)
        
        # Negative flag
        self.cpsr.N = bool(result_32 & 0x80000000)
        
        if subtract:
            # For subtraction: operand1 - operand2
            # Carry flag: set if no borrow occurred
            self.cpsr.C = (operand1 >= operand2)
            
            # Overflow flag: set if sign of result is wrong
            # Overflow occurs when subtracting positive from negative gives positive
            # or subtracting negative from positive gives negative
            op1_sign = bool(operand1 & 0x80000000)
            op2_sign = bool(operand2 & 0x80000000)
            result_sign = bool(result_32 & 0x80000000)
            
            self.cpsr.V = (op1_sign != op2_sign) and (op1_sign != result_sign)
        else:
            # For addition: operand1 + operand2 + carry_in
            # Carry flag: set if result overflows 32 bits
            if carry_in:
                full_result = operand1 + operand2 + 1
            else:
                full_result = operand1 + operand2
            
            self.cpsr.C = (full_result > 0xFFFFFFFF)
            
            # Overflow flag: set if sign of result is wrong
            op1_sign = bool(operand1 & 0x80000000)
            op2_sign = bool(operand2 & 0x80000000)
            result_sign = bool(result_32 & 0x80000000)
            
            self.cpsr.V = (op1_sign == op2_sign) and (op1_sign != result_sign)
    
    def update_flags_logical(self, result: int, carry_out: Optional[bool] = None) -> None:
        """
        Update CPSR flags based on logical operation result
        
        Args:
            result: Operation result
            carry_out: Carry output from shifter (if applicable)
        """
        result_32 = result & 0xFFFFFFFF
        
        # Zero flag
        self.cpsr.Z = (result_32 == 0)
        
        # Negative flag
        self.cpsr.N = bool(result_32 & 0x80000000)
        
        # Carry flag (only updated if carry_out provided)
        if carry_out is not None:
            self.cpsr.C = carry_out
        
        # Overflow flag unchanged for logical operations
    
    def check_condition(self, condition: int) -> bool:
        """
        Check if condition code is satisfied by current flags
        
        Args:
            condition: 4-bit condition code
            
        Returns:
            True if condition is satisfied
        """
        condition = condition & 0xF
        
        if condition == 0x0:    # EQ - Equal (Z set)
            return self.cpsr.Z
        elif condition == 0x1:  # NE - Not equal (Z clear)
            return not self.cpsr.Z
        elif condition == 0x2:  # CS/HS - Carry set/unsigned higher or same
            return self.cpsr.C
        elif condition == 0x3:  # CC/LO - Carry clear/unsigned lower
            return not self.cpsr.C
        elif condition == 0x4:  # MI - Minus/negative (N set)
            return self.cpsr.N
        elif condition == 0x5:  # PL - Plus/positive or zero (N clear)
            return not self.cpsr.N
        elif condition == 0x6:  # VS - Overflow (V set)
            return self.cpsr.V
        elif condition == 0x7:  # VC - No overflow (V clear)
            return not self.cpsr.V
        elif condition == 0x8:  # HI - Unsigned higher (C set and Z clear)
            return self.cpsr.C and not self.cpsr.Z
        elif condition == 0x9:  # LS - Unsigned lower or same (C clear or Z set)
            return not self.cpsr.C or self.cpsr.Z
        elif condition == 0xA:  # GE - Signed greater than or equal (N == V)
            return self.cpsr.N == self.cpsr.V
        elif condition == 0xB:  # LT - Signed less than (N != V)
            return self.cpsr.N != self.cpsr.V
        elif condition == 0xC:  # GT - Signed greater than (Z clear and N == V)
            return not self.cpsr.Z and (self.cpsr.N == self.cpsr.V)
        elif condition == 0xD:  # LE - Signed less than or equal (Z set or N != V)
            return self.cpsr.Z or (self.cpsr.N != self.cpsr.V)
        elif condition == 0xE:  # AL - Always
            return True
        elif condition == 0xF:  # NV - Never (deprecated)
            return False
        
        return False
    
    def push_stack(self, value: int) -> None:
        """Push value onto current mode's stack"""
        sp = self.get_sp()
        sp -= 4
        self.set_sp(sp)
        # Note: In real implementation, this would write to memory
        # Here we just update SP
    
    def pop_stack(self) -> int:
        """Pop value from current mode's stack"""
        sp = self.get_sp()
        # Note: In real implementation, this would read from memory
        # Here we just update SP and return 0
        self.set_sp(sp + 4)
        return 0
    
    def reset(self) -> None:
        """Reset register file to initial state"""
        # Clear all registers
        self.registers = [0] * 16
        
        # Reset to supervisor mode
        self.cpsr = ConditionFlags(ProcessorMode.SUPERVISOR.value)
        self.current_mode = ProcessorMode.SUPERVISOR
        
        # Reset stack pointers
        self.registers[13] = 0x60000000
        for mode, regs in self.banked_registers.items():
            if 'R13' in regs:
                # Keep the initialized stack pointer values
                pass
        
        # Reset statistics
        self.read_count = [0] * 16
        self.write_count = [0] * 16
        self.total_accesses = 0
        
        print("Register file reset")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get register file statistics"""
        return {
            'total_accesses': self.total_accesses,
            'register_reads': self.read_count.copy(),
            'register_writes': self.write_count.copy(),
            'current_mode': self.current_mode.name,
            'cpsr': f"0x{self.cpsr.value:08x}",
            'most_accessed_register': self.read_count.index(max(self.read_count)) if self.total_accesses > 0 else None
        }
    
    def dump_registers(self) -> str:
        """Get formatted dump of all registers"""
        lines = []
        lines.append(f"=== Register File Dump ({self.current_mode.name} mode) ===")
        
        # General purpose registers
        for i in range(0, 16, 4):
            reg_line = []
            for j in range(4):
                if i + j < 16:
                    reg_num = i + j
                    value = self.read_register(reg_num)
                    if reg_num == 13:
                        reg_line.append(f"R{reg_num:2d}(SP): 0x{value:08x}")
                    elif reg_num == 14:
                        reg_line.append(f"R{reg_num:2d}(LR): 0x{value:08x}")
                    elif reg_num == 15:
                        reg_line.append(f"R{reg_num:2d}(PC): 0x{value:08x}")
                    else:
                        reg_line.append(f"R{reg_num:2d}: 0x{value:08x}")
            lines.append("  ".join(reg_line))
        
        # Status registers
        lines.append(f"\nCPSR: {self.cpsr}")
        
        spsr = self.get_spsr()
        if spsr is not None:
            spsr_flags = ConditionFlags(spsr)
            lines.append(f"SPSR: {spsr_flags}")
        
        return "\n".join(lines)
    
    def __str__(self) -> str:
        return f"RegisterFile(mode={self.current_mode.name}, PC=0x{self.get_pc():08x})"


# Example usage and testing
if __name__ == "__main__":
    print("Testing Register File\n")
    
    # Create register file
    rf = RegisterFile()
    
    # Test basic register operations
    print("Testing basic register operations...")
    rf.write_register(0, 0x12345678)
    rf.write_register(1, 0xDEADBEEF)
    
    value0 = rf.read_register(0)
    value1 = rf.read_register(1)
    
    print(f"R0: 0x{value0:08x}")
    print(f"R1: 0x{value1:08x}")
    
    # Test flag operations
    print("\nTesting flag operations...")
    rf.update_flags_arithmetic(0, 1, 1, subtract=True)  # 1 - 1 = 0
    print(f"After 1-1: Z={rf.cpsr.Z}, N={rf.cpsr.N}, C={rf.cpsr.C}, V={rf.cpsr.V}")
    
    # Test condition checking
    condition_eq = rf.check_condition(0x0)  # EQ condition
    print(f"EQ condition satisfied: {condition_eq}")
    
    # Test mode switching
    print("\nTesting mode switching...")
    rf.set_cpsr((rf.get_cpsr() & ~0x1F) | ProcessorMode.IRQ.value)
    print(f"Current mode: {rf.current_mode.name}")
    
    # Dump all registers
    print(f"\n{rf.dump_registers()}")
    
    # Print statistics
    stats = rf.get_statistics()
    print(f"\nStatistics: {stats}")
