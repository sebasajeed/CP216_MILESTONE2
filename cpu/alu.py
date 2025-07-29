from enum import Enum
from typing import Tuple, Optional, Any, Dict
import struct


class ALUOperation(Enum):
    """ALU operation types"""
    # Arithmetic operations
    ADD = "add"
    ADC = "adc"      # Add with carry
    SUB = "sub"
    SBC = "sbc"      # Subtract with carry
    RSB = "rsb"      # Reverse subtract
    RSC = "rsc"      # Reverse subtract with carry
    
    # Logical operations
    AND = "and"
    ORR = "orr"      # Logical OR
    EOR = "eor"      # Exclusive OR
    BIC = "bic"      # Bit clear (AND NOT)
    
    # Comparison operations
    CMP = "cmp"      # Compare (SUB but don't store result)
    CMN = "cmn"      # Compare negative (ADD but don't store result)
    TST = "tst"      # Test (AND but don't store result)
    TEQ = "teq"      # Test equivalence (EOR but don't store result)
    
    # Data movement operations
    MOV = "mov"      # Move
    MVN = "mvn"      # Move NOT


class ShiftType(Enum):
    """ARM shift types"""
    LSL = "lsl"      # Logical shift left
    LSR = "lsr"      # Logical shift right
    ASR = "asr"      # Arithmetic shift right
    ROR = "ror"      # Rotate right
    RRX = "rrx"      # Rotate right with extend


class ALUResult:
    """Result of an ALU operation"""
    
    def __init__(self, result: int, carry_out: bool = False, 
                 overflow: bool = False, cycles: int = 1):
        self.result = result & 0xFFFFFFFF  # 32-bit result
        self.carry_out = carry_out
        self.overflow = overflow
        self.cycles = cycles


class Shifter:
    """
    ARM barrel shifter implementation
    
    Handles all ARM shift and rotate operations with carry output
    """
    
    @staticmethod
    def shift(value: int, shift_type: ShiftType, shift_amount: int, 
              carry_in: bool = False) -> Tuple[int, bool]:
        """
        Perform shift operation
        
        Args:
            value: 32-bit value to shift
            shift_type: Type of shift operation
            shift_amount: Number of positions to shift
            carry_in: Input carry flag
            
        Returns:
            Tuple of (shifted_value, carry_out)
        """
        value = value & 0xFFFFFFFF
        shift_amount = shift_amount & 0xFF  # Limit to 8 bits
        
        if shift_amount == 0:
            # Special case: no shift
            if shift_type == ShiftType.RRX:
                # RRX with shift_amount 0 is a single bit rotate through carry
                carry_out = bool(value & 1)
                result = (value >> 1) | (0x80000000 if carry_in else 0)
                return result, carry_out
            else:
                return value, carry_in
        
        if shift_type == ShiftType.LSL:
            # Logical shift left
            if shift_amount >= 32:
                if shift_amount == 32:
                    carry_out = bool(value & 1)
                else:
                    carry_out = False
                result = 0
            else:
                carry_out = bool(value & (1 << (32 - shift_amount)))
                result = (value << shift_amount) & 0xFFFFFFFF
            
        elif shift_type == ShiftType.LSR:
            # Logical shift right
            if shift_amount >= 32:
                if shift_amount == 32:
                    carry_out = bool(value & 0x80000000)
                else:
                    carry_out = False
                result = 0
            else:
                carry_out = bool(value & (1 << (shift_amount - 1)))
                result = value >> shift_amount
        
        elif shift_type == ShiftType.ASR:
            # Arithmetic shift right
            if shift_amount >= 32:
                if value & 0x80000000:  # Negative number
                    carry_out = True
                    result = 0xFFFFFFFF
                else:  # Positive number
                    carry_out = False
                    result = 0
            else:
                carry_out = bool(value & (1 << (shift_amount - 1)))
                if value & 0x80000000:  # Sign extend
                    result = (value >> shift_amount) | (0xFFFFFFFF << (32 - shift_amount))
                else:
                    result = value >> shift_amount
                result &= 0xFFFFFFFF
        
        elif shift_type == ShiftType.ROR:
            # Rotate right
            shift_amount = shift_amount % 32
            if shift_amount == 0:
                return value, carry_in
            
            carry_out = bool(value & (1 << (shift_amount - 1)))
            result = ((value >> shift_amount) | (value << (32 - shift_amount))) & 0xFFFFFFFF
        
        elif shift_type == ShiftType.RRX:
            # Rotate right with extend (single bit rotate through carry)
            carry_out = bool(value & 1)
            result = (value >> 1) | (0x80000000 if carry_in else 0)
        
        else:
            raise ValueError(f"Unknown shift type: {shift_type}")
        
        return result, carry_out


class ALU:
    """
    ARM Arithmetic Logic Unit implementation
    
    Provides all arithmetic, logical, and comparison operations
    required by the ARM instruction set with proper flag handling.
    """
    
    def __init__(self):
        """Initialize ALU"""
        self.shifter = Shifter()
        
        # Statistics
        self.operation_count = {op: 0 for op in ALUOperation}
        self.total_operations = 0
        self.total_cycles = 0
        
        print("Initialized ARM ALU")
    
    def execute(self, operation: ALUOperation, operand1: int, operand2: int = 0,
                carry_in: bool = False, update_flags: bool = False) -> ALUResult:
        """
        Execute ALU operation
        
        Args:
            operation: ALU operation to perform
            operand1: First operand (32-bit)
            operand2: Second operand (32-bit)
            carry_in: Input carry flag
            update_flags: Whether to calculate flag outputs
            
        Returns:
            ALUResult with operation result and flags
        """
        # Update statistics
        self.operation_count[operation] += 1
        self.total_operations += 1
        
        # Ensure 32-bit operands
        operand1 = operand1 & 0xFFFFFFFF
        operand2 = operand2 & 0xFFFFFFFF
        
        # Perform operation
        if operation == ALUOperation.ADD:
            return self._add(operand1, operand2, carry_in, update_flags)
        elif operation == ALUOperation.ADC:
            return self._add(operand1, operand2, carry_in, update_flags, use_carry=True)
        elif operation == ALUOperation.SUB:
            return self._subtract(operand1, operand2, carry_in, update_flags)
        elif operation == ALUOperation.SBC:
            return self._subtract(operand1, operand2, carry_in, update_flags, use_carry=True)
        elif operation == ALUOperation.RSB:
            return self._subtract(operand2, operand1, carry_in, update_flags)
        elif operation == ALUOperation.RSC:
            return self._subtract(operand2, operand1, carry_in, update_flags, use_carry=True)
        elif operation == ALUOperation.AND:
            return self._logical_and(operand1, operand2, carry_in, update_flags)
        elif operation == ALUOperation.ORR:
            return self._logical_or(operand1, operand2, carry_in, update_flags)
        elif operation == ALUOperation.EOR:
            return self._logical_eor(operand1, operand2, carry_in, update_flags)
        elif operation == ALUOperation.BIC:
            return self._logical_bic(operand1, operand2, carry_in, update_flags)
        elif operation == ALUOperation.CMP:
            return self._subtract(operand1, operand2, carry_in, True, store_result=False)
        elif operation == ALUOperation.CMN:
            return self._add(operand1, operand2, carry_in, True, store_result=False)
        elif operation == ALUOperation.TST:
            return self._logical_and(operand1, operand2, carry_in, True, store_result=False)
        elif operation == ALUOperation.TEQ:
            return self._logical_eor(operand1, operand2, carry_in, True, store_result=False)
        elif operation == ALUOperation.MOV:
            return self._move(operand2, carry_in, update_flags)
        elif operation == ALUOperation.MVN:
            return self._move_not(operand2, carry_in, update_flags)
        else:
            raise ValueError(f"Unsupported ALU operation: {operation}")
    
    def _add(self, op1: int, op2: int, carry_in: bool, update_flags: bool,
             use_carry: bool = False, store_result: bool = True) -> ALUResult:
        """Perform addition operation"""
        if use_carry and carry_in:
            result = op1 + op2 + 1
        else:
            result = op1 + op2
        
        carry_out = result > 0xFFFFFFFF
        overflow = self._check_add_overflow(op1, op2, result & 0xFFFFFFFF)
        
        cycles = 1
        self.total_cycles += cycles
        
        return ALUResult(
            result if store_result else 0,
            carry_out if update_flags else False,
            overflow if update_flags else False,
            cycles
        )
    
    def _subtract(self, op1: int, op2: int, carry_in: bool, update_flags: bool,
                  use_carry: bool = False, store_result: bool = True) -> ALUResult:
        """Perform subtraction operation"""
        if use_carry and not carry_in:
            result = op1 - op2 - 1
        else:
            result = op1 - op2
        
        # For subtraction, carry is set if no borrow occurred
        carry_out = result >= 0
        
        # Convert negative results to unsigned representation
        if result < 0:
            result = (1 << 32) + result
        
        overflow = self._check_sub_overflow(op1, op2, result & 0xFFFFFFFF)
        
        cycles = 1
        self.total_cycles += cycles
        
        return ALUResult(
            result if store_result else 0,
            carry_out if update_flags else False,
            overflow if update_flags else False,
            cycles
        )
    
    def _logical_and(self, op1: int, op2: int, carry_in: bool, update_flags: bool,
                     store_result: bool = True) -> ALUResult:
        """Perform logical AND operation"""
        result = op1 & op2
        
        cycles = 1
        self.total_cycles += cycles
        
        return ALUResult(
            result if store_result else 0,
            carry_in if update_flags else False,  # Carry unchanged for logical ops
            False,  # Overflow unchanged for logical ops
            cycles
        )
    
    def _logical_or(self, op1: int, op2: int, carry_in: bool, update_flags: bool,
                    store_result: bool = True) -> ALUResult:
        """Perform logical OR operation"""
        result = op1 | op2
        
        cycles = 1
        self.total_cycles += cycles
        
        return ALUResult(
            result if store_result else 0,
            carry_in if update_flags else False,
            False,
            cycles
        )
    
    def _logical_eor(self, op1: int, op2: int, carry_in: bool, update_flags: bool,
                     store_result: bool = True) -> ALUResult:
        """Perform logical exclusive OR operation"""
        result = op1 ^ op2
        
        cycles = 1
        self.total_cycles += cycles
        
        return ALUResult(
            result if store_result else 0,
            carry_in if update_flags else False,
            False,
            cycles
        )
    
    def _logical_bic(self, op1: int, op2: int, carry_in: bool, update_flags: bool,
                     store_result: bool = True) -> ALUResult:
        """Perform bit clear operation (AND NOT)"""
        result = op1 & (~op2 & 0xFFFFFFFF)
        
        cycles = 1
        self.total_cycles += cycles
        
        return ALUResult(
            result if store_result else 0,
            carry_in if update_flags else False,
            False,
            cycles
        )
    
    def _move(self, op2: int, carry_in: bool, update_flags: bool) -> ALUResult:
        """Perform move operation"""
        result = op2
        
        cycles = 1
        self.total_cycles += cycles
        
        return ALUResult(
            result,
            carry_in if update_flags else False,
            False,
            cycles
        )
    
    def _move_not(self, op2: int, carry_in: bool, update_flags: bool) -> ALUResult:
        """Perform move NOT operation"""
        result = (~op2) & 0xFFFFFFFF
        
        cycles = 1
        self.total_cycles += cycles
        
        return ALUResult(
            result,
            carry_in if update_flags else False,
            False,
            cycles
        )
    
    def _check_add_overflow(self, op1: int, op2: int, result: int) -> bool:
        """Check for signed overflow in addition"""
        op1_sign = bool(op1 & 0x80000000)
        op2_sign = bool(op2 & 0x80000000)
        result_sign = bool(result & 0x80000000)
        
        # Overflow occurs when adding two numbers of the same sign
        # produces a result with a different sign
        return (op1_sign == op2_sign) and (op1_sign != result_sign)
    
    def _check_sub_overflow(self, op1: int, op2: int, result: int) -> bool:
        """Check for signed overflow in subtraction"""
        op1_sign = bool(op1 & 0x80000000)
        op2_sign = bool(op2 & 0x80000000)
        result_sign = bool(result & 0x80000000)
        
        # Overflow occurs when subtracting numbers of different signs
        # produces a result with the wrong sign
        return (op1_sign != op2_sign) and (op1_sign != result_sign)
    
    def multiply(self, op1: int, op2: int, accumulate: int = 0, 
                 signed: bool = False, long_result: bool = False) -> Tuple[int, Optional[int], int]:
        """
        Perform multiplication operation
        
        Args:
            op1: First operand
            op2: Second operand
            accumulate: Value to add to result (for MLA/SMLAL/UMLAL)
            signed: Whether to perform signed multiplication
            long_result: Whether to return 64-bit result
            
        Returns:
            Tuple of (result_low, result_high_or_none, cycles)
        """
        self.total_operations += 1
        
        op1 = op1 & 0xFFFFFFFF
        op2 = op2 & 0xFFFFFFFF
        
        if signed:
            # Convert to signed values
            if op1 & 0x80000000:
                op1 = op1 - (1 << 32)
            if op2 & 0x80000000:
                op2 = op2 - (1 << 32)
            
            result = op1 * op2 + accumulate
        else:
            result = op1 * op2 + accumulate
        
        # Calculate cycles (simplified timing model)
        cycles = 2  # Base multiply cycles
        if long_result:
            cycles += 1
        if accumulate != 0:
            cycles += 1
        
        self.total_cycles += cycles
        
        if long_result:
            # 64-bit result
            if result < 0:
                result = (1 << 64) + result
            
            result_low = result & 0xFFFFFFFF
            result_high = (result >> 32) & 0xFFFFFFFF
            return result_low, result_high, cycles
        else:
            # 32-bit result
            result = result & 0xFFFFFFFF
            return result, None, cycles
    
    def shift_operand(self, value: int, shift_type: ShiftType, 
                     shift_amount: int, carry_in: bool = False) -> Tuple[int, bool]:
        """
        Apply shift to operand using barrel shifter
        
        Args:
            value: Value to shift
            shift_type: Type of shift
            shift_amount: Amount to shift
            carry_in: Input carry flag
            
        Returns:
            Tuple of (shifted_value, carry_out)
        """
        return self.shifter.shift(value, shift_type, shift_amount, carry_in)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get ALU statistics"""
        return {
            'total_operations': self.total_operations,
            'total_cycles': self.total_cycles,
            'average_cycles_per_operation': (self.total_cycles / max(1, self.total_operations)),
            'operation_counts': {op.value: count for op, count in self.operation_count.items()},
            'most_used_operation': max(self.operation_count.items(), key=lambda x: x[1])[0].value if self.total_operations > 0 else None
        }
    
    def reset_statistics(self):
        """Reset ALU statistics"""
        self.operation_count = {op: 0 for op in ALUOperation}
        self.total_operations = 0
        self.total_cycles = 0
    
    def print_status(self):
        """Print ALU status"""
        stats = self.get_statistics()
        
        print(f"\n=== ALU Status ===")
        print(f"Total Operations: {stats['total_operations']}")
        print(f"Total Cycles: {stats['total_cycles']}")
        print(f"Average Cycles/Op: {stats['average_cycles_per_operation']:.2f}")
        
        if stats['most_used_operation']:
            print(f"Most Used Operation: {stats['most_used_operation']}")
        
        # Print top operations
        sorted_ops = sorted(stats['operation_counts'].items(), key=lambda x: x[1], reverse=True)
        print("Operation Usage:")
        for op, count in sorted_ops[:5]:  # Top 5
            if count > 0:
                print(f"  {op}: {count}")
    
    def __str__(self) -> str:
        return f"ALU(operations={self.total_operations}, cycles={self.total_cycles})"


# Example usage and testing
if __name__ == "__main__":
    print("Testing ARM ALU\n")
    
    # Create ALU
    alu = ALU()
    
    # Test arithmetic operations
    print("Testing arithmetic operations...")
    
    # Addition
    result = alu.execute(ALUOperation.ADD, 10, 20, update_flags=True)
    print(f"10 + 20 = {result.result}, carry={result.carry_out}, overflow={result.overflow}")
    
    # Subtraction
    result = alu.execute(ALUOperation.SUB, 30, 10, update_flags=True)
    print(f"30 - 10 = {result.result}, carry={result.carry_out}")
    
    # Test overflow
    result = alu.execute(ALUOperation.ADD, 0x7FFFFFFF, 1, update_flags=True)
    print(f"0x7FFFFFFF + 1 = 0x{result.result:08x}, overflow={result.overflow}")
    
    # Test logical operations
    print("\nTesting logical operations...")
    
    result = alu.execute(ALUOperation.AND, 0xFF00FF00, 0x0F0F0F0F, update_flags=True)
    print(f"0xFF00FF00 & 0x0F0F0F0F = 0x{result.result:08x}")
    
    result = alu.execute(ALUOperation.ORR, 0xFF00FF00, 0x0F0F0F0F, update_flags=True)
    print(f"0xFF00FF00 | 0x0F0F0F0F = 0x{result.result:08x}")
    
    # Test shifts
    print("\nTesting shift operations...")
    
    shifted, carry = alu.shift_operand(0x80000001, ShiftType.LSR, 1)
    print(f"0x80000001 LSR #1 = 0x{shifted:08x}, carry={carry}")
    
    shifted, carry = alu.shift_operand(0x80000001, ShiftType.ASR, 1)
    print(f"0x80000001 ASR #1 = 0x{shifted:08x}, carry={carry}")
    
    # Test multiplication
    print("\nTesting multiplication...")
    
    result_low, result_high, cycles = alu.multiply(0x12345678, 0x9ABCDEF0, long_result=True)
    print(f"0x12345678 * 0x9ABCDEF0 = 0x{result_high:08x}{result_low:08x} ({cycles} cycles)")
    
    # Print statistics
    alu.print_status()
