import time
import struct
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

from .alu import ALU, ALUOperation, ShiftType
from .register_file import RegisterFile, ProcessorMode, ConditionFlags


class CPUState(Enum):
    """CPU execution states"""
    RUNNING = "running"
    HALTED = "halted"
    RESET = "reset"
    EXCEPTION = "exception"
    DEBUG = "debug"


class ExceptionType(Enum):
    """ARM exception types"""
    RESET = "reset"
    UNDEFINED_INSTRUCTION = "undefined_instruction"
    SOFTWARE_INTERRUPT = "software_interrupt"
    PREFETCH_ABORT = "prefetch_abort"
    DATA_ABORT = "data_abort"
    IRQ = "irq"
    FIQ = "fiq"


class PipelineStage(Enum):
    """ARM pipeline stages"""
    FETCH = "fetch"
    DECODE = "decode"
    EXECUTE = "execute"
    MEMORY = "memory"
    WRITEBACK = "writeback"


class InstructionFormat(Enum):
    """ARM instruction formats"""
    DATA_PROCESSING = "data_processing"
    MULTIPLY = "multiply"
    LOAD_STORE = "load_store"
    LOAD_STORE_MULTIPLE = "load_store_multiple"
    BRANCH = "branch"
    COPROCESSOR = "coprocessor"
    SOFTWARE_INTERRUPT = "software_interrupt"


class ARMInstruction:
    """Represents a decoded ARM instruction"""
    
    def __init__(self, raw_instruction: int, address: int):
        self.raw = raw_instruction
        self.address = address
        self.condition = (raw_instruction >> 28) & 0xF
        self.format = self._identify_format()
        self.cycles = 1  # Default cycle count
        self.decoded = False
    
    def _identify_format(self) -> InstructionFormat:
        """Identify instruction format from bit pattern"""
        bits_27_25 = (self.raw >> 25) & 0x7
        bits_7_4 = (self.raw >> 4) & 0xF
        
        if bits_27_25 == 0b000:
            if bits_7_4 == 0b1001:
                return InstructionFormat.MULTIPLY
            else:
                return InstructionFormat.DATA_PROCESSING
        elif bits_27_25 == 0b001:
            return InstructionFormat.DATA_PROCESSING
        elif bits_27_25 in [0b010, 0b011]:
            return InstructionFormat.LOAD_STORE
        elif bits_27_25 == 0b100:
            return InstructionFormat.LOAD_STORE_MULTIPLE
        elif bits_27_25 == 0b101:
            return InstructionFormat.BRANCH
        elif bits_27_25 == 0b110:
            return InstructionFormat.COPROCESSOR
        elif bits_27_25 == 0b111:
            return InstructionFormat.SOFTWARE_INTERRUPT
        else:
            return InstructionFormat.DATA_PROCESSING  # Default
    
    def __str__(self) -> str:
        return f"Instruction(0x{self.raw:08x} @ 0x{self.address:08x}, {self.format.value})"


class ARMCPU:
    """
    ARM CPU Core Implementation
    
    Provides a complete ARM processor simulation with:
    - Fetch-decode-execute cycle
    - Pipeline simulation
    - Exception handling
    - Memory interface
    - Cache interface
    """
    
    def __init__(self, memory_interface=None, cache_interface=None):
        """
        Initialize ARM CPU
        
        Args:
            memory_interface: Interface to main memory
            cache_interface: Interface to cache system
        """
        # Core components
        self.alu = ALU()
        self.registers = RegisterFile()
        
        # Interfaces
        self.memory = memory_interface
        self.cache = cache_interface
        
        # CPU state
        self.state = CPUState.RESET
        self.exception_pending = None
        
        # Pipeline state
        self.pipeline_enabled = True
        self.pipeline_stages = {stage: None for stage in PipelineStage}
        self.pipeline_stall_cycles = 0
        
        # Performance counters
        self.cycle_count = 0
        self.instruction_count = 0
        self.branch_count = 0
        self.branch_mispredict_count = 0
        self.memory_access_count = 0
        self.cache_hit_count = 0
        self.cache_miss_count = 0
        
        # Instruction statistics
        self.instruction_stats = {fmt: 0 for fmt in InstructionFormat}
        
        # Exception vectors (standard ARM addresses)
        self.exception_vectors = {
            ExceptionType.RESET: 0x00000000,
            ExceptionType.UNDEFINED_INSTRUCTION: 0x00000004,
            ExceptionType.SOFTWARE_INTERRUPT: 0x00000008,
            ExceptionType.PREFETCH_ABORT: 0x0000000C,
            ExceptionType.DATA_ABORT: 0x00000010,
            ExceptionType.IRQ: 0x00000018,
            ExceptionType.FIQ: 0x0000001C
        }
        
        # Debugging support
        self.breakpoints = set()
        self.single_step = False
        self.instruction_trace = []
        self.max_trace_length = 1000
        
        # Initialize CPU state
        self.reset()
        
        print("Initialized ARM CPU Core")
        print(f"Initial PC: 0x{self.registers.get_pc():08x}")
        print(f"Initial mode: {self.registers.current_mode.name}")
    
    def reset(self):
        """Reset CPU to initial state"""
        # Reset components
        self.registers.reset()
        self.alu.reset_statistics()
        
        # Set initial state
        self.state = CPUState.RUNNING
        self.exception_pending = None
        
        # Set reset vector
        self.registers.set_pc(self.exception_vectors[ExceptionType.RESET])
        
        # Start in supervisor mode with interrupts disabled
        cpsr = self.registers.get_cpsr()
        cpsr = (cpsr & ~0x1F) | ProcessorMode.SUPERVISOR.value
        cpsr |= (1 << 7) | (1 << 6)  # Disable IRQ and FIQ
        self.registers.set_cpsr(cpsr)
        
        # Clear pipeline
        self.pipeline_stages = {stage: None for stage in PipelineStage}
        
        # Reset statistics
        self.cycle_count = 0
        self.instruction_count = 0
        self.branch_count = 0
        self.branch_mispredict_count = 0
        self.memory_access_count = 0
        self.cache_hit_count = 0
        self.cache_miss_count = 0
        self.instruction_stats = {fmt: 0 for fmt in InstructionFormat}
        
        print("CPU Reset")
    
    def step(self) -> bool:
        """
        Execute one CPU cycle
        
        Returns:
            True if CPU should continue, False if halted
        """
        if self.state != CPUState.RUNNING:
            return False
        
        self.cycle_count += 1
        
        # Handle pending exceptions
        if self.exception_pending:
            self._handle_exception(self.exception_pending)
            self.exception_pending = None
            return True
        
        # Handle pipeline stalls
        if self.pipeline_stall_cycles > 0:
            self.pipeline_stall_cycles -= 1
            return True
        
        # Execute pipeline stages
        if self.pipeline_enabled:
            self._execute_pipeline()
        else:
            # Simple fetch-decode-execute without pipeline
            self._execute_simple()
        
        # Check for breakpoints
        if self.registers.get_pc() in self.breakpoints:
            self.state = CPUState.DEBUG
            print(f"Breakpoint hit at 0x{self.registers.get_pc():08x}")
            return False
        
        # Check for single step
        if self.single_step:
            self.state = CPUState.DEBUG
            return False
        
        return True
    
    def run(self, max_cycles: Optional[int] = None, max_instructions: Optional[int] = None) -> Dict[str, Any]:
        """
        Run CPU until halt condition
        
        Args:
            max_cycles: Maximum cycles to execute
            max_instructions: Maximum instructions to execute
            
        Returns:
            Execution statistics
        """
        start_time = time.time()
        start_cycles = self.cycle_count
        start_instructions = self.instruction_count
        
        print(f"Starting CPU execution from PC=0x{self.registers.get_pc():08x}")
        
        while self.step():
            # Check limits
            if max_cycles and (self.cycle_count - start_cycles) >= max_cycles:
                print(f"Reached maximum cycles limit: {max_cycles}")
                break
            
            if max_instructions and (self.instruction_count - start_instructions) >= max_instructions:
                print(f"Reached maximum instructions limit: {max_instructions}")
                break
            
            # Periodic status update
            if (self.cycle_count - start_cycles) % 10000 == 0 and (self.cycle_count - start_cycles) > 0:
                print(f"Executed {self.cycle_count - start_cycles} cycles, "
                      f"{self.instruction_count - start_instructions} instructions")
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Execution statistics
        cycles_executed = self.cycle_count - start_cycles
        instructions_executed = self.instruction_count - start_instructions
        
        stats = {
            'cycles_executed': cycles_executed,
            'instructions_executed': instructions_executed,
            'execution_time_seconds': execution_time,
            'cycles_per_second': cycles_executed / max(execution_time, 0.001),
            'instructions_per_second': instructions_executed / max(execution_time, 0.001),
            'average_cpi': cycles_executed / max(instructions_executed, 1),
            'final_pc': f"0x{self.registers.get_pc():08x}",
            'final_state': self.state.value
        }
        
        return stats
    
    def _execute_pipeline(self):
        """Execute pipeline stages"""
        # Writeback stage
        if self.pipeline_stages[PipelineStage.WRITEBACK]:
            self._writeback_stage(self.pipeline_stages[PipelineStage.WRITEBACK])
        
        # Memory stage
        self.pipeline_stages[PipelineStage.WRITEBACK] = self.pipeline_stages[PipelineStage.MEMORY]
        if self.pipeline_stages[PipelineStage.MEMORY]:
            self._memory_stage(self.pipeline_stages[PipelineStage.MEMORY])
        
        # Execute stage
        self.pipeline_stages[PipelineStage.MEMORY] = self.pipeline_stages[PipelineStage.EXECUTE]
        if self.pipeline_stages[PipelineStage.EXECUTE]:
            self._execute_stage(self.pipeline_stages[PipelineStage.EXECUTE])
        
        # Decode stage
        self.pipeline_stages[PipelineStage.EXECUTE] = self.pipeline_stages[PipelineStage.DECODE]
        if self.pipeline_stages[PipelineStage.DECODE]:
            self._decode_stage(self.pipeline_stages[PipelineStage.DECODE])
        
        # Fetch stage
        self.pipeline_stages[PipelineStage.DECODE] = self.pipeline_stages[PipelineStage.FETCH]
        if not self.pipeline_stall_cycles:  # Don't fetch if stalled
            self.pipeline_stages[PipelineStage.FETCH] = self._fetch_stage()
    
    def _execute_simple(self):
        """Simple fetch-decode-execute without pipeline"""
        # Fetch
        instruction = self._fetch_instruction(self.registers.get_pc())
        if not instruction:
            self.state = CPUState.HALTED
            return
        
        # Increment PC
        self.registers.set_pc(self.registers.get_pc() + 4)
        
        # Decode and execute
        self._decode_instruction(instruction)
        if self.registers.check_condition(instruction.condition):
            self._execute_instruction(instruction)
        
        self.instruction_count += 1
    
    def _fetch_stage(self) -> Optional[ARMInstruction]:
        """Fetch instruction from memory"""
        pc = self.registers.get_pc()
        
        # Check alignment
        if pc & 0x3:
            self._raise_exception(ExceptionType.PREFETCH_ABORT)
            return None
        
        instruction = self._fetch_instruction(pc)
        if instruction:
            # Increment PC for next fetch
            self.registers.set_pc(pc + 4)
        
        return instruction
    
    def _decode_stage(self, instruction: ARMInstruction):
        """Decode instruction"""
        if instruction and not instruction.decoded:
            self._decode_instruction(instruction)
            instruction.decoded = True
    
    def _execute_stage(self, instruction: ARMInstruction):
        """Execute instruction"""
        if instruction and self.registers.check_condition(instruction.condition):
            self._execute_instruction(instruction)
            self.instruction_count += 1
    
    def _memory_stage(self, instruction: ARMInstruction):
        """Memory access stage"""
        # This would handle load/store memory accesses
        # For now, memory accesses are handled in execute stage
        pass
    
    def _writeback_stage(self, instruction: ARMInstruction):
        """Writeback stage"""
        # Register writes are handled in execute stage
        # This stage could handle forwarding logic in a real implementation
        pass
    
    def _fetch_instruction(self, address: int) -> Optional[ARMInstruction]:
        """Fetch 32-bit instruction from memory"""
        try:
            if self.cache:
                # Try cache first
                result = self.cache.read(address, 4, self.memory)
                if result.hit:
                    self.cache_hit_count += 1
                else:
                    self.cache_miss_count += 1
                instruction_data = result.data
                self.cycle_count += result.cycles - 1  # Subtract 1 since main step() adds 1
            elif self.memory:
                # Direct memory access
                instruction_data = self.memory.read(address, 4)
                self.cycle_count += 10  # Memory access penalty
            else:
                # No memory interface - return NOP
                instruction_data = b'\\x00\\x00\\xa0\\xe1'  # NOP instruction
            
            self.memory_access_count += 1
            
            # Convert bytes to 32-bit instruction (little-endian)
            raw_instruction = int.from_bytes(instruction_data, byteorder='little')
            
            return ARMInstruction(raw_instruction, address)
            
        except Exception as e:
            print(f"Fetch error at 0x{address:08x}: {e}")
            self._raise_exception(ExceptionType.PREFETCH_ABORT)
            return None
    
    def _decode_instruction(self, instruction: ARMInstruction):
        """Decode instruction fields"""
        # Update instruction statistics
        self.instruction_stats[instruction.format] += 1
        
        # Add to instruction trace
        if len(self.instruction_trace) >= self.max_trace_length:
            self.instruction_trace.pop(0)
        
        self.instruction_trace.append({
            'address': instruction.address,
            'instruction': instruction.raw,
            'format': instruction.format.value,
            'cycle': self.cycle_count
        })
    
    def _execute_instruction(self, instruction: ARMInstruction):
        """Execute decoded instruction"""
        try:
            if instruction.format == InstructionFormat.DATA_PROCESSING:
                self._execute_data_processing(instruction)
            elif instruction.format == InstructionFormat.MULTIPLY:
                self._execute_multiply(instruction)
            elif instruction.format == InstructionFormat.LOAD_STORE:
                self._execute_load_store(instruction)
            elif instruction.format == InstructionFormat.LOAD_STORE_MULTIPLE:
                self._execute_load_store_multiple(instruction)
            elif instruction.format == InstructionFormat.BRANCH:
                self._execute_branch(instruction)
            elif instruction.format == InstructionFormat.SOFTWARE_INTERRUPT:
                self._execute_software_interrupt(instruction)
            else:
                # Unsupported instruction
                self._raise_exception(ExceptionType.UNDEFINED_INSTRUCTION)
        
        except Exception as e:
            print(f"Execution error: {e}")
            self._raise_exception(ExceptionType.UNDEFINED_INSTRUCTION)
    
    def _execute_data_processing(self, instruction: ARMInstruction):
        """Execute data processing instruction"""
        # Extract fields
        immediate = bool(instruction.raw & (1 << 25))
        opcode = (instruction.raw >> 21) & 0xF
        set_flags = bool(instruction.raw & (1 << 20))
        rn = (instruction.raw >> 16) & 0xF
        rd = (instruction.raw >> 12) & 0xF
        
        # Get first operand
        operand1 = self.registers.read_register(rn)
        
        # Get second operand
        if immediate:
            # Immediate value with rotation
            imm_value = instruction.raw & 0xFF
            rotate_amount = ((instruction.raw >> 8) & 0xF) * 2
            operand2, carry_out = self.alu.shift_operand(imm_value, ShiftType.ROR, rotate_amount, self.registers.cpsr.C)
        else:
            # Register operand with optional shift
            rm = instruction.raw & 0xF
            operand2 = self.registers.read_register(rm)
            
            # Apply shift if specified
            shift_type_bits = (instruction.raw >> 5) & 0x3
            shift_types = [ShiftType.LSL, ShiftType.LSR, ShiftType.ASR, ShiftType.ROR]
            shift_type = shift_types[shift_type_bits]
            
            if instruction.raw & (1 << 4):  # Register shift
                rs = (instruction.raw >> 8) & 0xF
                shift_amount = self.registers.read_register(rs) & 0xFF
            else:  # Immediate shift
                shift_amount = (instruction.raw >> 7) & 0x1F
            
            operand2, carry_out = self.alu.shift_operand(operand2, shift_type, shift_amount, self.registers.cpsr.C)
        
        # Map opcode to ALU operation
        alu_ops = [
            ALUOperation.AND,  # 0000 - AND
            ALUOperation.EOR,  # 0001 - EOR
            ALUOperation.SUB,  # 0010 - SUB
            ALUOperation.RSB,  # 0011 - RSB
            ALUOperation.ADD,  # 0100 - ADD
            ALUOperation.ADC,  # 0101 - ADC
            ALUOperation.SBC,  # 0110 - SBC
            ALUOperation.RSC,  # 0111 - RSC
            ALUOperation.TST,  # 1000 - TST
            ALUOperation.TEQ,  # 1001 - TEQ
            ALUOperation.CMP,  # 1010 - CMP
            ALUOperation.CMN,  # 1011 - CMN
            ALUOperation.ORR,  # 1100 - ORR
            ALUOperation.MOV,  # 1101 - MOV
            ALUOperation.BIC,  # 1110 - BIC
            ALUOperation.MVN   # 1111 - MVN
        ]
        
        if opcode < len(alu_ops):
            alu_op = alu_ops[opcode]
            
            # Execute ALU operation
            result = self.alu.execute(alu_op, operand1, operand2, 
                                    self.registers.cpsr.C, set_flags)
            
            # Update flags if required
            if set_flags:
                if alu_op in [ALUOperation.ADD, ALUOperation.ADC, ALUOperation.SUB, 
                             ALUOperation.SBC, ALUOperation.RSB, ALUOperation.RSC,
                             ALUOperation.CMP, ALUOperation.CMN]:
                    # Arithmetic operations
                    self.registers.update_flags_arithmetic(result.result, operand1, operand2,
                                                         self.registers.cpsr.C, 
                                                         alu_op in [ALUOperation.SUB, ALUOperation.SBC, 
                                                                   ALUOperation.CMP])
                else:
                    # Logical operations
                    self.registers.update_flags_logical(result.result, carry_out)
            
            # Write result to destination register (except for comparison operations)
            if alu_op not in [ALUOperation.CMP, ALUOperation.CMN, ALUOperation.TST, ALUOperation.TEQ]:
                self.registers.write_register(rd, result.result)
            
            # Add ALU cycles
            self.cycle_count += result.cycles - 1
    
    def _execute_multiply(self, instruction: ARMInstruction):
        """Execute multiply instruction"""
        # Extract fields
        accumulate = bool(instruction.raw & (1 << 21))
        set_flags = bool(instruction.raw & (1 << 20))
        rd = (instruction.raw >> 16) & 0xF
        rn = (instruction.raw >> 12) & 0xF
        rs = (instruction.raw >> 8) & 0xF
        rm = instruction.raw & 0xF
        
        # Get operands
        op1 = self.registers.read_register(rm)
        op2 = self.registers.read_register(rs)
        acc = self.registers.read_register(rn) if accumulate else 0
        
        # Perform multiplication
        result, _, cycles = self.alu.multiply(op1, op2, acc, signed=False, long_result=False)
        
        # Write result
        self.registers.write_register(rd, result)
        
        # Update flags if required
        if set_flags:
            self.registers.update_flags_logical(result)
        
        # Add multiply cycles
        self.cycle_count += cycles - 1
    
    def _execute_load_store(self, instruction: ARMInstruction):
        """Execute load/store instruction"""
        # This is a simplified implementation
        # Real ARM load/store has many variants and addressing modes
        
        immediate = not bool(instruction.raw & (1 << 25))
        pre_indexed = bool(instruction.raw & (1 << 24))
        up = bool(instruction.raw & (1 << 23))
        byte_access = bool(instruction.raw & (1 << 22))
        writeback = bool(instruction.raw & (1 << 21))
        load = bool(instruction.raw & (1 << 20))
        
        rn = (instruction.raw >> 16) & 0xF
        rd = (instruction.raw >> 12) & 0xF
        
        # Calculate address
        base_address = self.registers.read_register(rn)
        
        if immediate:
            offset = instruction.raw & 0xFFF
        else:
            # Register offset (simplified)
            rm = instruction.raw & 0xF
            offset = self.registers.read_register(rm)
        
        if not up:
            offset = -offset
        
        if pre_indexed:
            address = base_address + offset
        else:
            address = base_address
        
        # Perform memory access
        try:
            if load:
                # Load operation
                if byte_access:
                    if self.memory:
                        data = self.memory.read_byte(address)
                    else:
                        data = 0
                    self.registers.write_register(rd, data)
                else:
                    if self.memory:
                        data = self.memory.read_word(address)
                    else:
                        data = 0
                    self.registers.write_register(rd, data)
            else:
                # Store operation
                data = self.registers.read_register(rd)
                if byte_access:
                    if self.memory:
                        self.memory.write_byte(address, data & 0xFF)
                else:
                    if self.memory:
                        self.memory.write_word(address, data)
            
            # Update base register if writeback
            if writeback or not pre_indexed:
                new_base = base_address + offset
                self.registers.write_register(rn, new_base)
            
            self.memory_access_count += 1
            self.cycle_count += 2  # Memory access penalty
            
        except Exception as e:
            print(f"Memory access error: {e}")
            self._raise_exception(ExceptionType.DATA_ABORT)
    
    def _execute_load_store_multiple(self, instruction: ARMInstruction):
        """Execute load/store multiple instruction"""
        # Simplified implementation
        pre_indexed = bool(instruction.raw & (1 << 24))
        up = bool(instruction.raw & (1 << 23))
        psr_or_force_user = bool(instruction.raw & (1 << 22))
        writeback = bool(instruction.raw & (1 << 21))
        load = bool(instruction.raw & (1 << 20))
        
        rn = (instruction.raw >> 16) & 0xF
        register_list = instruction.raw & 0xFFFF
        
        base_address = self.registers.read_register(rn)
        address = base_address
        
        # Count registers
        register_count = bin(register_list).count('1')
        
        if not up:
            address -= register_count * 4
        
        # Process each register in the list
        for reg in range(16):
            if register_list & (1 << reg):
                if pre_indexed:
                    if up:
                        address += 4
                    else:
                        address -= 4
                
                try:
                    if load:
                        if self.memory:
                            data = self.memory.read_word(address)
                        else:
                            data = 0
                        self.registers.write_register(reg, data)
                    else:
                        data = self.registers.read_register(reg)
                        if self.memory:
                            self.memory.write_word(address, data)
                    
                    if not pre_indexed:
                        if up:
                            address += 4
                        else:
                            address -= 4
                
                except Exception as e:
                    print(f"Multiple memory access error: {e}")
                    self._raise_exception(ExceptionType.DATA_ABORT)
                    return
        
        # Update base register if writeback
        if writeback:
            if up:
                new_base = base_address + register_count * 4
            else:
                new_base = base_address - register_count * 4
            self.registers.write_register(rn, new_base)
        
        self.memory_access_count += register_count
        self.cycle_count += register_count + 1  # N+1 cycles for N registers
    
    def _execute_branch(self, instruction: ARMInstruction):
        """Execute branch instruction"""
        link = bool(instruction.raw & (1 << 24))
        
        # Extract 24-bit signed offset
        offset = instruction.raw & 0xFFFFFF
        if offset & 0x800000:  # Sign extend
            offset = offset | 0xFF000000
        
        # Convert to signed integer
        offset = struct.unpack('<i', struct.pack('<I', offset))[0]
        
        # Calculate target address
        current_pc = self.registers.get_pc()
        target_address = current_pc + (offset << 2)  # Offset is in words
        
        self.branch_count += 1
        
        # Save return address if link bit set
        if link:
            self.registers.set_lr(current_pc - 4)  # Return to instruction after branch
        
        # Update PC
        self.registers.set_pc(target_address)
        
        # Flush pipeline on branch
        if self.pipeline_enabled:
            self.pipeline_stages = {stage: None for stage in PipelineStage}
            self.pipeline_stall_cycles = 2  # Branch penalty
    
    def _execute_software_interrupt(self, instruction: ARMInstruction):
        """Execute software interrupt (SWI)"""
        self._raise_exception(ExceptionType.SOFTWARE_INTERRUPT)
    
    def _raise_exception(self, exception_type: ExceptionType):
        """Raise an exception"""
        print(f"Exception: {exception_type.value}")
        
        # Save current state
        current_cpsr = self.registers.get_cpsr()
        current_pc = self.registers.get_pc()
        
        # Determine new mode
        if exception_type == ExceptionType.FIQ:
            new_mode = ProcessorMode.FIQ
        elif exception_type == ExceptionType.IRQ:
            new_mode = ProcessorMode.IRQ
        elif exception_type in [ExceptionType.PREFETCH_ABORT, ExceptionType.DATA_ABORT]:
            new_mode = ProcessorMode.ABORT
        elif exception_type == ExceptionType.UNDEFINED_INSTRUCTION:
            new_mode = ProcessorMode.UNDEFINED
        else:
            new_mode = ProcessorMode.SUPERVISOR
        
        # Switch to new mode
        new_cpsr = (current_cpsr & ~0x1F) | new_mode.value
        new_cpsr |= (1 << 7)  # Disable IRQ
        if exception_type == ExceptionType.FIQ:
            new_cpsr |= (1 << 6)  # Disable FIQ
        
        self.registers.set_cpsr(new_cpsr)
        
        # Save return state
        self.registers.set_spsr(current_cpsr)
        self.registers.set_lr(current_pc)
        
        # Jump to exception vector
        vector_address = self.exception_vectors[exception_type]
        self.registers.set_pc(vector_address)
        
        # Clear pipeline
        if self.pipeline_enabled:
            self.pipeline_stages = {stage: None for stage in PipelineStage}
    
    def _handle_exception(self, exception_type: ExceptionType):
        """Handle pending exception"""
        self._raise_exception(exception_type)
    
    def halt(self):
        """Halt CPU execution"""
        self.state = CPUState.HALTED
        print("CPU Halted")
    
    def set_breakpoint(self, address: int):
        """Set breakpoint at address"""
        self.breakpoints.add(address)
        print(f"Breakpoint set at 0x{address:08x}")
    
    def clear_breakpoint(self, address: int):
        """Clear breakpoint at address"""
        self.breakpoints.discard(address)
        print(f"Breakpoint cleared at 0x{address:08x}")
    
    def enable_single_step(self):
        """Enable single step mode"""
        self.single_step = True
        print("Single step mode enabled")
    
    def disable_single_step(self):
        """Disable single step mode"""
        self.single_step = False
        print("Single step mode disabled")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive CPU statistics"""
        alu_stats = self.alu.get_statistics()
        reg_stats = self.registers.get_statistics()
        
        return {
            'cpu_state': self.state.value,
            'cycle_count': self.cycle_count,
            'instruction_count': self.instruction_count,
            'cpi': self.cycle_count / max(self.instruction_count, 1),
            'branch_count': self.branch_count,
            'branch_mispredict_count': self.branch_mispredict_count,
            'branch_mispredict_rate': self.branch_mispredict_count / max(self.branch_count, 1),
            'memory_access_count': self.memory_access_count,
            'cache_hit_count': self.cache_hit_count,
            'cache_miss_count': self.cache_miss_count,
            'cache_hit_rate': self.cache_hit_count / max(self.cache_hit_count + self.cache_miss_count, 1),
            'instruction_mix': {fmt.value: count for fmt, count in self.instruction_stats.items()},
            'alu_statistics': alu_stats,
            'register_statistics': reg_stats,
            'current_pc': f"0x{self.registers.get_pc():08x}",
            'current_mode': self.registers.current_mode.name
        }
    
    def print_status(self):
        """Print comprehensive CPU status"""
        stats = self.get_statistics()
        
        print(f"\n=== CPU Status ===")
        print(f"State: {stats['cpu_state']}")
        print(f"PC: {stats['current_pc']}")
        print(f"Mode: {stats['current_mode']}")
        print(f"Cycles: {stats['cycle_count']}")
        print(f"Instructions: {stats['instruction_count']}")
        print(f"CPI: {stats['cpi']:.2f}")
        print(f"Branches: {stats['branch_count']} (mispredict rate: {stats['branch_mispredict_rate']:.3f})")
        print(f"Memory Accesses: {stats['memory_access_count']}")
        print(f"Cache Hit Rate: {stats['cache_hit_rate']:.3f}")
        
        print(f"\nInstruction Mix:")
        for fmt, count in stats['instruction_mix'].items():
            if count > 0:
                percentage = (count / stats['instruction_count']) * 100
                print(f"  {fmt}: {count} ({percentage:.1f}%)")
        
        print(f"\n{self.registers.dump_registers()}")
    
    def __str__(self) -> str:
        return f"ARMCPU(PC=0x{self.registers.get_pc():08x}, state={self.state.value}, cycles={self.cycle_count})"


# Example usage and testing
if __name__ == "__main__":
    print("Testing ARM CPU Core\n")
    
    # Create CPU without memory interface for basic testing
    cpu = ARMCPU()
    
    # Create a simple test program in memory simulation
    # This would normally be loaded from actual memory
    test_instructions = [
        0xE3A00001,  # MOV R0, #1
        0xE3A01002,  # MOV R1, #2  
        0xE0802001,  # ADD R2, R0, R1
        0xEAFFFFFE   # B . (infinite loop)
    ]
    
    # Simulate program execution
    print("Simulating simple program execution...")
    
    # Set initial PC
    cpu.registers.set_pc(0x8000)
    
    # Run for a few cycles
    stats = cpu.run(max_cycles=20)
    
    print(f"\nExecution completed:")
    print(f"Cycles: {stats['cycles_executed']}")
    print(f"Instructions: {stats['instructions_executed']}")
    print(f"CPI: {stats['average_cpi']:.2f}")
    
    # Print final status
    cpu.print_status()
