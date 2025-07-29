# ARM CPU Simulator

A comprehensive ARM processor simulator implementing a complete CPU core with cache memory system, instruction set architecture, and performance analysis capabilities.

## Project Overview

This project implements a cycle-accurate ARM CPU simulator designed for educational purposes and performance analysis. The simulator includes:

- **Complete ARM CPU Core**: Fetch-decode-execute pipeline with exception handling
- **Memory System**: Configurable main memory with realistic timing
- **Cache System**: Multi-level cache hierarchy with various replacement policies
- **Instruction Set**: ARM instruction encoding/decoding and execution
- **Performance Analysis**: Comprehensive statistics and bottleneck identification

## Features

### CPU Core
- ARM7/ARM9-style processor architecture
- 16 general-purpose registers (R0-R15) with mode-specific banking
- Current Program Status Register (CPSR) with condition flags
- Multiple processor modes (User, Supervisor, IRQ, FIQ, etc.)
- Pipeline simulation with branch prediction
- Exception handling and interrupt support

### Memory System
- Configurable main memory (default: 64MB)
- Multiple memory regions (ROM, RAM, I/O, Stack)
- Realistic access latencies and bandwidth limitations
- Memory protection and access validation

### Cache System
- Configurable cache hierarchies (L1, L2)
- Multiple cache organizations:
  - Direct-mapped
  - Set-associative (2-way, 4-way, 8-way, 16-way)
  - Fully-associative
- Replacement policies: LRU, FIFO, Random, LFU
- Write policies: Write-through, Write-back
- Comprehensive cache statistics and analysis

### Instruction Set Architecture
- Complete ARM instruction set support:
  - Data processing (arithmetic, logical)
  - Memory operations (load/store single/multiple)
  - Branch and branch-with-link
  - Multiply and multiply-accumulate
  - Software interrupts
- Instruction encoding/decoding utilities
- Simple ARM assembler for test programs

### Performance Analysis
- Cycle-accurate timing simulation
- Instruction mix analysis
- Cache performance metrics
- Branch prediction analysis
- Bottleneck identification
- Performance optimization recommendations

## Installation

### Prerequisites
- Python 3.8 or higher
- No external dependencies required (uses only Python standard library)

### Setup
1. Clone or download the project
2. Navigate to the project directory
3. Optionally install development dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Command Line Interface

Run a simulation with default settings:
```bash
python cpu_simulator/main.py
```

Run with custom configuration:
```bash
python cpu_simulator/main.py --memory-size 32MB --cache-config l2_unified --max-cycles 10000
```

Load and run a custom assembly program:
```bash
python cpu_simulator/main.py --program my_program.s --stats-file results.json
```

Command line options:
- `--memory-size`: Memory size in bytes (default: 64MB)
- `--no-cache`: Disable cache simulation
- `--cache-config`: Cache configuration (l1_data, l1_instruction, l2_unified, direct_mapped)
- `--max-cycles`: Maximum cycles to simulate
- `--max-instructions`: Maximum instructions to execute
- `--program`: Assembly program file to load
- `--stats-file`: File to save simulation statistics

### Programmatic Interface

```python
from cpu_simulator.main import CPUSimulator

# Create simulator
simulator = CPUSimulator(
    memory_size=64*1024*1024,
    enable_cache=True,
    cache_config="l1_data"
)

# Load a program
assembly_code = """
    MOV R0, #10
    MOV R1, #20
    ADD R2, R0, R1
    B .
"""
simulator.load_program_assembly(assembly_code)

# Run simulation
stats = simulator.run_simulation(max_cycles=1000)

# Analyze results
simulator.print_status()
analysis = simulator.analyze_performance()
print(f"CPI: {analysis['performance_metrics']['cpi']:.3f}")
```

## Architecture

### Project Structure
```
CP216_MILESTONE2/
├── cpu/                    # CPU core implementation
│   ├── alu.py             # Arithmetic Logic Unit
│   ├── cpu.py             # Main CPU core
│   └── register_file.py   # Register file and CPSR
├── memory/                 # Memory system
│   └── memory.py          # Main memory implementation
├── cache/                  # Cache system
│   ├── cache_block.py     # Cache blocks and sets
│   ├── cache_config.py    # Cache configuration
│   └── cache.py           # Main cache implementation
├── isa/                   # Instruction Set Architecture
│   ├── arm_instructions.py # ARM instruction support
│   ├── decoder.py         # Instruction decoder
│   └── thumb_instructions.py # Thumb instruction support
├── cpu_simulator/         # Main simulator interface
│   └── main.py           # Entry point and CLI
├── benchmark/             # Benchmarking utilities
│   └── run_benchmarks.py # Performance benchmarks
└── output/               # Simulation outputs
    └── cost_summary.txt  # Cost analysis results
```

### Key Components

#### CPU Core (`cpu/cpu.py`)
The main CPU class integrates all components and provides the fetch-decode-execute cycle:
- Pipeline simulation with configurable stages
- Exception handling and mode switching
- Instruction execution and timing
- Performance counter management

#### Cache System (`cache/`)
Implements a flexible cache hierarchy:
- `CacheConfig`: Configuration management
- `CacheBlock`: Individual cache lines with metadata
- `Cache`: Main cache controller with replacement policies

#### Memory System (`memory/memory.py`)
Provides realistic memory simulation:
- Multiple memory regions with different properties
- Configurable access latencies and bandwidth
- Memory protection and error handling

#### Instruction Set (`isa/arm_instructions.py`)
Complete ARM instruction support:
- Instruction encoding and decoding
- Assembler for test programs
- Execution semantics for all instruction types

## Sample Programs

### Simple Arithmetic
```assembly
; Calculate sum of first 5 numbers
MOV R0, #0      ; Sum accumulator
MOV R1, #1      ; Counter
MOV R2, #5      ; Limit

loop:
ADD R0, R0, R1  ; Add counter to sum
ADD R1, R1, #1  ; Increment counter
SUB R3, R1, R2  ; Compare with limit
BLE loop        ; Branch if less than or equal
```

### Memory Operations
```assembly
; Array processing example
MOV R0, #0x1000 ; Array base address
MOV R1, #10     ; Array size
MOV R2, #0      ; Sum accumulator
MOV R3, #0      ; Index

sum_loop:
LDR R4, [R0, R3, LSL #2]  ; Load array element
ADD R2, R2, R4            ; Add to sum
ADD R3, R3, #1            ; Increment index
CMP R3, R1                ; Compare with size
BLT sum_loop              ; Continue if less than
```

## Performance Analysis

The simulator provides detailed performance metrics:

### CPU Metrics
- **CPI (Cycles Per Instruction)**: Average cycles needed per instruction
- **Instruction Mix**: Distribution of instruction types
- **Branch Statistics**: Branch frequency and prediction accuracy
- **Pipeline Efficiency**: Stall cycles and utilization

### Memory Metrics  
- **Access Patterns**: Read/write distribution and locality
- **Bandwidth Utilization**: Memory system efficiency
- **Region Usage**: Activity in different memory areas

### Cache Metrics
- **Hit Rate**: Percentage of cache hits vs misses
- **Miss Penalty**: Average cost of cache misses
- **Replacement Efficiency**: How well the replacement policy works
- **Spatial/Temporal Locality**: Memory access patterns

### Example Analysis Output
```
=== PERFORMANCE ANALYSIS ===
CPI: 1.45
Instruction Throughput: 2,500,000 inst/sec
Cache Hit Rate: 0.892
Branch Mispredict Rate: 0.056

Bottlenecks identified:
  - High branch mispredict rate: 0.056
  - Memory access latency contributing to CPI

Recommendations:
  - Consider branch prediction optimization
  - Optimize code for better cache locality
```

## Configuration

### Cache Configuration
The simulator supports various cache configurations:

```python
# L1 Data Cache (32KB, 8-way set associative)
config = CacheConfigPresets.l1_data_cache()

# L2 Unified Cache (256KB, 16-way set associative)
config = CacheConfigPresets.l2_unified_cache()

# Direct-mapped cache (16KB)
config = CacheConfigPresets.direct_mapped_cache()

# Custom configuration
config = CacheConfig(
    size=64*1024,           # 64KB
    block_size=64,          # 64-byte blocks
    associativity=4,        # 4-way set associative
    replacement_policy=ReplacementPolicy.LRU,
    write_policy=WritePolicy.WRITE_BACK,
    hit_time=2,             # 2 cycles
    miss_penalty=20         # 20 cycles
)
```

### Memory Configuration
```python
# Default memory regions
memory = MainMemory(64*1024*1024)  # 64MB total

# Add custom memory region
memory.add_region(
    start_address=0x50000000,
    size=1024*1024,           # 1MB
    name="CUSTOM",
    readable=True,
    writable=True,
    executable=False,
    latency_cycles=50
)
```

## Benchmarking

The simulator includes benchmarking utilities for performance evaluation:

```bash
python benchmark/run_benchmarks.py
```

This runs various test programs and generates performance reports comparing different configurations.

## Development

### Adding New Instructions
1. Extend the instruction decoder in `isa/decoder.py`
2. Add execution logic in `cpu/cpu.py`
3. Update the assembler in `isa/arm_instructions.py`
4. Add test cases

### Adding Cache Policies
1. Define new policy in `cache/cache_config.py`
2. Implement logic in `cache/cache.py`
3. Update block replacement in `cache/cache_block.py`

### Performance Optimization
- Use profiling tools to identify bottlenecks
- Optimize hot paths in the execution loop
- Consider just-in-time compilation for instruction dispatch

## Testing

Run the built-in tests:
```bash
python -m pytest tests/
```

Test individual components:
```bash
python cpu/alu.py        # Test ALU
python cache/cache.py    # Test cache system
python memory/memory.py  # Test memory system
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is designed for educational purposes. Please refer to your institution's policies regarding academic use.

## Authors

- CPU Simulator Project Team
- Computer Architecture Course (CP216)

## Acknowledgments

- ARM Architecture Reference Manuals
- Computer Architecture: A Quantitative Approach (Hennessy & Patterson)
- Modern Processor Design (Shen & Lipasti)

---

For more detailed information, refer to the inline documentation in each module or contact the development team.
