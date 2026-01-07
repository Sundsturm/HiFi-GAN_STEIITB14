# HiFi-GAN Shared Memory Modules

This directory contains the shared memory infrastructure for the HiFi-GAN hardware implementation.

## Module Overview

### 1. `weight_mem.v` - Weight Memory
**Purpose:** Stores convolution kernel weights for all layers

**Features:**
- 16-bit data width (Q2.14 fixed-point format)
- Single-port synchronous read BRAM
- Initialized from `.mem` file via `$readmemh`
- One-cycle read latency

**Interface:**
```verilog
Inputs:
  - i_addr  : Read address
  - i_rd_en : Read enable (synchronous)

Outputs:
  - o_data  : Weight data (Q2.14)
  - o_valid : Valid signal (delayed by 1 cycle)
```

**Memory Organization:**
Linear addressing: `addr = layer_offset + ch_out*ch_in*k_size + ch_in*k_size + k_pos`

### 2. `bias_mem.v` - Bias Memory
**Purpose:** Stores bias values for all layers

**Features:**
- 16-bit data width (Q4.12 fixed-point format)
- Single-port synchronous read BRAM
- Initialized from `.mem` file
- One-cycle read latency

**Interface:**
```verilog
Inputs:
  - i_addr  : Read address
  - i_rd_en : Read enable (synchronous)

Outputs:
  - o_data  : Bias data (Q4.12)
  - o_valid : Valid signal (delayed by 1 cycle)
```

**Memory Organization:**
Linear addressing: `addr = layer_offset + ch_out`

### 3. `param_rom.v` - Parameter ROM
**Purpose:** Stores static layer configuration parameters

**Features:**
- Combinational read (no clock latency)
- Pre-configured lookup table for layer metadata
- Supports up to 16 layers

**Interface:**
```verilog
Inputs:
  - i_layer_sel : Layer selection index

Outputs:
  - o_kernel_size  : Kernel size for layer
  - o_dilation     : Dilation factor
  - o_in_channels  : Number of input channels
  - o_out_channels : Number of output channels
```

## Fixed-Point Formats

| Data Type | Format | Range | Precision |
|-----------|--------|-------|-----------|
| Weights   | Q2.14  | [-2, 2) | 0.000061 |
| Biases    | Q4.12  | [-8, 8) | 0.000244 |
| Activations | Q4.12 | [-8, 8) | 0.000244 |
| Accumulator | Q6.26 | [-32, 32) | ~0.000000015 |

## Memory Initialization Files

### `weights.mem`
Format: One 16-bit hex value per line
```
0100  // Weight 0: 0.0625 in Q2.14
00C0  // Weight 1: 0.0469
0080  // Weight 2: 0.0313
...
```

### `biases.mem`
Format: One 16-bit hex value per line
```
0010  // Bias 0: 0.0039 in Q4.12
0020  // Bias 1: 0.0078
0015  // Bias 2: 0.0051
...
```

## Generating Memory Files

Use the provided Python script to generate `.mem` files:

```bash
python generate_mem_files.py
```

This creates:
- `weights.mem` - Random weights in [-0.1, 0.1] range
- `biases.mem` - Random biases in [-0.01, 0.01] range

## PostNet Memory Layout Example

For a 5-layer PostNet with configuration:
- Layer 0: 1 → 32 channels, kernel=5 → 160 weights, 32 biases
- Layer 1-3: 32 → 32 channels, kernel=5 → 5120 weights each, 32 biases each
- Layer 4: 32 → 1 channel, kernel=5 → 160 weights, 1 bias

**Weight Memory:**
- Total: 15,680 weights
- Address range: 0x0000 - 0x3D3F

**Bias Memory:**
- Total: 129 biases
- Address range: 0x00 - 0x80

## Usage in FSM

Sequential fetch pattern (as used in `postnet_stack.v`):

```verilog
// State: FETCH_WEIGHT
weight_addr <= base_addr + kernel_pos;
weight_rd_en <= 1'b1;

// Next cycle: weight_valid = 1, store in buffer
if (weight_valid)
    weight_buffer[kernel_pos] <= weight_data;

// State: COMPUTE
// Use accumulated weight_buffer for MAC operation
```

## Synthesis Notes

- **Inference:** Vivado infers BRAM from synchronous read reg arrays
- **Width:** 16-bit naturally maps to RAMB18E1 primitives
- **Depth:** Adjust DEPTH parameter based on model size
- **Dual-port:** Can extend to dual-port if needed (two independent reads)

## Testing

Run the testbench:
```bash
cd tb
iverilog -o memory_tb memory_tb.v ../weight_mem.v ../bias_mem.v ../param_rom.v
vvp memory_tb
gtkwave memory_tb.vcd
```

Or in Vivado:
```tcl
add_files memory_tb.v
add_files ../weight_mem.v ../bias_mem.v ../param_rom.v
launch_simulation
run all
```

## Directory Structure

```
shared/memory/
├── weight_mem.v          # Weight memory module
├── bias_mem.v            # Bias memory module
├── param_rom.v           # Parameter ROM module
└── tb/
    └── memory_tb.v       # Testbench for memory modules
```
