# Conv1D Engine - BRAM Version

## Overview

Simplified Conv1D engine for HiFi-GAN vocoder implementation on Zynq FPGA.
This version uses BRAM interface for batch processing, perfect for PS-PL integration.

## Architecture

- **Design**: Batch-mode convolution with BRAM read/write interface
- **Size**: ~280 lines (down from 547 in streaming version)
- **FSM**: 7 simple states (vs 12 in old version)
- **Interface**: Direct BRAM addressing (no streaming handshake complexity)

## Files

### Code (`code/`)
- `conv1d_engine.v` - Main engine (BRAM-based, multi-channel Conv1D)
- `archive_old/` - Previous implementations (streaming version, simple version)

### Testbench (`tb/`)
- `conv1d_engine_tb.v` - Comprehensive testbench with 4 test cases
- `run_sim.sh` / `run_sim.ps1` - Build and run scripts
- `archive_old_streaming/` - Old streaming testbench files

## Building and Testing

```bash
cd tb
# Linux/WSL:
bash run_sim.sh

# Windows PowerShell (if execution policy allows):
.\run_sim.ps1

# Or direct compilation:
iverilog -g2005 -o conv1d_engine_tb.vvp \
    ../code/conv1d_engine.v \
    ../../quantizer/code/quantizer_32_16.v \
    ../../activation_unit/code/leaky_relu_q15.v \
    ../../activation_unit/code/tanh_approx_q15.v \
    conv1d_engine_tb.v
vvp conv1d_engine_tb.vvp
```

## Integration Example

```verilog
conv1d_engine #(
    .KERNEL_SIZE(7),
    .ACTIVATION("LEAKY_RELU")
) conv_layer (
    .clk(clk),
    .rst_n(rst_n),
    
    // Control
    .start(start),
    .done(done),
    .busy(busy),
    
    // Configuration
    .seq_length(seq_len),
    .in_channels(in_ch),
    .out_channels(out_ch),
    .kernel_size(k_size),
    .dilation(dil),
    
    // BRAM interfaces
    .input_addr(in_addr),
    .input_rd_en(in_rd),
    .input_data(in_data),
    
    .output_addr(out_addr),
    .output_wr_en(out_wr),
    .output_data(out_data),
    
    .weight_addr(w_addr),
    .weight_data(weights[w_addr]),
    
    .bias_addr(b_addr),
    .bias_data(biases[b_addr])
);
```

## Performance

All tests pass with <100 LSB tolerance (Q4.12 fixed-point):

| Test | Config | Cycles |
|------|--------|--------|
| 1 | K=3, IN=2→OUT=3, SEQ=8 | 650 |
| 2 | K=5, IN=4→OUT=8, SEQ=10 | 5,362 |
| 3 | K=3, IN=1→OUT=1, SEQ=6 | 92 |
| 4 | K=3, IN=4→OUT=4, SEQ=8, DIL=2 | 1,634 |

## Design Rationale

### Why BRAM Interface?

**Target**: Zynq SoC (ARM PS + FPGA PL)
- ARM runs PyTorch/MATLAB training
- FPGA accelerates inference
- Data via DMA → BRAM → Conv engine

**BRAM Benefits**:
- ✅ Simpler: 7 states vs 12
- ✅ No streaming handshake overhead
- ✅ Batch processing (matches PS→PL workflow)
- ✅ Easy FSM control integration

### Multi-Channel Processing

Essential for HiFi-GAN (80→512 channels):

```
For each timestep t:
  For each output channel o:
    acc = 0
    For each input channel i:
      For kernel position k:
        acc += input[t+k*dil][i] * weight[o][i][k]
    output[t][o] = quantize(acc + bias[o])
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DATA_WIDTH` | 16 | Q4.12 fixed-point |
| `KERNEL_SIZE` | 3 | Max kernel size |
| `MAX_IN_CH` | 256 | Max input channels |
| `MAX_OUT_CH` | 512 | Max output channels |
| `MAX_SEQ_LEN` | 256 | Max sequence length |
| `ACTIVATION` | "NONE" | "LEAKY_RELU", "TANH", "NONE" |

## Dependencies

- `../../quantizer/code/quantizer_32_16.v` - Q6.26 → Q4.12 with saturation
- `../../activation_unit/code/leaky_relu_q15.v` - LeakyReLU (optional)
- `../../activation_unit/code/tanh_approx_q15.v` - Tanh via LUT (optional)

## Notes

- Pure Verilog-2001 (synthesizable)
- Xilinx Vivado target
- Fixed-point throughout
- Hardware-sharing: one engine for all layers
