# HiFi-GAN Memory System - Complete Guide

[Previous README content replaced with comprehensive guide - see src/rtl/shared/memory/MEMORY_SYSTEM.md]

## Quick Start

1. **Generate memory files:**
```bash
python generate_memory_files.py
```

2. **Check output:**
- weights.mem (1.46M entries)
- biases.mem (1,825 entries)  
- hifigan_addr_map.vh (address constants)

3. **Use in RTL:**
```verilog
`include "hifigan_addr_map.vh"

weight_mem #(.DEPTH(1462497)) u_weights (...);
bias_mem #(.DEPTH(1825)) u_biases (...);
```

## Key Files Created

✅ **generate_memory_files.py** - CSV parser and generator  
✅ **hifigan_addr_map.vh** - Verilog address constants  
✅ **weights.mem** - 1.46M weight parameters  
✅ **biases.mem** - 1,825 bias parameters  
✅ **weight_mem.v** - Enhanced weight memory module  
✅ **bias_mem.v** - Enhanced bias memory module  
✅ **hifigan_addr_calculator.v** - Address calculation helper  
✅ **memory_config.txt** - Configuration summary  

## Memory Statistics

- Total Parameters: 1,464,322
- Total Memory: ~2.8 MB @ 16-bit
- Weight Format: Q2.14 fixed-point
- Bias Format: Q4.12 fixed-point

See memory_config.txt for detailed breakdown.
