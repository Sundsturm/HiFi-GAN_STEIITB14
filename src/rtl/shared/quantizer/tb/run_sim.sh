#!/bin/bash
# Simulation script for quantizer_32_16 testbench using IVerilog + GTKWave

echo "========================================"
echo "Quantizer 32-16 Simulation"
echo "========================================"

# Clean previous simulation files
rm -f quantizer_32_16_tb.vcd
rm -f quantizer_sim

# Compile with IVerilog
echo "Compiling with IVerilog..."
iverilog -o quantizer_sim \
    -I../code \
    ../code/quantizer_32_16.v \
    quantizer_32_16_tb.v

if [ $? -ne 0 ]; then
    echo "ERROR: Compilation failed!"
    exit 1
fi

echo "Compilation successful!"

# Run simulation
echo ""
echo "Running simulation..."
vvp quantizer_sim

if [ $? -ne 0 ]; then
    echo "ERROR: Simulation failed!"
    exit 1
fi

echo ""
echo "Simulation complete!"

# Open GTKWave if VCD file exists
if [ -f quantizer_32_16_tb.vcd ]; then
    echo ""
    echo "Opening GTKWave..."
    gtkwave quantizer_32_16_tb.vcd &
else
    echo "WARNING: VCD file not generated!"
fi

echo "Done!"
