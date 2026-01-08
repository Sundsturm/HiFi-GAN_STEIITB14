#!/bin/bash
# Run conv1d_engine testbench simulation

echo "======================================================================="
echo "Conv1D Engine Testbench Simulation"
echo "======================================================================="

# Clean
rm -f conv1d_engine_tb.vvp conv1d_engine_tb.vcd

# Compile
echo "Compiling..."
iverilog -o conv1d_engine_tb.vvp \
    -I ../../line_buffer/code \
    -I ../../mac_array/code \
    -I ../../quantizer/code \
    ../../line_buffer/code/line_buffer.v \
    ../../mac_array/code/hifigan_mac_array.v \
    ../../mac_array/code/qmult.v \
    ../../quantizer/code/quantizer_32_16.v \
    ../code/conv1d_engine_simple.v \
    conv1d_engine_tb.v

if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

echo "Compilation successful!"

# Run
echo "Running simulation..."
vvp conv1d_engine_tb.vvp

if [ $? -ne 0 ]; then
    echo "Simulation failed!"
    exit 1
fi

echo "Simulation completed!"
if [ -f "conv1d_engine_tb.vcd" ]; then
    echo "Waveform: conv1d_engine_tb.vcd"
    echo "View with: gtkwave conv1d_engine_tb.vcd"
fi

echo "======================================================================="
