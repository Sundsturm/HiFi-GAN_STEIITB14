#!/bin/bash
# Build and run conv1d_engine_bram testbench

# Directories
TB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="${TB_DIR}/../code"
SHARED_DIR="${TB_DIR}/../../"

# Output files
VVP_FILE="${TB_DIR}/conv1d_engine_bram_tb.vvp"
VCD_FILE="${TB_DIR}/conv1d_engine_bram_tb.vcd"

echo "========================================="
echo "Building Conv1D BRAM Engine Testbench"
echo "========================================="

# Compile with iverilog
iverilog -g2005 \
    -o "${VVP_FILE}" \
    "${CODE_DIR}/conv1d_engine.v" \
    "${SHARED_DIR}/quantizer/code/quantizer_32_16.v" \
    "${SHARED_DIR}/mac_array/code/hifigan_mac_array.v" \
    "${SHARED_DIR}/mac_array/code/qmult.v" \
    "${SHARED_DIR}/activation_unit/code/leaky_relu_q15.v" \
    "${SHARED_DIR}/activation_unit/code/tanh_approx_q15.v" \
    "${TB_DIR}/conv1d_engine_tb.v"

if [ $? -eq 0 ]; then
    echo ""
    echo "Build successful! Running simulation..."
    echo "========================================="
    echo ""
    
    # Run simulation
    vvp "${VVP_FILE}"
    
    echo ""
    echo "========================================="
    echo "Waveform saved to: ${VCD_FILE}"
    echo "View with: gtkwave ${VCD_FILE}"
    echo "========================================="
else
    echo ""
    echo "[ERROR] Compilation failed!"
    exit 1
fi
