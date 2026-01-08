#!/bin/bash

# =============================================================================
# Script: run_sim.sh
# Purpose: Compile and run MAC array testbench with IVerilog and GTKWave
# Usage: ./run_sim.sh
# =============================================================================

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "============================================================================="
echo "MAC Array Simulation Script"
echo "============================================================================="

# Clean previous build
echo -e "${YELLOW}Cleaning previous build...${NC}"
rm -f mac_array_sim
rm -f hifigan_mac_array_tb.vcd

# Compile with IVerilog
echo -e "${YELLOW}Compiling with IVerilog...${NC}"
iverilog -g2001 \
    -o mac_array_sim \
    -I../code \
    ../code/qmult.v \
    ../code/hifigan_mac_array.v \
    hifigan_mac_array_tb.v

# Check compilation status
if [ $? -ne 0 ]; then
    echo -e "${RED}[ERROR] Compilation failed!${NC}"
    exit 1
fi

echo -e "${GREEN}[SUCCESS] Compilation complete${NC}"
echo ""

# Run simulation
echo -e "${YELLOW}Running simulation...${NC}"
echo "-----------------------------------------------------------------------------"
vvp mac_array_sim
echo "-----------------------------------------------------------------------------"

# Check if VCD file was generated
if [ -f "hifigan_mac_array_tb.vcd" ]; then
    echo -e "${GREEN}[SUCCESS] VCD file generated: hifigan_mac_array_tb.vcd${NC}"
    echo ""
    echo "To view waveforms, run:"
    echo "  gtkwave hifigan_mac_array_tb.vcd"
else
    echo -e "${RED}[ERROR] VCD file not generated${NC}"
    exit 1
fi

echo ""
echo "============================================================================="
echo "Simulation Complete"
echo "============================================================================="
