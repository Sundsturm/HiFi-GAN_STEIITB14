#!/bin/bash
# =======================================================================
# Script: run_sim.sh
# Purpose: Run line_buffer testbench simulation using Icarus Verilog
# Usage: ./run_sim.sh
# =======================================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "======================================================================="
echo "Line Buffer Testbench Simulation"
echo "======================================================================="

# Check if iverilog is installed
if ! command -v iverilog &> /dev/null
then
    echo -e "${RED}Error: Icarus Verilog (iverilog) not found!${NC}"
    echo "Please install it:"
    echo "  Ubuntu/Debian: sudo apt-get install iverilog"
    echo "  MacOS: brew install icarus-verilog"
    exit 1
fi

# Clean previous build
echo -e "${YELLOW}Cleaning previous build...${NC}"
rm -f line_buffer_tb.vvp line_buffer_tb.vcd

# Compile the design
echo -e "${YELLOW}Compiling design...${NC}"
iverilog -o line_buffer_tb.vvp \
    -I ../code \
    ../code/line_buffer.v \
    line_buffer_tb.v

if [ $? -ne 0 ]; then
    echo -e "${RED}Compilation failed!${NC}"
    exit 1
fi

echo -e "${GREEN}Compilation successful!${NC}"

# Run simulation
echo -e "${YELLOW}Running simulation...${NC}"
vvp line_buffer_tb.vvp

if [ $? -ne 0 ]; then
    echo -e "${RED}Simulation failed!${NC}"
    exit 1
fi

echo -e "${GREEN}Simulation completed!${NC}"

# Check if VCD file was generated
if [ -f "line_buffer_tb.vcd" ]; then
    echo -e "${GREEN}Waveform file generated: line_buffer_tb.vcd${NC}"
    echo "View with: gtkwave line_buffer_tb.vcd"
else
    echo -e "${YELLOW}Warning: VCD file not generated${NC}"
fi

echo "======================================================================="
echo "Simulation Complete!"
echo "======================================================================="
