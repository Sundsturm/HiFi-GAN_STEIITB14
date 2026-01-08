#!/bin/bash
#==============================================================================
# Simulation Run Script for PostNet Components
# Purpose: Run Icarus Verilog or ModelSim/Questa simulations
# Usage: ./run_postnet_sim.sh [stack|top|all] [icarus|modelsim]
#==============================================================================

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
MODULE="all"
SIMULATOR="icarus"

# Parse command line arguments
if [ $# -ge 1 ]; then
    MODULE=$1
fi

if [ $# -ge 2 ]; then
    SIMULATOR=$2
fi

# Directory setup
RTL_DIR="../"
SHARED_DIR="../../shared"
TB_DIR="."
SIM_DIR="./sim_output"

# Create simulation output directory
mkdir -p $SIM_DIR

echo "=========================================="
echo "PostNet Simulation Script"
echo "=========================================="
echo "Module: $MODULE"
echo "Simulator: $SIMULATOR"
echo "Output Directory: $SIM_DIR"
echo "=========================================="

#------------------------------------------------------------------------------
# Function: Compile and Run with Icarus Verilog
#------------------------------------------------------------------------------
run_icarus() {
    local testbench=$1
    local module_name=$2
    
    echo -e "\n${YELLOW}[Icarus Verilog] Compiling $module_name...${NC}"
    
    # Compile command
    iverilog -g2005 \
        -o $SIM_DIR/${module_name}.vvp \
        -I$RTL_DIR \
        -I$SHARED_DIR \
        $TB_DIR/${testbench}.v \
        $RTL_DIR/postnet_stack.v \
        $RTL_DIR/postnet_top.v \
        $RTL_DIR/postnet_fsm.v \
        $SHARED_DIR/activation_unit/code/tanh_approx_q15.v \
        $SHARED_DIR/activation_unit/code/leaky_relu_q15.v \
        $SHARED_DIR/activation_unit/code/pwl_activation.v \
        $SHARED_DIR/mac_array/code/hifigan_mac_array.v \
        $SHARED_DIR/mac_array/code/qmult.v \
        $SHARED_DIR/memory/weight_mem.v \
        $SHARED_DIR/memory/bias_mem.v \
        $SHARED_DIR/quantizer/code/quantizer_32_16.v
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}[ERROR] Compilation failed for $module_name${NC}"
        return 1
    fi
    
    echo -e "${GREEN}[Icarus Verilog] Compilation successful${NC}"
    echo -e "${YELLOW}[Icarus Verilog] Running simulation...${NC}"
    
    # Run simulation
    cd $SIM_DIR
    vvp ${module_name}.vvp | tee ${module_name}_sim.log
    RESULT=$?
    cd ..
    
    if [ $RESULT -eq 0 ]; then
        echo -e "${GREEN}[Icarus Verilog] Simulation completed${NC}"
        
        # Check for VCD file
        if [ -f "$SIM_DIR/${module_name}.vcd" ]; then
            echo -e "${GREEN}Waveform saved: $SIM_DIR/${module_name}.vcd${NC}"
            echo "View with: gtkwave $SIM_DIR/${module_name}.vcd"
        fi
    else
        echo -e "${RED}[ERROR] Simulation failed for $module_name${NC}"
        return 1
    fi
    
    return 0
}

#------------------------------------------------------------------------------
# Function: Compile and Run with ModelSim/Questa
#------------------------------------------------------------------------------
run_modelsim() {
    local testbench=$1
    local module_name=$2
    
    echo -e "\n${YELLOW}[ModelSim] Compiling $module_name...${NC}"
    
    # Create work library
    if [ ! -d "$SIM_DIR/work" ]; then
        vlib $SIM_DIR/work
    fi
    
    vmap work $SIM_DIR/work
    
    # Compile all source files
    vlog -work work \
        +incdir+$RTL_DIR \
        +incdir+$SHARED_DIR \
        $RTL_DIR/postnet_stack.v \
        $RTL_DIR/postnet_top.v \
        $RTL_DIR/postnet_fsm.v \
        $SHARED_DIR/activation_unit/code/tanh_approx_q15.v \
        $SHARED_DIR/activation_unit/code/leaky_relu_q15.v \
        $SHARED_DIR/activation_unit/code/pwl_activation.v \
        $SHARED_DIR/mac_array/code/hifigan_mac_array.v \
        $SHARED_DIR/mac_array/code/qmult.v \
        $SHARED_DIR/memory/weight_mem.v \
        $SHARED_DIR/memory/bias_mem.v \
        $SHARED_DIR/quantizer/code/quantizer_32_16.v \
        $TB_DIR/${testbench}.v
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}[ERROR] Compilation failed for $module_name${NC}"
        return 1
    fi
    
    echo -e "${GREEN}[ModelSim] Compilation successful${NC}"
    echo -e "${YELLOW}[ModelSim] Running simulation...${NC}"
    
    # Run simulation
    vsim -c -do "run -all; quit" work.${testbench} | tee $SIM_DIR/${module_name}_sim.log
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}[ModelSim] Simulation completed${NC}"
    else
        echo -e "${RED}[ERROR] Simulation failed for $module_name${NC}"
        return 1
    fi
    
    return 0
}

#------------------------------------------------------------------------------
# Main Simulation Flow
#------------------------------------------------------------------------------
ERRORS=0

case $MODULE in
    stack)
        echo -e "\n${YELLOW}===== Running PostNet Stack Simulation =====${NC}"
        if [ "$SIMULATOR" == "icarus" ]; then
            run_icarus "postnet_stack_tb" "postnet_stack_tb"
        else
            run_modelsim "postnet_stack_tb" "postnet_stack_tb"
        fi
        ERRORS=$?
        ;;
        
    top)
        echo -e "\n${YELLOW}===== Running PostNet Top Simulation =====${NC}"
        if [ "$SIMULATOR" == "icarus" ]; then
            run_icarus "postnet_top_tb" "postnet_top_tb"
        else
            run_modelsim "postnet_top_tb" "postnet_top_tb"
        fi
        ERRORS=$?
        ;;
        
    all)
        echo -e "\n${YELLOW}===== Running All PostNet Simulations =====${NC}"
        
        # Stack simulation
        echo -e "\n${YELLOW}--- PostNet Stack ---${NC}"
        if [ "$SIMULATOR" == "icarus" ]; then
            run_icarus "postnet_stack_tb" "postnet_stack_tb"
        else
            run_modelsim "postnet_stack_tb" "postnet_stack_tb"
        fi
        STACK_ERR=$?
        
        # Top simulation
        echo -e "\n${YELLOW}--- PostNet Top ---${NC}"
        if [ "$SIMULATOR" == "icarus" ]; then
            run_icarus "postnet_top_tb" "postnet_top_tb"
        else
            run_modelsim "postnet_top_tb" "postnet_top_tb"
        fi
        TOP_ERR=$?
        
        ERRORS=$((STACK_ERR + TOP_ERR))
        ;;
        
    *)
        echo -e "${RED}[ERROR] Unknown module: $MODULE${NC}"
        echo "Usage: $0 [stack|top|all] [icarus|modelsim]"
        exit 1
        ;;
esac

#------------------------------------------------------------------------------
# Summary
#------------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "Simulation Summary"
echo "=========================================="

if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}All simulations PASSED${NC}"
    echo "Log files: $SIM_DIR/*_sim.log"
    
    if [ "$SIMULATOR" == "icarus" ]; then
        echo "Waveforms: $SIM_DIR/*.vcd"
    fi
else
    echo -e "${RED}Some simulations FAILED (Errors: $ERRORS)${NC}"
    echo "Check log files in: $SIM_DIR/"
fi

echo "=========================================="

exit $ERRORS
