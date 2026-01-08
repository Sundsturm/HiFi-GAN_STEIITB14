#!/bin/bash
#==============================================================================
# Run All PostNet Testbenches
# Purpose: Execute all PostNet component testbenches with Icarus Verilog
# Usage: ./run_all_postnet_tests.sh
#==============================================================================

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Directory setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RTL_DIR="$SCRIPT_DIR/.."
SHARED_DIR="$SCRIPT_DIR/../../shared"
SIM_DIR="$SCRIPT_DIR/sim_output"

# Create simulation output directory
mkdir -p "$SIM_DIR"

echo -e "${BLUE}=========================================="
echo "PostNet Test Suite"
echo "==========================================${NC}"
echo "Simulator: Icarus Verilog"
echo "Output Directory: $SIM_DIR"
echo -e "${BLUE}==========================================${NC}\n"

# Test counter
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

#------------------------------------------------------------------------------
# Function: Run a single testbench
#------------------------------------------------------------------------------
run_test() {
    local testbench=$1
    local module_name=$2
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    echo -e "\n${YELLOW}===== Test $TOTAL_TESTS: $module_name =====${NC}"
    echo -e "${BLUE}[Icarus] Compiling $testbench...${NC}"
    
    # Compile command - only include necessary files for each test
    case $module_name in
        "PostNet FSM")
            # FSM only needs the FSM module
            iverilog -g2005 \
                -o "$SIM_DIR/${testbench}.vvp" \
                -I"$RTL_DIR" \
                -I"$SHARED_DIR" \
                "$SCRIPT_DIR/${testbench}.v" \
                "$RTL_DIR/postnet_fsm.v"
            ;;
            
        "PostNet Stack")
            # Stack needs all shared modules
            iverilog -g2005 \
                -o "$SIM_DIR/${testbench}.vvp" \
                -I"$RTL_DIR" \
                -I"$SHARED_DIR" \
                "$SCRIPT_DIR/${testbench}.v" \
                "$RTL_DIR/postnet_stack.v" \
                "$SHARED_DIR/activation_unit/code/tanh_approx_q15.v" \
                "$SHARED_DIR/activation_unit/code/leaky_relu_q15.v" \
                "$SHARED_DIR/activation_unit/code/pwl_activation.v" \
                "$SHARED_DIR/mac_array/code/hifigan_mac_array.v" \
                "$SHARED_DIR/mac_array/code/qmult.v" \
                "$SHARED_DIR/memory/weight_mem.v" \
                "$SHARED_DIR/memory/bias_mem.v" \
                "$SHARED_DIR/quantizer/code/quantizer_32_16.v"
            ;;
            
        "PostNet Top")
            # Top needs everything
            iverilog -g2005 \
                -o "$SIM_DIR/${testbench}.vvp" \
                -I"$RTL_DIR" \
                -I"$SHARED_DIR" \
                "$SCRIPT_DIR/${testbench}.v" \
                "$RTL_DIR/postnet_top.v" \
                "$RTL_DIR/postnet_stack.v" \
                "$RTL_DIR/postnet_fsm.v" \
                "$SHARED_DIR/activation_unit/code/tanh_approx_q15.v" \
                "$SHARED_DIR/activation_unit/code/leaky_relu_q15.v" \
                "$SHARED_DIR/activation_unit/code/pwl_activation.v" \
                "$SHARED_DIR/mac_array/code/hifigan_mac_array.v" \
                "$SHARED_DIR/mac_array/code/qmult.v" \
                "$SHARED_DIR/memory/weight_mem.v" \
                "$SHARED_DIR/memory/bias_mem.v" \
                "$SHARED_DIR/quantizer/code/quantizer_32_16.v"
            ;;
    esac
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}[FAIL] Compilation failed for $testbench${NC}"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        return 1
    fi
    
    echo -e "${GREEN}[OK] Compilation successful${NC}"
    echo -e "${BLUE}[Icarus] Running simulation...${NC}"
    
    # Run simulation and capture output
    cd "$SIM_DIR"
    vvp "${testbench}.vvp" > "${testbench}_sim.log" 2>&1
    RESULT=$?
    cd "$SCRIPT_DIR"
    
    # Check for PASS/FAIL in log
    if grep -q "STATUS: PASS" "$SIM_DIR/${testbench}_sim.log"; then
        echo -e "${GREEN}[PASS] $module_name test passed${NC}"
        PASSED_TESTS=$((PASSED_TESTS + 1))
        
        # Check for VCD file
        if [ -f "$SIM_DIR/${testbench}.vcd" ]; then
            echo -e "${GREEN}Waveform: $SIM_DIR/${testbench}.vcd${NC}"
        fi
        return 0
    else
        echo -e "${RED}[FAIL] $module_name test failed${NC}"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        
        # Show last few lines of log
        echo -e "${YELLOW}Last 10 lines of output:${NC}"
        tail -n 10 "$SIM_DIR/${testbench}_sim.log"
        return 1
    fi
}

#------------------------------------------------------------------------------
# Run All Tests
#------------------------------------------------------------------------------
START_TIME=$(date +%s)

# Test 1: PostNet FSM
run_test "postnet_fsm_tb" "PostNet FSM"

# Test 2: PostNet Stack
run_test "postnet_stack_tb" "PostNet Stack"

# Test 3: PostNet Top
run_test "postnet_top_tb" "PostNet Top"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

#------------------------------------------------------------------------------
# Final Summary
#------------------------------------------------------------------------------
echo -e "\n${BLUE}=========================================="
echo "Test Suite Summary"
echo -e "==========================================${NC}"
echo "Total Tests:  $TOTAL_TESTS"
echo -e "${GREEN}Passed:       $PASSED_TESTS${NC}"

if [ $FAILED_TESTS -gt 0 ]; then
    echo -e "${RED}Failed:       $FAILED_TESTS${NC}"
else
    echo "Failed:       $FAILED_TESTS"
fi

echo "Elapsed Time: ${ELAPSED}s"
echo ""
echo "Log Directory: $SIM_DIR"

# List generated waveforms
VCD_COUNT=$(ls -1 "$SIM_DIR"/*.vcd 2>/dev/null | wc -l)
if [ $VCD_COUNT -gt 0 ]; then
    echo -e "\n${GREEN}Generated Waveforms:${NC}"
    ls -1 "$SIM_DIR"/*.vcd
    echo -e "\nView with: ${YELLOW}gtkwave <waveform.vcd>${NC}"
fi

echo -e "\n${BLUE}==========================================${NC}"

# Exit with appropriate code
if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}\n"
    exit 0
else
    echo -e "${RED}✗ Some tests failed!${NC}\n"
    exit 1
fi
