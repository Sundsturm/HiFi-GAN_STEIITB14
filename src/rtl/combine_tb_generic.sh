#!/bin/bash

# =============================================================================
# Script: combine_tb_generic.sh
# Purpose: Generic script to combine any testbench with its module
# Usage: ./combine_tb_generic.sh <module_name>
# Example: ./combine_tb_generic.sh generator_top
#          ./combine_tb_generic.sh mrf_block
#          ./combine_tb_generic.sh residual_block
# =============================================================================

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# =============================================================================
# Function: Print usage
# =============================================================================
usage() {
    echo -e "${BLUE}Usage: $0 <module_name> [output_file]${NC}"
    echo ""
    echo "Arguments:"
    echo "  module_name    Name of the module to combine (e.g., generator_top, mrf_block)"
    echo "  output_file    (Optional) Output filename (default: combined_<module_name>.v)"
    echo ""
    echo "Examples:"
    echo "  $0 generator_top"
    echo "  $0 mrf_block sim_files/mrf_combined.v"
    exit 1
}

# =============================================================================
# Parse arguments
# =============================================================================
if [ $# -eq 0 ]; then
    echo -e "${RED}ERROR: Missing module name argument${NC}"
    usage
fi

MODULE_NAME="$1"
OUTPUT_FILE="${2:-combined_${MODULE_NAME}.v}"

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Define file paths - try to find the module in the same directory as the script
MODULE_FILE="$SCRIPT_DIR/${MODULE_NAME}.v"

# If module is in parent directory, look there
if [ ! -f "$MODULE_FILE" ]; then
    MODULE_FILE="$SCRIPT_DIR/../${MODULE_NAME}.v"
fi

# Look for testbench in tb subdirectory
TB_FILE="$SCRIPT_DIR/tb/tb_${MODULE_NAME}.v"

# If tb is not found, try looking in parent's tb
if [ ! -f "$TB_FILE" ]; then
    TB_FILE="$SCRIPT_DIR/../tb/tb_${MODULE_NAME}.v"
fi

# =============================================================================
# Validate files exist
# =============================================================================
echo -e "${YELLOW}Searching for files...${NC}"

if [ ! -f "$MODULE_FILE" ]; then
    echo -e "${RED}ERROR: Module file not found${NC}"
    echo "Searched paths:"
    echo "  - $SCRIPT_DIR/${MODULE_NAME}.v"
    echo "  - $SCRIPT_DIR/../${MODULE_NAME}.v"
    exit 1
fi

if [ ! -f "$TB_FILE" ]; then
    echo -e "${RED}ERROR: Testbench file not found${NC}"
    echo "Searched paths:"
    echo "  - $SCRIPT_DIR/tb/tb_${MODULE_NAME}.v"
    echo "  - $SCRIPT_DIR/../tb/tb_${MODULE_NAME}.v"
    exit 1
fi

echo -e "${GREEN}✓ Module file: $MODULE_FILE${NC}"
echo -e "${GREEN}✓ Testbench file: $TB_FILE${NC}"

# =============================================================================
# Extract dependencies from testbench
# =============================================================================
echo -e "${YELLOW}Extracting dependencies...${NC}"

INCLUDE_FILES=()
while IFS= read -r line; do
    if [[ $line =~ \`include[[:space:]]+\"([^\"]+)\" ]]; then
        INCLUDE_FILES+=("${BASH_REMATCH[1]}")
        echo -e "  Found include: ${BASH_REMATCH[1]}"
    fi
done < "$TB_FILE"

# =============================================================================
# Create combined file
# =============================================================================
echo -e "${YELLOW}Creating combined simulation file: $OUTPUT_FILE${NC}"

# Create output file with header
cat > "$OUTPUT_FILE" << EOF
// =============================================================================
// AUTO-GENERATED: Combined Testbench + Module File
// Generated: $(date)
// Module: $MODULE_NAME
// This file combines the module with its testbench for simulation
// DO NOT EDIT MANUALLY - Regenerate using combine_tb_generic.sh
// =============================================================================

EOF

# Add include dependencies first
if [ ${#INCLUDE_FILES[@]} -gt 0 ]; then
    echo "// ====================== INCLUDE FILES ==========================" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
    
    for inc_file in "${INCLUDE_FILES[@]}"; do
        inc_path="$SCRIPT_DIR/$inc_file"
        
        # Try alternative paths
        if [ ! -f "$inc_path" ]; then
            inc_path="$SCRIPT_DIR/../$inc_file"
        fi
        if [ ! -f "$inc_path" ]; then
            inc_path="$SCRIPT_DIR/../../$inc_file"
        fi
        
        if [ -f "$inc_path" ]; then
            echo "// Including: $inc_file" >> "$OUTPUT_FILE"
            cat "$inc_path" >> "$OUTPUT_FILE"
            echo "" >> "$OUTPUT_FILE"
            echo -e "  ${GREEN}✓${NC} Included: $inc_file"
        else
            echo -e "  ${YELLOW}⚠${NC} Warning: Could not find include file: $inc_file"
        fi
    done
fi

# Add module file content
echo "" >> "$OUTPUT_FILE"
echo "// ====================== MODULE DEFINITION =========================" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
cat "$MODULE_FILE" >> "$OUTPUT_FILE"

# Add testbench file content
echo "" >> "$OUTPUT_FILE"
echo "// ====================== TESTBENCH DEFINITION ======================" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
cat "$TB_FILE" >> "$OUTPUT_FILE"

# Add footer
cat >> "$OUTPUT_FILE" << 'EOF'

// =============================================================================
// End of Combined Simulation File
// =============================================================================
EOF

echo -e "${GREEN}✓ Combined file created: $OUTPUT_FILE${NC}"

# =============================================================================
# Display file statistics
# =============================================================================
echo ""
echo -e "${BLUE}File Statistics:${NC}"
echo "  Module lines: $(wc -l < "$MODULE_FILE")"
echo "  Testbench lines: $(wc -l < "$TB_FILE")"
echo "  Combined lines: $(wc -l < "$OUTPUT_FILE")"
echo "  File size: $(du -h "$OUTPUT_FILE" | cut -f1)"

# =============================================================================
# Provide usage information
# =============================================================================
echo ""
echo -e "${BLUE}To run simulation:${NC}"
echo ""
echo "Using Icarus Verilog (iverilog):"
echo "  iverilog -o sim.vvp $OUTPUT_FILE"
echo "  vvp sim.vvp"
echo ""
echo "Using Vivado Simulator (xsim):"
echo "  xverilog -m64 $OUTPUT_FILE"
echo ""
echo "Using ModelSim:"
echo "  vlog $OUTPUT_FILE"
echo "  vsim tb_${MODULE_NAME}"
echo ""

exit 0
