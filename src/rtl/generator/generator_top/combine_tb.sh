#!/bin/bash

# =============================================================================
# Script: combine_tb.sh
# Purpose: Combine testbench with original Verilog module
# =============================================================================

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Define file paths
MODULE_FILE="$SCRIPT_DIR/generator_top.v"
TB_FILE="$SCRIPT_DIR/tb/tb_generator_top.v"
OUTPUT_FILE="$SCRIPT_DIR/combined_sim.v"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# =============================================================================
# Check if required files exist
# =============================================================================
echo -e "${YELLOW}Checking for required files...${NC}"

if [ ! -f "$MODULE_FILE" ]; then
    echo -e "${RED}ERROR: Module file not found: $MODULE_FILE${NC}"
    exit 1
fi

if [ ! -f "$TB_FILE" ]; then
    echo -e "${RED}ERROR: Testbench file not found: $TB_FILE${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Found module file: $MODULE_FILE${NC}"
echo -e "${GREEN}✓ Found testbench file: $TB_FILE${NC}"

# =============================================================================
# Create combined file
# =============================================================================
echo -e "${YELLOW}Creating combined simulation file...${NC}"

# Create output file with header
cat > "$OUTPUT_FILE" << 'EOF'
// =============================================================================
// AUTO-GENERATED: Combined Testbench + Module File
// This file combines the module with its testbench for simulation
// DO NOT EDIT MANUALLY - Regenerate using combine_tb.sh
// =============================================================================

EOF

# Add module file content (skip the `include directives from testbench)
echo "" >> "$OUTPUT_FILE"
echo "// ====================== MODULE DEFINITION =========================" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
cat "$MODULE_FILE" >> "$OUTPUT_FILE"

# Add testbench file content (the `includes will be ignored during compilation
# since the modules are already defined)
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
echo -e "${YELLOW}File Statistics:${NC}"
echo "  Module lines: $(wc -l < "$MODULE_FILE")"
echo "  Testbench lines: $(wc -l < "$TB_FILE")"
echo "  Combined lines: $(wc -l < "$OUTPUT_FILE")"

# =============================================================================
# Provide usage information
# =============================================================================
echo ""
echo -e "${YELLOW}To run simulation:${NC}"
echo "  iverilog -o sim.vvp $OUTPUT_FILE"
echo "  vvp sim.vvp"
echo ""
echo "Or with VCS:"
echo "  vcs $OUTPUT_FILE -o sim"
echo "  ./sim"
echo ""

exit 0
