#!/bin/bash

# =============================================================================
# Script: run_all_combined_tests.sh
# Purpose: Automatically find and combine all tb files with their modules
#          and run simulations
# =============================================================================

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Simulator to use (iverilog, vcs, xverilog, etc.)
SIMULATOR="${1:-iverilog}"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Combined Testbench Runner${NC}"
echo -e "${BLUE}========================================${NC}"
echo "Simulator: $SIMULATOR"
echo "Search directory: $SCRIPT_DIR"
echo ""

# =============================================================================
# Find all testbench files
# =============================================================================
echo -e "${YELLOW}Finding testbench files...${NC}"

# Array to store found testbenches
declare -a TESTBENCHES

# Search for all tb_*.v files
while IFS= read -r -d '' tb_file; do
    TESTBENCHES+=("$tb_file")
    filename=$(basename "$tb_file")
    echo -e "  ${GREEN}✓${NC} Found: $filename"
done < <(find "$SCRIPT_DIR" -name "tb_*.v" -print0 2>/dev/null)

if [ ${#TESTBENCHES[@]} -eq 0 ]; then
    echo -e "${RED}ERROR: No testbench files found${NC}"
    exit 1
fi

echo ""
echo -e "${YELLOW}Total testbenches found: ${#TESTBENCHES[@]}${NC}"
echo ""

# =============================================================================
# Process each testbench
# =============================================================================
PASSED=0
FAILED=0
SKIPPED=0

for tb_path in "${TESTBENCHES[@]}"; do
    tb_dir=$(dirname "$tb_path")
    tb_filename=$(basename "$tb_path")
    module_name="${tb_filename#tb_}"  # Remove "tb_" prefix
    module_name="${module_name%.v}"   # Remove ".v" suffix
    
    echo -e "${BLUE}─────────────────────────────────────${NC}"
    echo -e "Processing: ${YELLOW}$module_name${NC}"
    echo -e "${BLUE}─────────────────────────────────────${NC}"
    
    # Look for module file in same directory or parent
    module_file="$tb_dir/../${module_name}.v"
    if [ ! -f "$module_file" ]; then
        module_file="$tb_dir/${module_name}.v"
    fi
    
    if [ ! -f "$module_file" ]; then
        echo -e "${YELLOW}⚠ SKIPPED: Module file not found (${module_name}.v)${NC}"
        ((SKIPPED++))
        continue
    fi
    
    # Create output file
    output_file="$tb_dir/combined_${module_name}.v"
    
    echo "  Module: $(basename "$module_file")"
    echo "  Testbench: $tb_filename"
    echo "  Output: $(basename "$output_file")"
    echo ""
    
    # Create combined file
    echo -e "${YELLOW}  Creating combined file...${NC}"
    
    {
        cat << EOF
// =============================================================================
// AUTO-GENERATED: Combined Testbench + Module
// Generated: $(date)
// Module: $module_name
// =============================================================================

EOF
        
        # Extract and include dependencies
        deps=$(grep -h '`include' "$tb_path" 2>/dev/null | sort -u)
        if [ ! -z "$deps" ]; then
            echo "// ====================== DEPENDENCIES =========================="
            echo ""
            echo "$deps" | while read -r inc_line; do
                # Extract filename from include statement
                if [[ $inc_line =~ \`include[[:space:]]+\"([^\"]+)\" ]]; then
                    inc_file="${BASH_REMATCH[1]}"
                    inc_path="$tb_dir/../$inc_file"
                    
                    if [ -f "$inc_path" ]; then
                        cat "$inc_path"
                        echo ""
                    else
                        echo "// Warning: Could not find $inc_file"
                        echo ""
                    fi
                fi
            done
        fi
        
        echo "// ====================== MODULE DEFINITION ========================="
        echo ""
        cat "$module_file"
        echo ""
        echo "// ====================== TESTBENCH DEFINITION ======================"
        echo ""
        cat "$tb_path"
        echo ""
        echo "// ====================== END OF FILE =============================="
        
    } > "$output_file"
    
    echo -e "  ${GREEN}✓ Combined file created${NC}"
    
    # Compile
    echo -e "${YELLOW}  Compiling...${NC}"
    
    case "$SIMULATOR" in
        iverilog)
            if iverilog -o "${output_file%.v}.vvp" "$output_file" 2>&1; then
                echo -e "  ${GREEN}✓ Compilation successful${NC}"
                
                # Run simulation
                echo -e "${YELLOW}  Running simulation...${NC}"
                if vvp "${output_file%.v}.vvp" 2>&1; then
                    echo -e "  ${GREEN}✓ Simulation completed${NC}"
                    ((PASSED++))
                else
                    echo -e "  ${RED}✗ Simulation failed${NC}"
                    ((FAILED++))
                fi
            else
                echo -e "  ${RED}✗ Compilation failed${NC}"
                ((FAILED++))
            fi
            ;;
        *)
            echo -e "  ${YELLOW}⚠ Simulator '$SIMULATOR' not yet supported${NC}"
            echo "     Supported: iverilog"
            ((SKIPPED++))
            ;;
    esac
    
    echo ""
done

# =============================================================================
# Summary
# =============================================================================
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Test Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "  ${GREEN}Passed:  $PASSED${NC}"
echo -e "  ${RED}Failed:  $FAILED${NC}"
echo -e "  ${YELLOW}Skipped: $SKIPPED${NC}"
echo -e "  ${BLUE}Total:   $((PASSED + FAILED + SKIPPED))${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed!${NC}"
    exit 1
fi
