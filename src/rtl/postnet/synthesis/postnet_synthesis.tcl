#==============================================================================
# Vivado Synthesis Script for PostNet Components
# Purpose: Synthesize PostNet modules for HiFi-GAN implementation
# Target: Xilinx FPGAs (configurable device)
# Design: HiFi-GAN PostNet waveform refinement module
#==============================================================================

#------------------------------------------------------------------------------
# Project Configuration
#------------------------------------------------------------------------------
set project_name "postnet_synthesis"
set project_dir "./postnet_synth_project"
set rtl_dir "../"
set shared_dir "../../shared"

# Target FPGA - Modify based on your target device
# Example options:
# - xc7z020clg484-1 (Zynq-7000)
# - xc7a35tcpg236-1 (Artix-7)
# - xc7k325tffg900-2 (Kintex-7)
set target_part "xc7z020clg484-1"

#------------------------------------------------------------------------------
# Create Project
#------------------------------------------------------------------------------
puts "=========================================="
puts "Creating Vivado Synthesis Project"
puts "=========================================="

# Remove existing project if present
if {[file exists $project_dir]} {
    puts "Removing existing project directory..."
    file delete -force $project_dir
}

create_project $project_name $project_dir -part $target_part -force
set_property target_language Verilog [current_project]

#------------------------------------------------------------------------------
# Add RTL Source Files
#------------------------------------------------------------------------------
puts "\n=========================================="
puts "Adding RTL Source Files"
puts "=========================================="

# PostNet modules
add_files -fileset sources_1 [list \
    "$rtl_dir/postnet_stack.v" \
    "$rtl_dir/postnet_top.v" \
    "$rtl_dir/postnet_fsm.v" \
]

# Shared modules - Activation Units
add_files -fileset sources_1 [list \
    "$shared_dir/activation_unit/code/tanh_approx_q15.v" \
    "$shared_dir/activation_unit/code/leaky_relu_q15.v" \
    "$shared_dir/activation_unit/code/pwl_activation.v" \
]

# Shared modules - MAC Array
add_files -fileset sources_1 [list \
    "$shared_dir/mac_array/code/hifigan_mac_array.v" \
    "$shared_dir/mac_array/code/qmult.v" \
]

# Shared modules - Memory
add_files -fileset sources_1 [list \
    "$shared_dir/memory/weight_mem.v" \
    "$shared_dir/memory/bias_mem.v" \
]

# Shared modules - Quantizer
add_files -fileset sources_1 [list \
    "$shared_dir/quantizer/code/quantizer_32_16.v" \
]

puts "RTL files added successfully"

#------------------------------------------------------------------------------
# Add Memory Initialization Files
#------------------------------------------------------------------------------
puts "\n=========================================="
puts "Adding Memory Initialization Files"
puts "=========================================="

if {[file exists "../../../weights.mem"]} {
    add_files -fileset sources_1 "../../../weights.mem"
    set_property file_type "Memory File" [get_files "../../../weights.mem"]
    puts "weights.mem added"
}

if {[file exists "../../../biases.mem"]} {
    add_files -fileset sources_1 "../../../biases.mem"
    set_property file_type "Memory File" [get_files "../../../biases.mem"]
    puts "biases.mem added"
}

#------------------------------------------------------------------------------
# Set Top Module
#------------------------------------------------------------------------------
puts "\n=========================================="
puts "Setting Top Module"
puts "=========================================="

set_property top postnet_top [current_fileset]
update_compile_order -fileset sources_1

#------------------------------------------------------------------------------
# Apply Synthesis Constraints
#------------------------------------------------------------------------------
puts "\n=========================================="
puts "Applying Synthesis Constraints"
puts "=========================================="

# Create timing constraints
create_clock -period 10.000 -name clk [get_ports clk]
set_input_delay -clock clk 2.000 [get_ports -filter {NAME !~ clk}]
set_output_delay -clock clk 2.000 [get_ports -filter {NAME !~ clk}]

# Set clock uncertainty (jitter + skew)
set_clock_uncertainty -setup 0.500 [get_clocks clk]
set_clock_uncertainty -hold 0.250 [get_clocks clk]

# Input/Output delays for better timing
set_max_delay 10.0 -from [all_inputs] -to [all_outputs]

# Area optimization strategy
set_property strategy Flow_AreaOptimized_high [get_runs synth_1]

# Enable retiming for better performance
set_property STEPS.SYNTH_DESIGN.ARGS.RETIMING true [get_runs synth_1]

# Additional synthesis options
set_property STEPS.SYNTH_DESIGN.ARGS.DIRECTIVE AreaOptimized_high [get_runs synth_1]
set_property STEPS.SYNTH_DESIGN.ARGS.FLATTEN_HIERARCHY rebuilt [get_runs synth_1]
set_property STEPS.SYNTH_DESIGN.ARGS.GATED_CLOCK_CONVERSION auto [get_runs synth_1]
set_property STEPS.SYNTH_DESIGN.ARGS.KEEP_EQUIVALENT_REGISTERS true [get_runs synth_1]

# Resource sharing for area reduction
set_property STEPS.SYNTH_DESIGN.ARGS.RESOURCE_SHARING auto [get_runs synth_1]

puts "Synthesis constraints applied"

#------------------------------------------------------------------------------
# Run Synthesis
#------------------------------------------------------------------------------
puts "\n=========================================="
puts "Starting Synthesis"
puts "=========================================="

# Launch synthesis
launch_runs synth_1 -jobs 4
wait_on_run synth_1

# Check synthesis status
if {[get_property PROGRESS [get_runs synth_1]] != "100%"} {
    puts "\n[ERROR] Synthesis failed!"
    exit 1
}

puts "\n=========================================="
puts "Synthesis Completed Successfully"
puts "=========================================="

#------------------------------------------------------------------------------
# Open Synthesized Design
#------------------------------------------------------------------------------
open_run synth_1 -name synth_1

#------------------------------------------------------------------------------
# Generate Reports
#------------------------------------------------------------------------------
puts "\n=========================================="
puts "Generating Synthesis Reports"
puts "=========================================="

# Create reports directory
set report_dir "$project_dir/reports"
file mkdir $report_dir

# Utilization Report
report_utilization -file $report_dir/postnet_utilization.rpt
report_utilization -hierarchical -file $report_dir/postnet_utilization_hierarchical.rpt

# Timing Report
report_timing_summary -file $report_dir/postnet_timing_summary.rpt
report_timing -sort_by slack -max_paths 10 -file $report_dir/postnet_timing_detail.rpt

# Power Report
report_power -file $report_dir/postnet_power.rpt

# Clock Report
report_clocks -file $report_dir/postnet_clocks.rpt

# DRC Report
report_drc -file $report_dir/postnet_drc.rpt

# Methodology Report
report_methodology -file $report_dir/postnet_methodology.rpt

puts "Reports generated in: $report_dir"

#------------------------------------------------------------------------------
# Print Resource Utilization Summary
#------------------------------------------------------------------------------
puts "\n=========================================="
puts "Resource Utilization Summary"
puts "=========================================="

set util [report_utilization -return_string]
puts $util

#------------------------------------------------------------------------------
# Extract Key Metrics
#------------------------------------------------------------------------------
puts "\n=========================================="
puts "Key Synthesis Metrics"
puts "=========================================="

# LUT utilization
set luts [get_property LUT [get_cells -hierarchical -filter {PRIMITIVE_TYPE =~ LUT*}]]
puts "Total LUTs: [llength $luts]"

# Register utilization
set regs [get_property REG [get_cells -hierarchical -filter {PRIMITIVE_TYPE =~ REGISTER.*}]]
puts "Total Registers: [llength $regs]"

# DSP blocks
set dsps [get_cells -hierarchical -filter {PRIMITIVE_TYPE =~ DSP*}]
puts "Total DSP48s: [llength $dsps]"

# BRAM blocks
set brams [get_cells -hierarchical -filter {PRIMITIVE_TYPE =~ BMEM*}]
puts "Total BRAMs: [llength $brams]"

# Timing
set wns [get_property SLACK [get_timing_paths]]
puts "Worst Negative Slack (WNS): $wns ns"

#------------------------------------------------------------------------------
# Export Netlist
#------------------------------------------------------------------------------
puts "\n=========================================="
puts "Exporting Netlist"
puts "=========================================="

write_verilog -force -mode funcsim $project_dir/postnet_top_funcsim.v
write_verilog -force -mode timesim $project_dir/postnet_top_timesim.v
write_vhdl -force -mode funcsim $project_dir/postnet_top_funcsim.vhd

puts "Netlist exported"

#------------------------------------------------------------------------------
# Export Constraints
#------------------------------------------------------------------------------
write_xdc -force $project_dir/postnet_constraints.xdc

#------------------------------------------------------------------------------
# Create Checkpoint
#------------------------------------------------------------------------------
write_checkpoint -force $project_dir/postnet_synth.dcp

#------------------------------------------------------------------------------
# Implementation (Optional - Uncomment to run)
#------------------------------------------------------------------------------
# puts "\n=========================================="
# puts "Starting Implementation"
# puts "=========================================="
# 
# launch_runs impl_1 -jobs 4
# wait_on_run impl_1
# 
# if {[get_property PROGRESS [get_runs impl_1]] != "100%"} {
#     puts "\n[ERROR] Implementation failed!"
#     exit 1
# }
# 
# open_run impl_1
# 
# # Implementation reports
# report_utilization -file $report_dir/postnet_utilization_impl.rpt
# report_timing_summary -file $report_dir/postnet_timing_impl.rpt
# report_power -file $report_dir/postnet_power_impl.rpt
# 
# # Generate bitstream (optional)
# # write_bitstream -force $project_dir/postnet_top.bit

#------------------------------------------------------------------------------
# Summary
#------------------------------------------------------------------------------
puts "\n=========================================="
puts "Synthesis Flow Complete"
puts "=========================================="
puts "Project: $project_name"
puts "Device: $target_part"
puts "Output: $project_dir"
puts "Reports: $report_dir"
puts "=========================================="

# Keep Vivado open for inspection (comment out to close)
# close_project
