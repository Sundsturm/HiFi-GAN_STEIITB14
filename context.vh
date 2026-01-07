// === CONTEXT FOR COPILOT - HiFi-GAN Generator + PostNet 
// You are helping write RTL Verilog (NOT SystemVerilog) for a HiFi-GAN
// Generator G + PostNet hardware implementation.
// Paper: https://arxiv.org/pdf/2006.05694

// GENERAL CONSTRAINTS:
// - Use pure Verilog-2001 syntax only.
// - Target tool: Xilinx Vivado.
// - No AXI, no DMA, no PS, no Zynq-specific IP.
// - Simulation-first design (Vivado RTL simulation).
// - Fixed-point arithmetic only (e.g., signed Q-format).
// - Weights and inputs are loaded from .mem files using $readmemh.
// - All memories are modeled as reg arrays or inferred BRAM.
// - No floating-point, no runtime quantizer module.

// ARCHITECTURE OVERVIEW:
// - The design implements ONLY inference.
// - Hardware-sharing is used: one Conv1D engine reused across layers.
// - Control is handled by hierarchical FSMs:
//   1) top_fsm: controls overall flow (Generator -> PostNet)
//   2) generator_fsm: controls Generator G layers
//   3) postnet_fsm: controls PostNet layers

// CORE SHARED MODULES:
// - mac_array: fixed-point multiply-accumulate with saturation.
// - line_buffer: sliding window buffer for 1-D convolution (supports dilation).
// - conv1d_engine: parameterizable Conv1D engine (kernel size, dilation).
// - activation_unit: supports LeakyReLU and tanh (tanh via LUT).
// - weight_mem / bias_mem: reg-array-based memory initialized via .mem files.

// GENERATOR G MODULES:
// - upsample_module: repeat-based upsampling followed by Conv1D.
// - residual_block: dilated Conv1D + activation + residual add.
// - mrf_block: Multi-Receptive-Field fusion using multiple residual blocks.
// - generator_top: structural wrapper for Generator G.
// - generator_fsm: FSM sequencing Generator layers.

// POSTNET MODULES:
// - postnet_stack: sequential Conv1D layers for waveform refinement.
// - postnet_top: wrapper and residual summation.
// - postnet_fsm: FSM sequencing PostNet layers.

// DESIGN STYLE RULES:
// - Separate datapath and control clearly.
// - Use start/done handshake signals between modules.
// - Avoid monolithic always blocks; prefer readable FSMs.
// - Use parameters for bit width, kernel size, and buffer depth.
// - Comment each module header with:
//   - Purpose
//   - Inputs/outputs
//   - Fixed-point assumptions

// MEMORY RULES:
// - .mem files are only used to initialize memory.
// - Memory must be explicitly declared in Verilog.
// - Assume weights are already quantized offline.

// FSM RULES:
// - Use explicit state encoding with localparam.
// - One always block for state register.
// - One always block for next-state logic.
// - One always block for outputs/control signals.

// OUTPUT EXPECTATION:
// - Generate clean, synthesizable Verilog.
// - Prefer clarity over extreme optimization.
// - Assume correctness and debuggability are more important than speed.

// When generating code:
// - Only generate the module requested.
// - Do not invent AXI or DMA interfaces.
// - Do not use SystemVerilog features.
// =======================================================================