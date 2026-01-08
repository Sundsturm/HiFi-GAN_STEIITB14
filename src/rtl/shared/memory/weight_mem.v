//==============================================================================
// Module: weight_mem
// Purpose: Shared weight memory for HiFi-GAN layers (Generator & PostNet).
//          Single-port synchronous read BRAM initialized from .mem file.
//          Stores 16-bit weights in simple linear addressing (one weight per addr).
//
// Inputs:
//   - clk      : System clock
//   - rst_n    : Active-low async reset
//   - i_addr   : Read address
//   - i_rd_en  : Read enable (synchronous read)
//
// Outputs:
//   - o_data   : Weight data output (Q2.14 format, registered)
//   - o_valid  : Valid signal (registered, follows i_rd_en by 1 cycle)
//
// Fixed-point Format:
//   - Weights: Q2.14 (16-bit signed, 2 integer bits, 14 fractional bits)
//
// Memory Organization:
//   - Linear addressing: weights stored sequentially
//   - Address calculation done by caller (FSM) or address calculator
//   - Generated from CSV using generate_memory_files.py
//   - Default depth supports full HiFi-GAN model (1,462,497 weights)
//   - Use hifigan_addr_map.vh for layer address constants
//==============================================================================

module weight_mem #(
    parameter DATA_WIDTH = 16,                  // Weight bit width (Q2.14)
    parameter DEPTH      = 1462497,             // Memory depth (words) - full HiFiGAN
    parameter MEM_FILE   = "weights.mem",       // Initialization file
    parameter ADDR_CHECK = 0                    // Enable address range checking (debug)
)(
    input wire                          clk,
    input wire                          rst_n,
    
    // Read Interface
    input wire [$clog2(DEPTH)-1:0]      i_addr,
    input wire                          i_rd_en,
    
    // Data Output
    output reg signed [DATA_WIDTH-1:0]  o_data,
    output reg                          o_valid
);

    //==========================================================================
    // Memory Declaration (inferred BRAM)
    //==========================================================================
    reg signed [DATA_WIDTH-1:0] mem [0:DEPTH-1];
    
    //==========================================================================
    // Memory Initialization
    //==========================================================================
    initial begin
        $readmemh(MEM_FILE, mem);
    end

    //==========================================================================
    // Synchronous Read (single-cycle latency)
    //==========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            o_data  <= {DATA_WIDTH{1'b0}};
            o_valid <= 1'b0;
        end
        else begin
            o_valid <= i_rd_en;
            
            if (i_rd_en) begin
                // Address range checking (optional, for debug)
                if (ADDR_CHECK && i_addr >= DEPTH) begin
                    $display("[ERROR] weight_mem: Address out of range: %d >= %d", i_addr, DEPTH);
                    o_data <= {DATA_WIDTH{1'b0}};
                end
                else begin
                    o_data <= mem[i_addr];
                end
            end
        end
    end

endmodule
