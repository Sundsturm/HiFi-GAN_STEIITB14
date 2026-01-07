//==============================================================================
// Module: bias_mem
// Purpose: Shared bias memory for HiFi-GAN layers (Generator & PostNet).
//          Single-port synchronous read BRAM initialized from .mem file.
//          Stores 16-bit biases in simple linear addressing.
//
// Inputs:
//   - clk      : System clock
//   - rst_n    : Active-low async reset
//   - i_addr   : Read address
//   - i_rd_en  : Read enable (synchronous read)
//
// Outputs:
//   - o_data   : Bias data output (Q4.12 format, registered)
//   - o_valid  : Valid signal (registered, follows i_rd_en by 1 cycle)
//
// Fixed-point Format:
//   - Biases: Q4.12 (16-bit signed, 4 integer bits, 12 fractional bits)
//
// Memory Organization:
//   - Linear addressing: biases stored sequentially
//   - Address calculation: bias[layer][ch_out] = layer_offset + ch_out
//==============================================================================

module bias_mem #(
    parameter DATA_WIDTH = 16,              // Bias bit width (Q4.12)
    parameter DEPTH      = 512,             // Memory depth (words)
    parameter MEM_FILE   = "biases.mem"     // Initialization file
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
                o_data <= mem[i_addr];
            end
        end
    end

endmodule
