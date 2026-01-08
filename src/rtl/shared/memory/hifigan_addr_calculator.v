//==============================================================================
// Module: hifigan_addr_calculator
// Purpose: Calculate memory addresses for HiFi-GAN layer parameters.
//          Converts layer indices and channel positions to linear memory addresses.
//          Uses address map constants from hifigan_addr_map.vh
//
// Inputs:
//   - i_layer_type    : Layer type selection (UPS, RESBLOCK, CONV_PRE, CONV_POST)
//   - i_layer_idx     : Layer index within type (e.g., ups.0, ups.1, etc.)
//   - i_sub_layer_idx : Sub-layer index (for resblocks with multiple convs)
//   - i_param_type    : Parameter type (WEIGHT_G, WEIGHT_V, BIAS)
//   - i_ch_out        : Output channel index
//   - i_ch_in         : Input channel index (for weights only)
//   - i_kernel_pos    : Kernel position (for weights only)
//   - i_calc_en       : Enable address calculation
//
// Outputs:
//   - o_addr          : Calculated memory address
//   - o_valid         : Address calculation valid
//
// Usage Example:
//   To access ups.1.weight_v[ch_out=32][ch_in=16][k=8]:
//   - i_layer_type = LAYER_UPS
//   - i_layer_idx = 1
//   - i_param_type = PARAM_WEIGHT_V
//   - i_ch_out = 32, i_ch_in = 16, i_kernel_pos = 8
//==============================================================================

`include "hifigan_addr_map.vh"

module hifigan_addr_calculator #(
    parameter ADDR_WIDTH = 21  // log2(1,464,322) = 21 bits
)(
    input wire                          clk,
    input wire                          rst_n,
    
    // Layer Selection
    input wire [3:0]                    i_layer_type,     // Layer type
    input wire [7:0]                    i_layer_idx,      // Layer index
    input wire [3:0]                    i_sub_layer_idx,  // Sub-layer (for resblocks)
    input wire [2:0]                    i_param_type,     // Parameter type
    
    // Indexing
    input wire [9:0]                    i_ch_out,         // Output channel
    input wire [9:0]                    i_ch_in,          // Input channel
    input wire [7:0]                    i_kernel_pos,     // Kernel position
    
    // Control
    input wire                          i_calc_en,        // Calculate enable
    
    // Output
    output reg [ADDR_WIDTH-1:0]         o_addr,
    output reg                          o_valid
);

    //==========================================================================
    // Layer Type Encodings
    //==========================================================================
    localparam [3:0] LAYER_CONV_PRE  = 4'd0;
    localparam [3:0] LAYER_UPS       = 4'd1;
    localparam [3:0] LAYER_RESBLOCK  = 4'd2;
    localparam [3:0] LAYER_CONV_POST = 4'd3;
    
    //==========================================================================
    // Parameter Type Encodings
    //==========================================================================
    localparam [2:0] PARAM_BIAS     = 3'd0;
    localparam [2:0] PARAM_WEIGHT_G = 3'd1;
    localparam [2:0] PARAM_WEIGHT_V = 3'd2;
    
    //==========================================================================
    // Address Calculation Logic
    //==========================================================================
    reg [ADDR_WIDTH-1:0] base_addr;
    reg [ADDR_WIDTH-1:0] offset;
    reg [ADDR_WIDTH-1:0] calc_addr;
    
    always @(*) begin
        base_addr = 0;
        offset = 0;
        calc_addr = 0;
        
        case (i_layer_type)
            //------------------------------------------------------------------
            // CONV_PRE Layer
            //------------------------------------------------------------------
            LAYER_CONV_PRE: begin
                case (i_param_type)
                    PARAM_BIAS: begin
                        base_addr = CONV_PRE_BIAS_START;
                        offset = i_ch_out;
                    end
                    
                    PARAM_WEIGHT_G: begin
                        base_addr = CONV_PRE_WEIGHT_G_START;
                        offset = i_ch_out;
                    end
                    
                    PARAM_WEIGHT_V: begin
                        base_addr = CONV_PRE_WEIGHT_V_START;
                        // Shape: [256, 80, 7]
                        // offset = ch_out * 80 * 7 + ch_in * 7 + k_pos
                        offset = (i_ch_out * 560) + (i_ch_in * 7) + i_kernel_pos;
                    end
                endcase
            end
            
            //------------------------------------------------------------------
            // Upsampler Layers (ups.0, ups.1, ups.2)
            //------------------------------------------------------------------
            LAYER_UPS: begin
                case (i_layer_idx)
                    8'd0: begin // ups.0
                        case (i_param_type)
                            PARAM_BIAS: begin
                                base_addr = UPS_0_BIAS_START;
                                offset = i_ch_out;
                            end
                            PARAM_WEIGHT_G: begin
                                base_addr = UPS_0_WEIGHT_G_START;
                                offset = i_ch_out;
                            end
                            PARAM_WEIGHT_V: begin
                                base_addr = UPS_0_WEIGHT_V_START;
                                // Shape: [256, 128, 16]
                                offset = (i_ch_out * 2048) + (i_ch_in * 16) + i_kernel_pos;
                            end
                        endcase
                    end
                    
                    8'd1: begin // ups.1
                        case (i_param_type)
                            PARAM_BIAS: begin
                                base_addr = UPS_1_BIAS_START;
                                offset = i_ch_out;
                            end
                            PARAM_WEIGHT_G: begin
                                base_addr = UPS_1_WEIGHT_G_START;
                                offset = i_ch_out;
                            end
                            PARAM_WEIGHT_V: begin
                                base_addr = UPS_1_WEIGHT_V_START;
                                // Shape: [128, 64, 16]
                                offset = (i_ch_out * 1024) + (i_ch_in * 16) + i_kernel_pos;
                            end
                        endcase
                    end
                    
                    8'd2: begin // ups.2
                        case (i_param_type)
                            PARAM_BIAS: begin
                                base_addr = UPS_2_BIAS_START;
                                offset = i_ch_out;
                            end
                            PARAM_WEIGHT_G: begin
                                base_addr = UPS_2_WEIGHT_G_START;
                                offset = i_ch_out;
                            end
                            PARAM_WEIGHT_V: begin
                                base_addr = UPS_2_WEIGHT_V_START;
                                // Shape: [64, 32, 8]
                                offset = (i_ch_out * 256) + (i_ch_in * 8) + i_kernel_pos;
                            end
                        endcase
                    end
                endcase
            end
            
            //------------------------------------------------------------------
            // Residual Block Layers (resblocks.0-8, each with 2 convs)
            //------------------------------------------------------------------
            LAYER_RESBLOCK: begin
                // For simplicity, implement resblock.0 and resblock.3 as examples
                // Add others as needed
                case (i_layer_idx)
                    8'd0: begin // resblocks.0 (kernel=3, channels=128)
                        case (i_sub_layer_idx)
                            4'd0: begin // convs.0
                                case (i_param_type)
                                    PARAM_BIAS: begin
                                        base_addr = RESBLOCKS_0_CONVS_0_BIAS_START;
                                        offset = i_ch_out;
                                    end
                                    PARAM_WEIGHT_G: begin
                                        base_addr = RESBLOCKS_0_CONVS_0_WEIGHT_G_START;
                                        offset = i_ch_out;
                                    end
                                    PARAM_WEIGHT_V: begin
                                        base_addr = RESBLOCKS_0_CONVS_0_WEIGHT_V_START;
                                        // Shape: [128, 128, 3]
                                        offset = (i_ch_out * 384) + (i_ch_in * 3) + i_kernel_pos;
                                    end
                                endcase
                            end
                            
                            4'd1: begin // convs.1
                                case (i_param_type)
                                    PARAM_BIAS: begin
                                        base_addr = RESBLOCKS_0_CONVS_1_BIAS_START;
                                        offset = i_ch_out;
                                    end
                                    PARAM_WEIGHT_G: begin
                                        base_addr = RESBLOCKS_0_CONVS_1_WEIGHT_G_START;
                                        offset = i_ch_out;
                                    end
                                    PARAM_WEIGHT_V: begin
                                        base_addr = RESBLOCKS_0_CONVS_1_WEIGHT_V_START;
                                        // Shape: [128, 128, 3]
                                        offset = (i_ch_out * 384) + (i_ch_in * 3) + i_kernel_pos;
                                    end
                                endcase
                            end
                        endcase
                    end
                    
                    // Add more resblocks as needed...
                    // resblocks.1-2: kernel=5,7 channels=128
                    // resblocks.3-5: kernel=3,5,7 channels=64
                    // resblocks.6-8: kernel=3,5,7 channels=32
                endcase
            end
            
            //------------------------------------------------------------------
            // CONV_POST Layer
            //------------------------------------------------------------------
            LAYER_CONV_POST: begin
                case (i_param_type)
                    PARAM_BIAS: begin
                        base_addr = CONV_POST_BIAS_START;
                        offset = i_ch_out;
                    end
                    
                    PARAM_WEIGHT_G: begin
                        base_addr = CONV_POST_WEIGHT_G_START;
                        offset = i_ch_out;
                    end
                    
                    PARAM_WEIGHT_V: begin
                        base_addr = CONV_POST_WEIGHT_V_START;
                        // Shape: [1, 32, 7]
                        offset = (i_ch_out * 224) + (i_ch_in * 7) + i_kernel_pos;
                    end
                endcase
            end
        endcase
        
        calc_addr = base_addr + offset;
    end
    
    //==========================================================================
    // Register Output
    //==========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            o_addr  <= 0;
            o_valid <= 1'b0;
        end
        else begin
            o_valid <= i_calc_en;
            
            if (i_calc_en) begin
                o_addr <= calc_addr;
            end
        end
    end

endmodule
