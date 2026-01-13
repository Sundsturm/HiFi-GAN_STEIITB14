// =======================================================================
// Module: conv1d_engine (DEPRECATED - Use conv1d_simple instead)
// Purpose: Legacy 1-D convolution engine - kept for compatibility
// 
// Description:
//   This is the original complex conv1d_engine. It is DEPRECATED.
//   New designs should use conv1d_simple.v for simpler streaming interface.
//   This file is kept only for backward compatibility with existing code.
//
// DEPRECATION NOTICE:
//   - PostNet does NOT use this module (uses direct MAC array)
//   - Generator residual_block should use conv1d_simple.v instead
//   - This module has interface complexity that is unnecessary
//
// Simplified Alternative:
//   Use conv1d_simple.v for streaming convolution in residual blocks.
//   It has cleaner interface without multi-channel complexity.
//
// Notes:
//   - This module is overly complex for simple use cases
//   - Interface mismatch with residual_block instantiation
//   - Consider refactoring to use conv1d_simple.v
// =======================================================================

module conv1d_engine #(
    parameter DATA_WIDTH      = 16,     // Bit width for data (Q4.12)
    parameter KERNEL_SIZE     = 3,      // Convolution kernel size
    parameter DILATION        = 1,      // Dilation factor (simplified)
    parameter MAX_DILATION    = 9,      // Maximum dilation factor
    parameter BUFFER_DEPTH    = 64,     // Line buffer depth
    parameter ACTIVATION      = "NONE"  // "LEAKY_RELU", "TANH", "NONE"
)(
    input  wire                           clk,
    input  wire                           rst_n,
    
    // Simplified control signals
    input  wire                           start,        // Start/reset
    output reg                            done,         // Always high after init
    
    // Streaming data interface (simplified)
    input  wire signed [DATA_WIDTH-1:0]   d_in,         // Input sample (Q4.12)
    input  wire                           d_in_valid,   // Input valid
    output reg  signed [DATA_WIDTH-1:0]   d_out,        // Output sample (Q4.12)
    output reg                            d_out_valid   // Output valid
);

    // ===================================================================
    // SIMPLIFIED IMPLEMENTATION
    // This is now a thin wrapper - actual work delegated to modules
    // ===================================================================
    
    // Internal Signals
    wire signed [DATA_WIDTH*KERNEL_SIZE-1:0] window_out;
    reg  lb_enable;
    reg  lb_clear;
    wire lb_valid;
    
    wire signed [DATA_WIDTH*KERNEL_SIZE-1:0] mac_activations;
    wire signed [DATA_WIDTH*KERNEL_SIZE-1:0] mac_weights;
    wire signed [31:0] mac_acc_raw;
    wire mac_valid;
    reg  mac_calc_en;
    reg  mac_clear_acc;
    
    wire signed [DATA_WIDTH-1:0] quant_data;
    wire quant_valid;
    
    wire signed [DATA_WIDTH-1:0] act_data;
    
    // Simple state machine
    reg [1:0] state;
    localparam IDLE = 2'd0;
    localparam RUN  = 2'd1;
    localparam DONE_ST = 2'd2;
    
    integer i;
    genvar g;
    
    // ===================================================================
    // Line Buffer Instantiation
    // ===================================================================
    line_buffer #(
        .DATA_WIDTH(DATA_WIDTH),
        .KERNEL_SIZE(KERNEL_SIZE),
        .MAX_DILATION(MAX_DILATION),
        .BUFFER_DEPTH(BUFFER_DEPTH)
    ) u_line_buffer (
        .clk(clk),
        .rst_n(rst_n),
        .enable(lb_enable),
        .data_in(d_in),
        .dilation(DILATION[3:0]),
        .clear(lb_clear),
        .window_out(window_out),
        .valid(lb_valid)
    );
    
    // ===================================================================
    // MAC Array Instantiation (with dummy weights for now)
    // ===================================================================
    assign mac_activations = window_out;
    assign mac_weights = {KERNEL_SIZE{16'h1000}};  // Dummy weight = 0.25 in Q2.14
    
    hifigan_mac_array #(
        .KERNEL_SIZE(KERNEL_SIZE),
        .DATA_WIDTH(DATA_WIDTH)
    ) u_mac_array (
        .clk(clk),
        .rst_n(rst_n),
        .i_calc_en(mac_calc_en),
        .i_clear_acc(mac_clear_acc),
        .i_activations(mac_activations),
        .i_weights(mac_weights),
        .o_acc_raw(mac_acc_raw),
        .o_valid(mac_valid)
    );
    
    // ===================================================================
    // Quantizer Instantiation
    // ===================================================================
    quantizer_32_16 u_quantizer (
        .clk(clk),
        .rst_n(rst_n),
        .i_valid(mac_valid),
        .i_acc_raw(mac_acc_raw),
        .o_data(quant_data),
        .o_valid_out(quant_valid)
    );
    
    // ===================================================================
    // Optional Activation Unit
    // ===================================================================
    generate
        if (ACTIVATION == "LEAKY_RELU") begin : gen_leaky_relu
            leaky_relu_q15 u_activation (
                .x(quant_data),
                .y(act_data)
            );
        end else if (ACTIVATION == "TANH") begin : gen_tanh
            tanh_approx_q15 u_activation (
                .x(quant_data),
                .y(act_data)
            );
        end else begin : gen_bypass
            assign act_data = quant_data;
        end
    endgenerate
    
    // ===================================================================
    // Simplified FSM Control
    // ===================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            done <= 1'b0;
            d_out <= 0;
            d_out_valid <= 1'b0;
            lb_enable <= 1'b0;
            lb_clear <= 1'b0;
            mac_calc_en <= 1'b0;
            mac_clear_acc <= 1'b0;
        end else begin
            // Default
            lb_enable <= 1'b0;
            mac_calc_en <= 1'b0;
            d_out_valid <= 1'b0;
            
            case (state)
                IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        state <= RUN;
                        lb_clear <= 1'b1;
                    end
                end
                
                RUN: begin
                    lb_clear <= 1'b0;
                    done <= 1'b1;  // Always ready after start
                    
                    // Simple streaming pipeline
                    if (d_in_valid) begin
                        lb_enable <= 1'b1;
                    end
                    
                    if (lb_valid) begin
                        mac_calc_en <= 1'b1;
                        mac_clear_acc <= 1'b1;
                    end
                    
                    if (quant_valid) begin
                        d_out <= act_data;
                        d_out_valid <= 1'b1;
                    end
                end
                
                default: state <= IDLE;
            endcase
        end
    end

endmodule
