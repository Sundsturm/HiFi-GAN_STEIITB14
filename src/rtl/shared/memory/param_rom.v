//==============================================================================
// Module: param_rom
// Purpose: Static parameter ROM for HiFi-GAN layer configurations.
//          Stores layer-specific metadata (kernel size, dilation, channels, etc.)
//          Provides combinational read based on layer selection.
//
// Inputs:
//   - i_layer_sel : Layer selection index
//
// Outputs:
//   - o_kernel_size  : Kernel size for selected layer
//   - o_dilation     : Dilation factor for selected layer
//   - o_in_channels  : Number of input channels
//   - o_out_channels : Number of output channels
//
// Fixed-point Format:
//   - N/A (integer parameters only)
//
// Usage:
//   - Pre-configure parameters for each layer in the lookup table
//   - FSM reads parameters based on current layer being processed
//   - Supports up to MAX_LAYERS different layer configurations
//==============================================================================

module param_rom #(
    parameter MAX_LAYERS = 16,              // Maximum number of layers supported
    parameter PARAM_WIDTH = 8               // Bit width for each parameter
)(
    input wire [$clog2(MAX_LAYERS)-1:0]     i_layer_sel,
    
    // Layer Parameters Output
    output reg [PARAM_WIDTH-1:0]            o_kernel_size,
    output reg [PARAM_WIDTH-1:0]            o_dilation,
    output reg [PARAM_WIDTH-1:0]            o_in_channels,
    output reg [PARAM_WIDTH-1:0]            o_out_channels
);

    //==========================================================================
    // Parameter Lookup Table
    // Format: {kernel_size, dilation, in_channels, out_channels}
    //==========================================================================
    always @(*) begin
        case (i_layer_sel)
            // PostNet layers (example configuration)
            4'd0: begin
                o_kernel_size  = 8'd5;      // Kernel size = 5
                o_dilation     = 8'd1;      // Dilation = 1 (standard conv)
                o_in_channels  = 8'd1;      // Input: 1 channel (raw waveform)
                o_out_channels = 8'd32;     // Output: 32 channels
            end
            
            4'd1: begin
                o_kernel_size  = 8'd5;
                o_dilation     = 8'd1;
                o_in_channels  = 8'd32;
                o_out_channels = 8'd32;
            end
            
            4'd2: begin
                o_kernel_size  = 8'd5;
                o_dilation     = 8'd1;
                o_in_channels  = 8'd32;
                o_out_channels = 8'd32;
            end
            
            4'd3: begin
                o_kernel_size  = 8'd5;
                o_dilation     = 8'd1;
                o_in_channels  = 8'd32;
                o_out_channels = 8'd32;
            end
            
            4'd4: begin
                o_kernel_size  = 8'd5;
                o_dilation     = 8'd1;
                o_in_channels  = 8'd32;
                o_out_channels = 8'd1;      // Output: 1 channel (final waveform)
            end
            
            // Generator layers (example - customize as needed)
            4'd5: begin
                o_kernel_size  = 8'd7;
                o_dilation     = 8'd1;
                o_in_channels  = 8'd80;     // Mel spectrogram input
                o_out_channels = 8'd256;
            end
            
            4'd6: begin
                o_kernel_size  = 8'd3;
                o_dilation     = 8'd1;
                o_in_channels  = 8'd256;
                o_out_channels = 8'd128;
            end
            
            4'd7: begin
                o_kernel_size  = 8'd3;
                o_dilation     = 8'd2;      // Dilated conv
                o_in_channels  = 8'd128;
                o_out_channels = 8'd128;
            end
            
            4'd8: begin
                o_kernel_size  = 8'd3;
                o_dilation     = 8'd4;
                o_in_channels  = 8'd128;
                o_out_channels = 8'd64;
            end
            
            // Add more layers as needed...
            
            default: begin
                o_kernel_size  = 8'd3;
                o_dilation     = 8'd1;
                o_in_channels  = 8'd32;
                o_out_channels = 8'd32;
            end
        endcase
    end

endmodule
