// =======================================================================
// Module: line_buffer
// Purpose: Circular buffer for 1-D dilated convolution
// 
// Description:
//   Sliding window buffer with configurable dilation. Stores incoming
//   samples and outputs KERNEL_SIZE samples spaced by dilation factor.
//
// Key Features:
//   - Supports dilation rates 1 to MAX_DILATION
//   - Circular addressing with automatic wrap-around
//   - Valid flag indicates sufficient samples available
//   - Fixed-point Q15 format (16-bit signed)
//
// Notes:
//   - Pure Verilog-2001, synthesizable for FPGA
//   - Output is flattened array for Verilog compatibility
// =======================================================================

module line_buffer #(
    parameter DATA_WIDTH   = 16,        // Bit width of each sample
    parameter KERNEL_SIZE  = 3,         // Number of samples in conv kernel
    parameter MAX_DILATION = 9,         // Maximum dilation factor
    parameter BUFFER_DEPTH = 64         // Total buffer depth (must be >= KERNEL_SIZE * MAX_DILATION)
)(
    input  wire                                  clk,
    input  wire                                  rst_n,
    input  wire                                  enable,      // Enable buffer shift
    input  wire signed [DATA_WIDTH-1:0]         data_in,     // Input sample
    input  wire [3:0]                            dilation,    // Current dilation (1, 2, 3, etc.)
    input  wire                                  clear,       // Clear buffer
    output wire signed [DATA_WIDTH*KERNEL_SIZE-1:0] window_out,  // Flattened output window
    output reg                                   valid        // Window is valid
);

    // ===================================================================
    // Internal Signals
    // ===================================================================
    
    // Circular buffer memory
    reg signed [DATA_WIDTH-1:0] buffer [0:BUFFER_DEPTH-1];
    
    // Write pointer (circular)
    reg [$clog2(BUFFER_DEPTH)-1:0] write_ptr;
    
    // Sample counter for valid flag
    reg [$clog2(BUFFER_DEPTH):0] sample_count;
    
    // Read indices for windowed access
    reg [$clog2(BUFFER_DEPTH)-1:0] read_idx [0:KERNEL_SIZE-1];
    
    // Internal window array (unpacked)
    reg signed [DATA_WIDTH-1:0] window_internal [0:KERNEL_SIZE-1];
    
    // Minimum samples needed for valid window
    wire [$clog2(BUFFER_DEPTH):0] min_samples_needed;
    assign min_samples_needed = (KERNEL_SIZE - 1) * dilation + 1;
    
    integer i;
    
    // Pack internal array to flattened output
    genvar g;
    generate
        for (g = 0; g < KERNEL_SIZE; g = g + 1) begin : gen_window_pack
            assign window_out[DATA_WIDTH*(g+1)-1 : DATA_WIDTH*g] = window_internal[g];
        end
    endgenerate
    
    // ===================================================================
    // Buffer Write Logic (Sequential)
    // ===================================================================
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            write_ptr <= 0;
            sample_count <= 0;
            for (i = 0; i < BUFFER_DEPTH; i = i + 1) begin
                buffer[i] <= 0;
            end
        end else if (clear) begin
            write_ptr <= 0;
            sample_count <= 0;
            for (i = 0; i < BUFFER_DEPTH; i = i + 1) begin
                buffer[i] <= 0;
            end
        end else if (enable) begin
            // Write new sample to buffer
            buffer[write_ptr] <= data_in;
            
            // Update write pointer (circular)
            if (write_ptr == BUFFER_DEPTH - 1)
                write_ptr <= 0;
            else
                write_ptr <= write_ptr + 1;
            
            // Update sample counter (saturate at BUFFER_DEPTH)
            if (sample_count < BUFFER_DEPTH)
                sample_count <= sample_count + 1;
        end
    end
    
    // ===================================================================
    // Window Read Index Calculation (Combinational)
    // ===================================================================
    
    always @(*) begin
        for (i = 0; i < KERNEL_SIZE; i = i + 1) begin
            // Calculate read index with dilation
            // Read from most recent samples backwards with dilation spacing
            if (write_ptr >= (i * dilation + 1)) begin
                read_idx[i] = write_ptr - (i * dilation + 1);
            end else begin
                // Wrap around circular buffer
                read_idx[i] = BUFFER_DEPTH - ((i * dilation + 1) - write_ptr);
            end
        end
    end
    
    // ===================================================================
    // Window Output Generation (Combinational)
    // ===================================================================
    
    always @(*) begin
        for (i = 0; i < KERNEL_SIZE; i = i + 1) begin
            window_internal[i] = buffer[read_idx[i]];
        end
    end
    
    // ===================================================================
    // Valid Flag Generation (Sequential)
    // ===================================================================
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid <= 1'b0;
        end else if (clear) begin
            valid <= 1'b0;
        end else begin
            // Valid when we have enough samples for current dilation
            valid <= (sample_count >= min_samples_needed);
        end
    end

endmodule
