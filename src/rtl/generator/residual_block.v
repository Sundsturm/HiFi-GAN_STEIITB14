`timescale 1ns / 1ps

module residual_block #(
    parameter DATA_WIDTH = 16,
    parameter KERNEL_SIZE = 3,
    parameter DILATION_1 = 1,
    parameter DILATION_2 = 1,
    parameter CHANNELS = 128,  // Number of input/output channels
    parameter FIFO_DEPTH = 128 // Estimated depth to cover conv latency
)(
    input  wire clk,
    input  wire rst_n,
    
    // Control Signals
    input  wire start,        // Signal to reset internal states/counters if needed
    output wire done,         // Indicates block processing is complete (optional/pipeline dependent)
    
    // Data Input
    input  wire signed [DATA_WIDTH-1:0] d_in,
    input  wire d_in_valid,
    
    // Data Output
    output reg  signed [DATA_WIDTH-1:0] d_out,
    output reg  d_out_valid
);

    // ========================================================================
    // Internal Signals & Wires
    // ========================================================================
    
    // --- Path 1: Activation 1 ---
    wire signed [DATA_WIDTH-1:0] act1_out;
    wire act1_valid;

    // --- Path 1: Conv1D Layer 1 ---
    wire signed [DATA_WIDTH-1:0] conv1_out;
    wire conv1_valid;
    
    // --- Path 1: Activation 2 ---
    wire signed [DATA_WIDTH-1:0] act2_out;
    wire act2_valid;

    // --- Path 1: Conv1D Layer 2 ---
    wire signed [DATA_WIDTH-1:0] conv2_out;
    wire conv2_valid;

    // --- Path 2: Skip Connection (Delay FIFO) ---
    wire signed [DATA_WIDTH-1:0] skip_data_out;
    // Note: FIFO read enable is driven by the validity of the processing path output
    // to ensure alignment.
    
    // --- Result Summation ---
    wire signed [DATA_WIDTH:0] sum_temp; // 1 bit larger for overflow check
    wire signed [DATA_WIDTH-1:0] sum_saturated;

    // ========================================================================
    // 1. First Activation (LeakyReLU)
    // ========================================================================
    activation_unit #(
        .DATA_WIDTH(DATA_WIDTH),
        .ACT_TYPE("LEAKY_RELU") 
    ) u_act_1 (
        .clk(clk),
        .rst_n(rst_n),
        .in_data(d_in),
        .in_valid(d_in_valid),
        .out_data(act1_out),
        .out_valid(act1_valid)
    );

    // ========================================================================
    // 2. First Convolution (Dilated)
    // ========================================================================
    // Assumes conv1d_engine handles weights internally or via external bus.
    // For this module, we assume it's a structural wrapper.
    conv1d_engine #(
        .DATA_WIDTH(DATA_WIDTH),
        .KERNEL_SIZE(KERNEL_SIZE),
        .DILATION(DILATION_1)
    ) u_conv_1 (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .d_in(act1_out),
        .d_in_valid(act1_valid),
        .d_out(conv1_out),
        .d_out_valid(conv1_valid)
        // Note: Weight interface omitted for brevity, usually connects to weight_mem
    );

    // ========================================================================
    // 3. Second Activation (LeakyReLU)
    // ========================================================================
    activation_unit #(
        .DATA_WIDTH(DATA_WIDTH),
        .ACT_TYPE("LEAKY_RELU")
    ) u_act_2 (
        .clk(clk),
        .rst_n(rst_n),
        .in_data(conv1_out),
        .in_valid(conv1_valid),
        .out_data(act2_out),
        .out_valid(act2_valid)
    );

    // ========================================================================
    // 4. Second Convolution (Dilated or Standard)
    // ========================================================================
    conv1d_engine #(
        .DATA_WIDTH(DATA_WIDTH),
        .KERNEL_SIZE(KERNEL_SIZE),
        .DILATION(DILATION_2)
    ) u_conv_2 (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .d_in(act2_out),
        .d_in_valid(act2_valid),
        .d_out(conv2_out),
        .d_out_valid(conv2_valid)
    );

    // ========================================================================
    // 5. Skip Connection Buffer (FIFO)
    // ========================================================================
    // We must store the original 'd_in' and read it only when 'conv2_valid' is high.
    // This acts as a variable delay line matching the pipeline latency.
    
    reg signed [DATA_WIDTH-1:0] fifo_mem [0:FIFO_DEPTH-1];
    reg [$clog2(FIFO_DEPTH)-1:0] wr_ptr;
    reg [$clog2(FIFO_DEPTH)-1:0] rd_ptr;
    reg [$clog2(FIFO_DEPTH):0]   count;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_ptr <= 0;
            rd_ptr <= 0;
            count  <= 0;
        end else begin
            // Write when input is valid
            if (d_in_valid) begin
                fifo_mem[wr_ptr] <= d_in;
                if (wr_ptr == FIFO_DEPTH-1)
                    wr_ptr <= 0;
                else
                    wr_ptr <= wr_ptr + 1;
            end

            // Read when the convolution path produces a result
            if (conv2_valid) begin
                if (rd_ptr == FIFO_DEPTH-1)
                    rd_ptr <= 0;
                else
                    rd_ptr <= rd_ptr + 1;
            end
            
            // Count tracking (optional, for debug/overflow protection)
            if (d_in_valid && !conv2_valid)
                count <= count + 1;
            else if (!d_in_valid && conv2_valid)
                count <= count - 1;
        end
    end

    assign skip_data_out = fifo_mem[rd_ptr];

    // ========================================================================
    // 6. Residual Addition & Saturation
    // ========================================================================
    // Sum = Conv_Result + Original_Input
    assign sum_temp = conv2_out + skip_data_out;

    // Saturation Logic (clamping to min/max of signed fixed point)
    // Max positive: 011...1
    // Min negative: 100...0
    wire signed [DATA_WIDTH-1:0] max_pos = {1'b0, {(DATA_WIDTH-1){1'b1}}};
    wire signed [DATA_WIDTH-1:0] min_neg = {1'b1, {(DATA_WIDTH-1){1'b0}}};

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            d_out <= 0;
            d_out_valid <= 0;
        end else begin
            if (conv2_valid) begin
                d_out_valid <= 1'b1;
                
                // Overflow Check
                // If both operands positive and result negative -> Overflow
                // If both operands negative and result positive -> Underflow
                if ((conv2_out > 0 && skip_data_out > 0 && sum_temp < 0)) begin
                    d_out <= max_pos;
                end else if ((conv2_out < 0 && skip_data_out < 0 && sum_temp > 0)) begin
                    d_out <= min_neg;
                end else begin
                    d_out <= sum_temp[DATA_WIDTH-1:0];
                end
            end else begin
                d_out_valid <= 1'b0;
            end
        end
    end
    
    // Done signal could be derived from internal counters or passed from engines
    assign done = !d_out_valid; // Simplified

endmodule