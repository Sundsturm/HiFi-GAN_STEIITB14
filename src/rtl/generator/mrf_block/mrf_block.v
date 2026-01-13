`timescale 1ns / 1ps

module mrf_block #(
    parameter DWIDTH = 16,          // Data width (fixed-point)
    parameter K1 = 3,               // Kernel size for Branch 1
    parameter K2 = 7,               // Kernel size for Branch 2
    parameter K3 = 11,              // Kernel size for Branch 3
    // Dilation parameters would typically be passed down to residual blocks
    // but are simplified here for structural clarity.
    parameter Q_SCALE = 0           // Placeholder for Q-format scaling if needed
)(
    input  wire                clk,
    input  wire                rst_n,
    input  wire                start,      // Handshake: Start trigger
    input  wire signed [DWIDTH-1:0] din,   // Input data (from Upsample)
    
    output reg                 done,       // Handshake: Done signal
    output reg  signed [DWIDTH-1:0] dout   // Output data (Sum of all branches)
);

    //-------------------------------------------------------------------------
    // 1. Internal Signals
    //-------------------------------------------------------------------------
    wire signed [DWIDTH-1:0] branch1_out;
    wire signed [DWIDTH-1:0] branch2_out;
    wire signed [DWIDTH-1:0] branch3_out;

    wire branch1_done;
    wire branch2_done;
    wire branch3_done;

    // Temporary sum with extra bits to handle overflow before saturation
    // Summing 3 numbers requires 2 extra bits (log2(3) ~ 1.58 -> 2 bits)
    wire signed [DWIDTH+1:0] temp_sum;

    //-------------------------------------------------------------------------
    // 2. Instantiate Residual Blocks (Branches)
    //-------------------------------------------------------------------------
    
    // Branch 1: Kernel Size K1
    residual_block #(
        .DWIDTH(DWIDTH),
        .KERNEL_SIZE(K1)
    ) u_res_block_1 (
        .clk    (clk),
        .rst_n  (rst_n),
        .start  (start),
        .din    (din),
        .done   (branch1_done),
        .dout   (branch1_out)
    );

    // Branch 2: Kernel Size K2
    residual_block #(
        .DWIDTH(DWIDTH),
        .KERNEL_SIZE(K2)
    ) u_res_block_2 (
        .clk    (clk),
        .rst_n  (rst_n),
        .start  (start),
        .din    (din),
        .done   (branch2_done),
        .dout   (branch2_out)
    );

    // Branch 3: Kernel Size K3
    residual_block #(
        .DWIDTH(DWIDTH),
        .KERNEL_SIZE(K3)
    ) u_res_block_3 (
        .clk    (clk),
        .rst_n  (rst_n),
        .start  (start),
        .din    (din),
        .done   (branch3_done),
        .dout   (branch3_out)
    );

    //-------------------------------------------------------------------------
    // 3. Summation and Saturation Logic
    //-------------------------------------------------------------------------
    
    // Add all branches: A + B + C
    assign temp_sum = branch1_out + branch2_out + branch3_out;

    // Constants for saturation (Max positive and Min negative values)
    localparam signed [DWIDTH-1:0] MAX_VAL = {1'b0, {(DWIDTH-1){1'b1}}}; // e.g., 011...1
    localparam signed [DWIDTH-1:0] MIN_VAL = {1'b1, {(DWIDTH-1){1'b0}}}; // e.g., 100...0

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dout <= 0;
            done <= 0;
        end else begin
            // Wait for all branches to complete
            if (branch1_done && branch2_done && branch3_done) begin
                done <= 1'b1;
                
                // SATURATION LOGIC
                // Check if temp_sum exceeds the positive limit of DWIDTH
                if (temp_sum > MAX_VAL) begin
                    dout <= MAX_VAL;
                end 
                // Check if temp_sum exceeds the negative limit of DWIDTH
                else if (temp_sum < MIN_VAL) begin
                    dout <= MIN_VAL;
                end 
                // Otherwise, take the lower bits
                else begin
                    dout <= temp_sum[DWIDTH-1:0];
                end
            end else begin
                done <= 1'b0;
                // Keep dout stable or reset depending on protocol requirements
                // Here we keep it stable until new valid data is ready
            end
        end
    end

endmodule