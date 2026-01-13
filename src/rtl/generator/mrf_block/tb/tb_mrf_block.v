`timescale 1ns / 1ps

// =========================================================
// MODULE 1: MRF BLOCK (Design)
// =========================================================
module mrf_block #(
    parameter DWIDTH = 16,
    parameter K1 = 3,
    parameter K2 = 7,
    parameter K3 = 11,
    parameter Q_SCALE = 0
)(
    input  wire                clk,
    input  wire                rst_n,
    input  wire                start,
    input  wire signed [DWIDTH-1:0] din,
    output reg                 done,
    output reg  signed [DWIDTH-1:0] dout
);
    wire signed [DWIDTH-1:0] branch1_out, branch2_out, branch3_out;
    wire branch1_done, branch2_done, branch3_done;
    wire signed [DWIDTH+1:0] temp_sum;

    // Instansiasi
    residual_block #(.DWIDTH(DWIDTH), .KERNEL_SIZE(K1)) u_res_1 (
        .clk(clk), .rst_n(rst_n), .start(start), .din(din), .done(branch1_done), .dout(branch1_out)
    );
    residual_block #(.DWIDTH(DWIDTH), .KERNEL_SIZE(K2)) u_res_2 (
        .clk(clk), .rst_n(rst_n), .start(start), .din(din), .done(branch2_done), .dout(branch2_out)
    );
    residual_block #(.DWIDTH(DWIDTH), .KERNEL_SIZE(K3)) u_res_3 (
        .clk(clk), .rst_n(rst_n), .start(start), .din(din), .done(branch3_done), .dout(branch3_out)
    );

    assign temp_sum = branch1_out + branch2_out + branch3_out;
    
    // Saturation Constants
    localparam signed [DWIDTH-1:0] MAX_VAL = {1'b0, {(DWIDTH-1){1'b1}}};
    localparam signed [DWIDTH-1:0] MIN_VAL = {1'b1, {(DWIDTH-1){1'b0}}};

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dout <= 0; done <= 0;
        end else begin
            if (branch1_done && branch2_done && branch3_done) begin
                done <= 1'b1;
                if (temp_sum > MAX_VAL) dout <= MAX_VAL;
                else if (temp_sum < MIN_VAL) dout <= MIN_VAL;
                else dout <= temp_sum[DWIDTH-1:0];
            end else begin
                done <= 1'b0;
            end
        end
    end
endmodule

// =========================================================
// MODULE 2: RESIDUAL BLOCK (Dummy/Mock for Simulation)
// =========================================================
module residual_block #(
    parameter DWIDTH = 16,
    parameter KERNEL_SIZE = 3
)(
    input wire clk, rst_n, start,
    input wire signed [DWIDTH-1:0] din,
    output reg done,
    output reg signed [DWIDTH-1:0] dout
);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dout <= 0; done <= 0;
        end else begin
            if (start) begin
                done <= 0;
                dout <= din; // Pass through
                repeat(2) @(posedge clk); // Delay
                done <= 1;
            end else done <= 0;
        end
    end
endmodule

// =========================================================
// MODULE 3: TESTBENCH
// =========================================================
module tb_full_check();
    parameter DWIDTH = 16;
    reg clk, rst_n, start;
    reg signed [DWIDTH-1:0] din;
    wire done;
    wire signed [DWIDTH-1:0] dout;

    mrf_block #(.DWIDTH(DWIDTH)) uut (
        .clk(clk), .rst_n(rst_n), .start(start), .din(din), .done(done), .dout(dout)
    );

    always #5 clk = ~clk;

    initial begin
        clk = 0; rst_n = 0; start = 0; din = 0;
        #20 rst_n = 1; #20;

        // Test
        din = 1000; start = 1; #10 start = 0;
        wait(done);
        $display("Output: %d (Expect 3000)", dout);
        
        #50 $finish;
    end
endmodule