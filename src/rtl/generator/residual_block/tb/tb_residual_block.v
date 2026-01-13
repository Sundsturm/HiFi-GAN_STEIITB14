`timescale 1ns / 1ps
`include "residual_block.v"
module tb_residual_block;

    // ========================================================================
    // Parameters
    // ========================================================================
    parameter DATA_WIDTH = 16;
    parameter FIFO_DEPTH = 32;

    // ========================================================================
    // DUT Signals
    // ========================================================================
    reg clk;
    reg rst_n;
    reg start;
    reg signed [DATA_WIDTH-1:0] d_in;
    reg d_in_valid;
    
    wire signed [DATA_WIDTH-1:0] d_out;
    wire d_out_valid;
    wire done;

    // ========================================================================
    // Clock Generation
    // ========================================================================
    initial begin
        clk = 0;
        forever #5 clk = ~clk; // Period 10ns (100MHz)
    end

    // ========================================================================
    // DUT Instantiation
    // ========================================================================
    residual_block #(
        .DATA_WIDTH(DATA_WIDTH),
        .KERNEL_SIZE(3),
        .DILATION_1(1),
        .DILATION_2(1),
        .FIFO_DEPTH(FIFO_DEPTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .done(done),
        .d_in(d_in),
        .d_in_valid(d_in_valid),
        .d_out(d_out),
        .d_out_valid(d_out_valid)
    );

    // ========================================================================
    // Test Scenario
    // ========================================================================
    integer i;

    initial begin
        // 1. Initialize
        rst_n = 0;
        start = 0;
        d_in = 0;
        d_in_valid = 0;

        // Setup Waveform Dump
        $dumpfile("tb_residual_block.vcd");
        $dumpvars(0, tb_residual_block);

        // 2. Reset Sequence
        #20;
        rst_n = 1;
        #10;
        start = 1; // Start pulse
        #10;
        start = 0;

        $display("-------------------------------------------------------------");
        $display("Starting Test: Residual Block with Mock Submodules");
        $display("Scenario: Conv1D adds +10 to input, Activations are pass-through.");
        $display("Expected: Output = Input + (Input + 10 + 10) = 2*Input + 20");
        $display("-------------------------------------------------------------");

        // 3. Inject Data Stream (5 samples)
        // We send: 10, 20, 30, 40, 50
        for (i = 1; i <= 5; i = i + 1) begin
            @(posedge clk);
            d_in <= i * 10;      // 10, 20, 30...
            d_in_valid <= 1;
        end

        // 4. Stop Injection
        @(posedge clk);
        d_in <= 0;
        d_in_valid <= 0;

        // 5. Wait for pipeline to drain
        #200;
        
        $display("Test Finished.");
        $finish;
    end

    // ========================================================================
    // Monitor / Checker
    // ========================================================================
    always @(posedge clk) begin
        if (d_out_valid) begin
            // Logic Checker based on our Dummy Modules:
            // Path: Input -> Act(pass) -> Conv(+10) -> Act(pass) -> Conv(+10)
            // Conv Path Result = Input + 20
            // Residual Sum = (Input + 20) + Input = 2*Input + 20
            
            // Reverse calculate input from output to verify logic
            // Expected for Input 10: (2*10) + 20 = 40
            $display("Time: %t | Output: %d | Valid: %b", $time, d_out, d_out_valid);
            
            // Simple assertion for the first value (10)
            if (d_out == 40) 
                $display("  -> CHECK PASS: Input 10 resulted in 40");
            else if (d_out == 60)
                $display("  -> CHECK PASS: Input 20 resulted in 60");
        end
    end

endmodule


// ============================================================================
// DUMMY MODULES (MOCKS)
// These replace the real modules just for this testbench to compile and run.
// ============================================================================

// Mock Activation: Pass-through (Identity)
module activation_unit #(
    parameter DATA_WIDTH = 16,
    parameter ACT_TYPE = "LEAKY_RELU"
)(
    input wire clk,
    input wire rst_n,
    input wire signed [DATA_WIDTH-1:0] in_data,
    input wire in_valid,
    output reg signed [DATA_WIDTH-1:0] out_data,
    output reg out_valid
);
    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            out_data <= 0;
            out_valid <= 0;
        end else begin
            out_data <= in_data; // No change
            out_valid <= in_valid;
        end
    end
endmodule

// Mock Conv1D: Adds 10 to input, 2 cycle latency
module conv1d_engine #(
    parameter DATA_WIDTH = 16,
    parameter KERNEL_SIZE = 3,
    parameter DILATION = 1
)(
    input wire clk,
    input wire rst_n,
    input wire start,
    input wire signed [DATA_WIDTH-1:0] d_in,
    input wire d_in_valid,
    output reg signed [DATA_WIDTH-1:0] d_out,
    output reg d_out_valid
);
    reg signed [DATA_WIDTH-1:0] stage1_data;
    reg stage1_valid;

    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            stage1_data <= 0;
            stage1_valid <= 0;
            d_out <= 0;
            d_out_valid <= 0;
        end else begin
            // Pipeline Stage 1
            stage1_data <= d_in;
            stage1_valid <= d_in_valid;

            // Pipeline Stage 2 (Output)
            // Adds constant 10 to simulate weight processing
            if (stage1_valid)
                d_out <= stage1_data + 16'd10;
            else
                d_out <= 0;
                
            d_out_valid <= stage1_valid;
        end
    end
endmodule