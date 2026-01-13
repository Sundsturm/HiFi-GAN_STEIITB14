`timescale 1ns / 1ps
`include "generator_top.v"
`include "mrf_block.v"
`include "residual_block.v"

module tb_generator_top();

    //-------------------------------------------------------------------------
    // 1. Parameters & Signals
    //-------------------------------------------------------------------------
    parameter DWIDTH = 16;
    
    reg clk;
    reg rst_n;
    reg start;
    reg signed [DWIDTH-1:0] din;
    
    wire done;
    wire signed [DWIDTH-1:0] dout;

    //-------------------------------------------------------------------------
    // 2. Instantiate Unit Under Test (UUT)
    //-------------------------------------------------------------------------
    generator_top #(
        .DWIDTH(DWIDTH),
        .UPSAMPLE_RATE_1(8), .MRF_K1_S1(3), .MRF_K2_S1(7), .MRF_K3_S1(11),
        .UPSAMPLE_RATE_2(8), .MRF_K1_S2(3), .MRF_K2_S2(7), .MRF_K3_S2(11)
    ) uut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .done(done),
        .din(din),
        .dout(dout)
    );

    //-------------------------------------------------------------------------
    // 3. Clock Generation
    //-------------------------------------------------------------------------
    always #5 clk = ~clk; // 100MHz

    //-------------------------------------------------------------------------
    // 4. Test Scenario
    //-------------------------------------------------------------------------
    initial begin
        // Initialize
        dumpfile("tb_generator_top.vcd");
        dumpvars(0, tb_generator_top);
        clk = 0;
        rst_n = 0;
        start = 0;
        din = 0;

        // Reset
        #20 rst_n = 1;
        #20;

        // TEST CASE 1: Basic Flow
        $display("--- Starting Test Case 1: Pipeline Flow ---");
        
        // Input Value: 100
        // Expectation: 
        // Stage 1 (Upsample Dummy pass-through) -> 100
        // Stage 1 (MRF Dummy pass-through)      -> 100
        // Stage 2 (Upsample Dummy pass-through) -> 100
        // Stage 2 (MRF Dummy pass-through)      -> 100
        // Final Output should be 100.
        
        din = 16'd100;
        start = 1;
        #10 start = 0; // Pulse start

        // Wait for completion
        wait(done);
        
        #10;
        if (dout == 16'd100) begin
            $display("PASS: Data propagated correctly through all stages. Output: %d", dout);
        end else begin
            $display("FAIL: Expected 100, got %d", dout);
        end

        #50;
        $finish;
    end

endmodule


// ============================================================================
// DUMMY MODULES (Hanya untuk Simulasi agar tidak Error "Module Not Found")
// ============================================================================

// 1. DUMMY UPSAMPLE
module upsample_module #(parameter DWIDTH=16, RATE=8)(
    input clk, rst_n, start,
    input signed [DWIDTH-1:0] din,
    output reg done,
    output reg signed [DWIDTH-1:0] dout
);
    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin done<=0; dout<=0; end
        else if(start) begin
            done <= 0;
            dout <= din; // Pass-through data
            repeat(2) @(posedge clk); // Simulate latency
            done <= 1;
        end else done <= 0;
    end
endmodule

// 2. DUMMY FSM (Penting: Mengatur urutan trigger)
module generator_fsm (
    input clk, rst_n, gen_start,
    input s1_up_done, s1_mrf_done, s2_up_done, s2_mrf_done,
    output reg gen_done,
    output reg s1_up_start, s1_mrf_start, s2_up_start, s2_mrf_start
);
    // Simple State Machine for Simulation Sequence
    reg [2:0] state;
    localparam IDLE=0, S1_UP=1, S1_MRF=2, S2_UP=3, S2_MRF=4, DONE=5;

    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            state <= IDLE;
            s1_up_start<=0; s1_mrf_start<=0; s2_up_start<=0; s2_mrf_start<=0; gen_done<=0;
        end else begin
            case(state)
                IDLE: if(gen_start) begin state <= S1_UP; s1_up_start <= 1; gen_done<=0; end
                S1_UP: begin 
                    s1_up_start <= 0; 
                    if(s1_up_done) begin state <= S1_MRF; s1_mrf_start <= 1; end 
                end
                S1_MRF: begin 
                    s1_mrf_start <= 0; 
                    if(s1_mrf_done) begin state <= S2_UP; s2_up_start <= 1; end 
                end
                S2_UP: begin 
                    s2_up_start <= 0; 
                    if(s2_up_done) begin state <= S2_MRF; s2_mrf_start <= 1; end 
                end
                S2_MRF: begin 
                    s2_mrf_start <= 0; 
                    if(s2_mrf_done) begin state <= DONE; gen_done <= 1; end 
                end
                DONE: begin 
                    state <= IDLE; gen_done <= 0; 
                end
            endcase
        end
    end
endmodule

// 3. DUMMY MRF (Jika mrf_block belum ada di project, gunakan ini)
// Jika mrf_block.v SUDAH ada di project tree Anda, hapus modul di bawah ini
// untuk menghindari "Redefinition Error".
/*
module mrf_block #(parameter DWIDTH=16, K1=3, K2=7, K3=11)(
    input clk, rst_n, start,
    input signed [DWIDTH-1:0] din,
    output reg done,
    output reg signed [DWIDTH-1:0] dout
);
    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin done<=0; dout<=0; end
        else if(start) begin
            done <= 0;
            dout <= din; // Pass-through
            repeat(2) @(posedge clk);
            done <= 1;
        end else done <= 0;
    end
endmodule
*/