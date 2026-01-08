`timescale 1ns / 1ps

module generator_top #(
    parameter DWIDTH = 16,
    // Stage 1 Parameters
    parameter UPSAMPLE_RATE_1 = 8,
    parameter MRF_K1_S1 = 3,
    parameter MRF_K2_S1 = 7,
    parameter MRF_K3_S1 = 11,
    // Stage 2 Parameters
    parameter UPSAMPLE_RATE_2 = 8,
    parameter MRF_K1_S2 = 3,
    parameter MRF_K2_S2 = 7,
    parameter MRF_K3_S2 = 11
)(
    input  wire                clk,
    input  wire                rst_n,
    
    // Top-Level Control
    input  wire                start,      // Trigger from Top FSM
    output wire                done,       // Signal to Top FSM
    
    // Data Interfaces
    input  wire signed [DWIDTH-1:0] din,   // Input Feature (Mel-Spectrogram frame)
    output wire signed [DWIDTH-1:0] dout   // Generated Raw Waveform
);

    //=========================================================================
    // 1. Internal Interconnect Signals
    //=========================================================================
    
    // Control Signals (From FSM to Stages)
    wire start_stage1_up;
    wire start_stage1_mrf;
    wire start_stage2_up;
    wire start_stage2_mrf;

    // Status Signals (From Stages to FSM)
    wire done_stage1_up;
    wire done_stage1_mrf;
    wire done_stage2_up;
    wire done_stage2_mrf;

    // Data Path Signals (Stage to Stage chaining)
    wire signed [DWIDTH-1:0] data_s1_up_to_mrf;
    wire signed [DWIDTH-1:0] data_s1_to_s2;
    wire signed [DWIDTH-1:0] data_s2_up_to_mrf;
    
    //=========================================================================
    // 2. Control Module: Generator FSM
    //=========================================================================
    
    generator_fsm u_gen_fsm (
        .clk            (clk),
        .rst_n          (rst_n),
        
        // Interaction with Top Level
        .gen_start      (start),
        .gen_done       (done),
        
        // Control for Stage 1
        .s1_up_start    (start_stage1_up),
        .s1_up_done     (done_stage1_up),
        .s1_mrf_start   (start_stage1_mrf),
        .s1_mrf_done    (done_stage1_mrf),
        
        // Control for Stage 2
        .s2_up_start    (start_stage2_up),
        .s2_up_done     (done_stage2_up),
        .s2_mrf_start   (start_stage2_mrf),
        .s2_mrf_done    (done_stage2_mrf)
    );

    //=========================================================================
    // 3. Stage 1: Upsample x8 -> MRF
    //=========================================================================

    upsample_module #(
        .DWIDTH(DWIDTH),
        .RATE(UPSAMPLE_RATE_1)
    ) u_stage1_upsample (
        .clk    (clk),
        .rst_n  (rst_n),
        .start  (start_stage1_up),
        .din    (din),                  // Input from outside
        .done   (done_stage1_up),
        .dout   (data_s1_up_to_mrf)
    );

    mrf_block #(
        .DWIDTH(DWIDTH),
        .K1(MRF_K1_S1), .K2(MRF_K2_S1), .K3(MRF_K3_S1)
    ) u_stage1_mrf (
        .clk    (clk),
        .rst_n  (rst_n),
        .start  (start_stage1_mrf),
        .din    (data_s1_up_to_mrf),    // Connect from Upsample
        .done   (done_stage1_mrf),
        .dout   (data_s1_to_s2)
    );

    //=========================================================================
    // 4. Stage 2: Upsample x8 -> MRF
    //=========================================================================

    upsample_module #(
        .DWIDTH(DWIDTH),
        .RATE(UPSAMPLE_RATE_2)
    ) u_stage2_upsample (
        .clk    (clk),
        .rst_n  (rst_n),
        .start  (start_stage2_up),
        .din    (data_s1_to_s2),        // Connect from Stage 1
        .done   (done_stage2_up),
        .dout   (data_s2_up_to_mrf)
    );

    mrf_block #(
        .DWIDTH(DWIDTH),
        .K1(MRF_K1_S2), .K2(MRF_K2_S2), .K3(MRF_K3_S2)
    ) u_stage2_mrf (
        .clk    (clk),
        .rst_n  (rst_n),
        .start  (start_stage2_mrf),
        .din    (data_s2_up_to_mrf),    // Connect from Upsample
        .done   (done_stage2_mrf),
        .dout   (dout)                  // Final Output (or to next stage)
    );

endmodule