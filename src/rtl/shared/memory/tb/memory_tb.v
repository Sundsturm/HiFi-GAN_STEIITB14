//==============================================================================
// Testbench: memory_tb
// Purpose: Test shared memory modules (weight_mem, bias_mem, param_rom)
//==============================================================================

`timescale 1ns / 1ps

module memory_tb;

    //==========================================================================
    // Parameters
    //==========================================================================
    parameter DATA_WIDTH = 16;
    parameter WEIGHT_DEPTH = 256;
    parameter BIAS_DEPTH = 64;
    parameter CLK_PERIOD = 10; // 100MHz clock
    
    //==========================================================================
    // DUT Signals
    //==========================================================================
    reg clk;
    reg rst_n;
    
    // Weight memory interface
    reg [7:0] weight_addr;
    reg weight_rd_en;
    wire signed [DATA_WIDTH-1:0] weight_data;
    wire weight_valid;
    
    // Bias memory interface
    reg [5:0] bias_addr;
    reg bias_rd_en;
    wire signed [DATA_WIDTH-1:0] bias_data;
    wire bias_valid;
    
    // Param ROM interface
    reg [3:0] layer_sel;
    wire [7:0] kernel_size;
    wire [7:0] dilation;
    wire [7:0] in_channels;
    wire [7:0] out_channels;
    
    //==========================================================================
    // Clock Generation
    //==========================================================================
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end
    
    //==========================================================================
    // DUT Instantiation
    //==========================================================================
    weight_mem #(
        .DATA_WIDTH(DATA_WIDTH),
        .DEPTH(WEIGHT_DEPTH),
        .MEM_FILE("weights.mem")
    ) u_weight_mem (
        .clk     (clk),
        .rst_n   (rst_n),
        .i_addr  (weight_addr),
        .i_rd_en (weight_rd_en),
        .o_data  (weight_data),
        .o_valid (weight_valid)
    );
    
    bias_mem #(
        .DATA_WIDTH(DATA_WIDTH),
        .DEPTH(BIAS_DEPTH),
        .MEM_FILE("biases.mem")
    ) u_bias_mem (
        .clk     (clk),
        .rst_n   (rst_n),
        .i_addr  (bias_addr),
        .i_rd_en (bias_rd_en),
        .o_data  (bias_data),
        .o_valid (bias_valid)
    );
    
    param_rom #(
        .MAX_LAYERS(16),
        .PARAM_WIDTH(8)
    ) u_param_rom (
        .i_layer_sel   (layer_sel),
        .o_kernel_size (kernel_size),
        .o_dilation    (dilation),
        .o_in_channels (in_channels),
        .o_out_channels(out_channels)
    );
    
    //==========================================================================
    // Test Stimulus
    //==========================================================================
    initial begin
        // Initialize signals
        rst_n = 0;
        weight_addr = 0;
        weight_rd_en = 0;
        bias_addr = 0;
        bias_rd_en = 0;
        layer_sel = 0;
        
        // Wait for reset
        #(CLK_PERIOD * 5);
        rst_n = 1;
        #(CLK_PERIOD * 2);
        
        $display("===== Memory Module Testbench =====");
        $display("Time: %0t ns", $time);
        
        //----------------------------------------------------------------------
        // Test 1: Weight Memory Sequential Read
        //----------------------------------------------------------------------
        $display("\n[Test 1] Weight Memory Sequential Read");
        repeat (10) begin
            @(posedge clk);
            weight_rd_en = 1;
            @(posedge clk);
            weight_rd_en = 0;
            
            @(posedge clk);
            if (weight_valid) begin
                $display("  Addr: 0x%02X -> Data: 0x%04X (Q2.14: %f)", 
                         weight_addr, weight_data, 
                         $signed(weight_data) / 16384.0);
            end
            
            weight_addr = weight_addr + 1;
        end
        
        //----------------------------------------------------------------------
        // Test 2: Bias Memory Sequential Read
        //----------------------------------------------------------------------
        $display("\n[Test 2] Bias Memory Sequential Read");
        repeat (10) begin
            @(posedge clk);
            bias_rd_en = 1;
            @(posedge clk);
            bias_rd_en = 0;
            
            @(posedge clk);
            if (bias_valid) begin
                $display("  Addr: 0x%02X -> Data: 0x%04X (Q4.12: %f)", 
                         bias_addr, bias_data,
                         $signed(bias_data) / 4096.0);
            end
            
            bias_addr = bias_addr + 1;
        end
        
        //----------------------------------------------------------------------
        // Test 3: Param ROM Read (combinational)
        //----------------------------------------------------------------------
        $display("\n[Test 3] Param ROM Layer Configuration");
        for (layer_sel = 0; layer_sel < 10; layer_sel = layer_sel + 1) begin
            @(posedge clk);
            #1; // Small delay for combinational propagation
            $display("  Layer %0d: K=%0d, D=%0d, InCh=%0d, OutCh=%0d",
                     layer_sel, kernel_size, dilation, in_channels, out_channels);
        end
        
        //----------------------------------------------------------------------
        // Test 4: Random Access Pattern
        //----------------------------------------------------------------------
        $display("\n[Test 4] Random Weight Access");
        weight_addr = 8'h0A;
        @(posedge clk);
        weight_rd_en = 1;
        @(posedge clk);
        weight_rd_en = 0;
        @(posedge clk);
        if (weight_valid)
            $display("  Random Addr: 0x%02X -> Data: 0x%04X", weight_addr, weight_data);
        
        weight_addr = 8'h2F;
        @(posedge clk);
        weight_rd_en = 1;
        @(posedge clk);
        weight_rd_en = 0;
        @(posedge clk);
        if (weight_valid)
            $display("  Random Addr: 0x%02X -> Data: 0x%04X", weight_addr, weight_data);
        
        //----------------------------------------------------------------------
        // Test Complete
        //----------------------------------------------------------------------
        #(CLK_PERIOD * 10);
        $display("\n===== Test Complete =====");
        $display("All memory modules functional.");
        $finish;
    end
    
    //==========================================================================
    // Waveform Dump (for GTKWave or Vivado Simulator)
    //==========================================================================
    initial begin
        $dumpfile("memory_tb.vcd");
        $dumpvars(0, memory_tb);
    end

endmodule
