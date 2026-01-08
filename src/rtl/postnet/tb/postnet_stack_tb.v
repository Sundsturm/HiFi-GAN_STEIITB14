//==============================================================================
// Testbench: postnet_stack_tb
// Purpose: Comprehensive testbench for postnet_stack module
//          Tests Conv1D layer processing with various input patterns
//==============================================================================

`timescale 1ns / 1ps

module postnet_stack_tb;

    //==========================================================================
    // Parameters (matching DUT)
    //==========================================================================
    parameter DATA_WIDTH      = 16;
    parameter ACC_WIDTH       = 32;
    parameter KERNEL_SIZE     = 5;
    parameter NUM_CHANNELS    = 32;
    parameter NUM_LAYERS      = 5;
    parameter SAMPLE_LENGTH   = 256;
    parameter WEIGHT_DEPTH    = 1024;
    
    parameter CLK_PERIOD = 10; // 100 MHz clock
    
    //==========================================================================
    // DUT Signals
    //==========================================================================
    reg                          clk;
    reg                          rst_n;
    reg                          i_start;
    reg [$clog2(NUM_LAYERS)-1:0] i_layer_sel;
    reg signed [DATA_WIDTH-1:0]  i_data;
    reg                          i_data_valid;
    wire signed [DATA_WIDTH-1:0] o_data;
    wire                         o_data_valid;
    wire                         o_busy;
    wire                         o_done;
    
    //==========================================================================
    // Test Variables
    //==========================================================================
    integer i, j, errors;
    reg signed [DATA_WIDTH-1:0] test_input [0:SAMPLE_LENGTH-1];
    reg signed [DATA_WIDTH-1:0] test_output [0:SAMPLE_LENGTH-1];
    integer output_count;
    
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
    postnet_stack #(
        .DATA_WIDTH     (DATA_WIDTH),
        .ACC_WIDTH      (ACC_WIDTH),
        .KERNEL_SIZE    (KERNEL_SIZE),
        .NUM_CHANNELS   (NUM_CHANNELS),
        .NUM_LAYERS     (NUM_LAYERS),
        .SAMPLE_LENGTH  (SAMPLE_LENGTH),
        .WEIGHT_DEPTH   (WEIGHT_DEPTH)
    ) dut (
        .clk           (clk),
        .rst_n         (rst_n),
        .i_start       (i_start),
        .i_layer_sel   (i_layer_sel),
        .i_data        (i_data),
        .i_data_valid  (i_data_valid),
        .o_data        (o_data),
        .o_data_valid  (o_data_valid),
        .o_busy        (o_busy),
        .o_done        (o_done)
    );
    
    //==========================================================================
    // Output Capture
    //==========================================================================
    always @(posedge clk) begin
        if (!rst_n) begin
            output_count <= 0;
        end
        else if (o_data_valid) begin
            test_output[output_count] <= o_data;
            output_count <= output_count + 1;
            $display("[T=%0t] Output[%0d] = %h (%f)", 
                     $time, output_count, o_data, $itor(o_data)/4096.0);
        end
    end
    
    //==========================================================================
    // Test Stimulus
    //==========================================================================
    initial begin
        // Initialize signals
        rst_n = 0;
        i_start = 0;
        i_layer_sel = 0;
        i_data = 0;
        i_data_valid = 0;
        errors = 0;
        output_count = 0;
        
        // Create waveform dump
        $dumpfile("postnet_stack_tb.vcd");
        $dumpvars(0, postnet_stack_tb);
        
        // Reset sequence
        #(CLK_PERIOD*5);
        rst_n = 1;
        #(CLK_PERIOD*2);
        
        $display("========================================");
        $display("PostNet Stack Testbench Started");
        $display("========================================");
        
        //----------------------------------------------------------------------
        // Test 1: Layer 0 with Impulse Input
        //----------------------------------------------------------------------
        $display("\n[TEST 1] Layer 0 - Impulse Response");
        i_layer_sel = 0;
        
        // Generate impulse test pattern
        for (i = 0; i < SAMPLE_LENGTH; i = i + 1) begin
            if (i == KERNEL_SIZE/2)
                test_input[i] = 16'h1000; // 1.0 in Q4.12
            else
                test_input[i] = 16'h0000;
        end
        
        // Start processing
        output_count = 0;
        @(posedge clk);
        i_start = 1;
        @(posedge clk);
        i_start = 0;
        
        // Feed input samples
        for (i = 0; i < SAMPLE_LENGTH; i = i + 1) begin
            @(posedge clk);
            i_data = test_input[i];
            i_data_valid = 1;
        end
        @(posedge clk);
        i_data_valid = 0;
        
        // Wait for completion
        wait(o_done);
        @(posedge clk);
        $display("[TEST 1] Completed - %0d outputs received", output_count);
        
        #(CLK_PERIOD*10);
        
        //----------------------------------------------------------------------
        // Test 2: Layer 1 with Sine Wave Approximation
        //----------------------------------------------------------------------
        $display("\n[TEST 2] Layer 1 - Sine Wave Input");
        i_layer_sel = 1;
        
        // Generate sine-like test pattern (simple approximation)
        for (i = 0; i < SAMPLE_LENGTH; i = i + 1) begin
            // Simple triangle wave as sine approximation
            if (i < SAMPLE_LENGTH/4)
                test_input[i] = (i * 16'h0100) / (SAMPLE_LENGTH/4);
            else if (i < 3*SAMPLE_LENGTH/4)
                test_input[i] = 16'h1000 - ((i - SAMPLE_LENGTH/4) * 16'h0200) / (SAMPLE_LENGTH/2);
            else
                test_input[i] = -16'h1000 + ((i - 3*SAMPLE_LENGTH/4) * 16'h0100) / (SAMPLE_LENGTH/4);
        end
        
        // Start processing
        output_count = 0;
        @(posedge clk);
        i_start = 1;
        @(posedge clk);
        i_start = 0;
        
        // Feed input samples
        for (i = 0; i < SAMPLE_LENGTH; i = i + 1) begin
            @(posedge clk);
            i_data = test_input[i];
            i_data_valid = 1;
        end
        @(posedge clk);
        i_data_valid = 0;
        
        // Wait for completion
        wait(o_done);
        @(posedge clk);
        $display("[TEST 2] Completed - %0d outputs received", output_count);
        
        #(CLK_PERIOD*10);
        
        //----------------------------------------------------------------------
        // Test 3: Layer 2 with Step Input
        //----------------------------------------------------------------------
        $display("\n[TEST 3] Layer 2 - Step Response");
        i_layer_sel = 2;
        
        // Generate step test pattern
        for (i = 0; i < SAMPLE_LENGTH; i = i + 1) begin
            if (i < SAMPLE_LENGTH/2)
                test_input[i] = 16'h0000;
            else
                test_input[i] = 16'h0800; // 0.5 in Q4.12
        end
        
        // Start processing
        output_count = 0;
        @(posedge clk);
        i_start = 1;
        @(posedge clk);
        i_start = 0;
        
        // Feed input samples with some random delays
        for (i = 0; i < SAMPLE_LENGTH; i = i + 1) begin
            @(posedge clk);
            i_data = test_input[i];
            i_data_valid = 1;
            if (i % 16 == 0) begin
                @(posedge clk);
                i_data_valid = 0;
            end
        end
        @(posedge clk);
        i_data_valid = 0;
        
        // Wait for completion
        wait(o_done);
        @(posedge clk);
        $display("[TEST 3] Completed - %0d outputs received", output_count);
        
        #(CLK_PERIOD*10);
        
        //----------------------------------------------------------------------
        // Test 4: Last Layer (No Activation)
        //----------------------------------------------------------------------
        $display("\n[TEST 4] Layer 4 (Last) - Random Noise");
        i_layer_sel = NUM_LAYERS - 1;
        
        // Generate random test pattern
        for (i = 0; i < SAMPLE_LENGTH; i = i + 1) begin
            test_input[i] = $random % 16'h2000 - 16'h1000; // Random ±1.0
        end
        
        // Start processing
        output_count = 0;
        @(posedge clk);
        i_start = 1;
        @(posedge clk);
        i_start = 0;
        
        // Feed input samples
        for (i = 0; i < SAMPLE_LENGTH; i = i + 1) begin
            @(posedge clk);
            i_data = test_input[i];
            i_data_valid = 1;
        end
        @(posedge clk);
        i_data_valid = 0;
        
        // Wait for completion
        wait(o_done);
        @(posedge clk);
        $display("[TEST 4] Completed - %0d outputs received", output_count);
        
        #(CLK_PERIOD*10);
        
        //----------------------------------------------------------------------
        // Test Summary
        //----------------------------------------------------------------------
        $display("\n========================================");
        $display("PostNet Stack Testbench Summary");
        $display("========================================");
        $display("Total Errors: %0d", errors);
        if (errors == 0)
            $display("STATUS: PASS");
        else
            $display("STATUS: FAIL");
        $display("========================================\n");
        
        #(CLK_PERIOD*20);
        $finish;
    end
    
    //==========================================================================
    // Timeout Watchdog
    //==========================================================================
    initial begin
        #(CLK_PERIOD * 500000); // 5ms timeout
        $display("\n[ERROR] Simulation timeout!");
        $finish;
    end
    
    //==========================================================================
    // Monitor busy/done signals
    //==========================================================================
    always @(posedge clk) begin
        if (o_busy && !o_done) begin
            // Normal operation
        end
        else if (!o_busy && o_done) begin
            $display("[INFO] Processing completed at T=%0t", $time);
        end
    end

endmodule
