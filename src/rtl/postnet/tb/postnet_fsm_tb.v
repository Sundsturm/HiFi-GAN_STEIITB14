//==============================================================================
// Testbench: postnet_fsm_tb
// Purpose: Comprehensive testbench for postnet_fsm module
//          Tests FSM state transitions and layer sequencing control
//==============================================================================

`timescale 1ns / 1ps

module postnet_fsm_tb;

    //==========================================================================
    // Parameters (matching DUT)
    //==========================================================================
    parameter NUM_LAYERS    = 5;
    parameter SAMPLE_LENGTH = 256;
    parameter CLK_PERIOD    = 10; // 100 MHz clock
    
    //==========================================================================
    // DUT Signals
    //==========================================================================
    reg                          clk;
    reg                          rst_n;
    reg                          i_start;
    reg                          i_stack_done;
    wire                         o_stack_start;
    wire [$clog2(NUM_LAYERS)-1:0] o_layer_sel;
    wire                         o_busy;
    wire                         o_done;
    
    //==========================================================================
    // Test Variables
    //==========================================================================
    integer i, j, errors;
    integer layer_count;
    integer cycle_count;
    
    // Expected values
    reg [$clog2(NUM_LAYERS)-1:0] expected_layer;
    
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
    postnet_fsm #(
        .NUM_LAYERS    (NUM_LAYERS),
        .SAMPLE_LENGTH (SAMPLE_LENGTH)
    ) dut (
        .clk           (clk),
        .rst_n         (rst_n),
        .i_start       (i_start),
        .i_stack_done  (i_stack_done),
        .o_stack_start (o_stack_start),
        .o_layer_sel   (o_layer_sel),
        .o_busy        (o_busy),
        .o_done        (o_done)
    );
    
    //==========================================================================
    // Stack Done Signal Generation Task
    //==========================================================================
    task simulate_stack_processing;
        input integer cycles;
        begin
            $display("[T=%0t] Stack processing for %0d cycles...", $time, cycles);
            repeat(cycles) @(posedge clk);
            i_stack_done = 1;
            @(posedge clk);
            i_stack_done = 0;
            $display("[T=%0t] Stack done signal asserted", $time);
        end
    endtask
    
    //==========================================================================
    // Layer Monitoring
    //==========================================================================
    always @(posedge clk) begin
        if (o_stack_start) begin
            $display("[T=%0t] Layer %0d started", $time, o_layer_sel);
            
            // Check if layer selection is correct
            if (o_layer_sel != expected_layer) begin
                $display("[ERROR] Expected layer %0d but got %0d", expected_layer, o_layer_sel);
                errors = errors + 1;
            end
            
            expected_layer = expected_layer + 1;
        end
    end
    
    //==========================================================================
    // Test Stimulus
    //==========================================================================
    initial begin
        // Initialize signals
        rst_n = 0;
        i_start = 0;
        i_stack_done = 0;
        errors = 0;
        layer_count = 0;
        cycle_count = 0;
        expected_layer = 0;
        
        // Create waveform dump
        $dumpfile("postnet_fsm_tb.vcd");
        $dumpvars(0, postnet_fsm_tb);
        
        // Reset sequence
        #(CLK_PERIOD*5);
        rst_n = 1;
        #(CLK_PERIOD*2);
        
        $display("========================================");
        $display("PostNet FSM Testbench Started");
        $display("========================================");
        $display("NUM_LAYERS = %0d", NUM_LAYERS);
        $display("========================================");
        
        //----------------------------------------------------------------------
        // Test 1: Basic Layer Sequencing
        //----------------------------------------------------------------------
        $display("\n[TEST 1] Basic Layer Sequencing");
        expected_layer = 0;
        
        // Start FSM
        @(posedge clk);
        i_start = 1;
        @(posedge clk);
        i_start = 0;
        
        // Check that FSM becomes busy (wait for FSM to transition)
        wait(o_busy);
        @(posedge clk);
        if (!o_busy) begin
            $display("[ERROR] FSM should be busy during processing");
            errors = errors + 1;
        end
        else begin
            $display("[INFO] FSM correctly entered busy state");
        end
        
        // Simulate stack processing for each layer
        for (i = 0; i < NUM_LAYERS; i = i + 1) begin
            // Wait for stack_start
            wait(o_stack_start);
            @(posedge clk);
            
            // Simulate stack processing time (variable cycles)
            simulate_stack_processing(50 + i*10);
        end
        
        // Wait for done signal
        wait(o_done);
        @(posedge clk);
        
        if (expected_layer != NUM_LAYERS) begin
            $display("[ERROR] Expected %0d layers but processed %0d", NUM_LAYERS, expected_layer);
            errors = errors + 1;
        end
        else begin
            $display("[PASS] All %0d layers processed correctly", NUM_LAYERS);
        end
        
        // Check FSM returns to idle
        @(posedge clk);
        if (o_busy) begin
            $display("[ERROR] FSM should not be busy after done");
            errors = errors + 1;
        end
        
        #(CLK_PERIOD*10);
        
        //----------------------------------------------------------------------
        // Test 2: Fast Stack Processing (minimal cycles)
        //----------------------------------------------------------------------
        $display("\n[TEST 2] Fast Stack Processing");
        expected_layer = 0;
        
        @(posedge clk);
        i_start = 1;
        @(posedge clk);
        i_start = 0;
        
        // Fast stack processing
        for (i = 0; i < NUM_LAYERS; i = i + 1) begin
            wait(o_stack_start);
            @(posedge clk);
            simulate_stack_processing(5); // Only 5 cycles per layer
        end
        
        wait(o_done);
        @(posedge clk);
        
        $display("[PASS] Fast processing completed");
        
        #(CLK_PERIOD*10);
        
        //----------------------------------------------------------------------
        // Test 3: Slow Stack Processing (many cycles)
        //----------------------------------------------------------------------
        $display("\n[TEST 3] Slow Stack Processing");
        expected_layer = 0;
        
        @(posedge clk);
        i_start = 1;
        @(posedge clk);
        i_start = 0;
        
        // Slow stack processing
        for (i = 0; i < NUM_LAYERS; i = i + 1) begin
            wait(o_stack_start);
            @(posedge clk);
            simulate_stack_processing(200); // Many cycles per layer
        end
        
        wait(o_done);
        @(posedge clk);
        
        $display("[PASS] Slow processing completed");
        
        #(CLK_PERIOD*10);
        
        //----------------------------------------------------------------------
        // Test 4: Back-to-Back Starts
        //----------------------------------------------------------------------
        $display("\n[TEST 4] Back-to-Back Processing Runs");
        
        for (j = 0; j < 3; j = j + 1) begin
            $display("\n  --- Run %0d ---", j+1);
            expected_layer = 0;
            
            @(posedge clk);
            i_start = 1;
            @(posedge clk);
            i_start = 0;
            
            // Process all layers
            for (i = 0; i < NUM_LAYERS; i = i + 1) begin
                wait(o_stack_start);
                @(posedge clk);
                simulate_stack_processing(30);
            end
            
            wait(o_done);
            @(posedge clk);
            
            #(CLK_PERIOD*5); // Small gap between runs
        end
        
        $display("[PASS] Back-to-back runs completed");
        
        #(CLK_PERIOD*10);
        
        //----------------------------------------------------------------------
        // Test 5: Start Signal During Processing (should be ignored)
        //----------------------------------------------------------------------
        $display("\n[TEST 5] Start Signal During Active Processing");
        expected_layer = 0;
        
        @(posedge clk);
        i_start = 1;
        @(posedge clk);
        i_start = 0;
        
        // Process first 2 layers
        for (i = 0; i < 2; i = i + 1) begin
            wait(o_stack_start);
            @(posedge clk);
            simulate_stack_processing(30);
        end
        
        // Try to start again while busy (should be ignored)
        @(posedge clk);
        i_start = 1;
        @(posedge clk);
        i_start = 0;
        $display("[INFO] Sent start signal during processing (should be ignored)");
        
        // Continue processing remaining layers
        for (i = 2; i < NUM_LAYERS; i = i + 1) begin
            wait(o_stack_start);
            @(posedge clk);
            simulate_stack_processing(30);
        end
        
        wait(o_done);
        @(posedge clk);
        
        if (expected_layer == NUM_LAYERS) begin
            $display("[PASS] Spurious start signal correctly ignored");
        end
        else begin
            $display("[ERROR] FSM behavior incorrect with spurious start");
            errors = errors + 1;
        end
        
        #(CLK_PERIOD*10);
        
        //----------------------------------------------------------------------
        // Test 6: Reset During Processing
        //----------------------------------------------------------------------
        $display("\n[TEST 6] Reset During Active Processing");
        expected_layer = 0;
        
        @(posedge clk);
        i_start = 1;
        @(posedge clk);
        i_start = 0;
        
        // Process first layer
        wait(o_stack_start);
        @(posedge clk);
        simulate_stack_processing(30);
        
        // Process second layer partially
        wait(o_stack_start);
        repeat(10) @(posedge clk);
        
        // Assert reset
        $display("[INFO] Asserting reset during layer processing");
        rst_n = 0;
        #(CLK_PERIOD*3);
        rst_n = 1;
        #(CLK_PERIOD*2);
        
        // Check FSM is idle
        if (o_busy) begin
            $display("[ERROR] FSM should be idle after reset");
            errors = errors + 1;
        end
        else begin
            $display("[PASS] FSM correctly reset to idle state");
        end
        
        #(CLK_PERIOD*10);
        
        //----------------------------------------------------------------------
        // Test 7: Layer Selection Verification
        //----------------------------------------------------------------------
        $display("\n[TEST 7] Layer Selection Output Verification");
        expected_layer = 0;
        
        @(posedge clk);
        i_start = 1;
        @(posedge clk);
        i_start = 0;
        
        for (i = 0; i < NUM_LAYERS; i = i + 1) begin
            wait(o_stack_start);
            
            // Verify layer selection matches expected
            if (o_layer_sel != i) begin
                $display("[ERROR] Layer %0d: o_layer_sel=%0d, expected=%0d", 
                         i, o_layer_sel, i);
                errors = errors + 1;
            end
            else begin
                $display("[INFO] Layer %0d: o_layer_sel=%0d - CORRECT", i, o_layer_sel);
            end
            
            @(posedge clk);
            simulate_stack_processing(25);
        end
        
        wait(o_done);
        @(posedge clk);
        
        $display("[PASS] Layer selection verification complete");
        
        #(CLK_PERIOD*10);
        
        //----------------------------------------------------------------------
        // Test Summary
        //----------------------------------------------------------------------
        $display("\n========================================");
        $display("PostNet FSM Testbench Summary");
        $display("========================================");
        $display("Total Test Cases: 7");
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
        #(CLK_PERIOD * 100000); // 1ms timeout
        $display("\n[ERROR] Simulation timeout!");
        $finish;
    end
    
    //==========================================================================
    // Signal Monitoring
    //==========================================================================
    always @(posedge clk) begin
        if (o_done && o_busy) begin
            $display("[ERROR] o_done and o_busy both high at T=%0t", $time);
            errors = errors + 1;
        end
    end

endmodule
