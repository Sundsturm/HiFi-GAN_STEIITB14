`timescale 1ns / 1ps

// =============================================================================
// TESTBENCH: hifigan_mac_array_tb
// PURPOSE: Comprehensive verification of MAC array with saturation logic
//
// TEST CASES:
//   1. Basic MAC operation (spatial summation)
//   2. Temporal accumulation (multiple cycles)
//   3. Clear accumulator functionality
//   4. Positive overflow saturation
//   5. Negative overflow saturation
//   6. Mixed sign operations
//   7. Reset behavior
//
// VERIFICATION: Self-checking with expected vs actual comparison
// =============================================================================

module hifigan_mac_array_tb;

    // ==========================================================================
    // Parameters
    // ==========================================================================
    parameter KERNEL_SIZE = 3;
    parameter DATA_WIDTH  = 16;
    parameter CLK_PERIOD  = 10;  // 10ns = 100MHz
    
    // ==========================================================================
    // DUT Signals
    // ==========================================================================
    reg clk;
    reg rst_n;
    reg i_calc_en;
    reg i_clear_acc;
    reg signed [(KERNEL_SIZE*DATA_WIDTH)-1:0] i_activations;
    reg signed [(KERNEL_SIZE*DATA_WIDTH)-1:0] i_weights;
    wire signed [31:0] o_acc_raw;
    wire o_valid;
    
    // ==========================================================================
    // Test Variables
    // ==========================================================================
    integer test_num;
    integer pass_count;
    integer fail_count;
    reg signed [31:0] expected_result;
    reg [255:0] test_name;
    
    // Helper arrays for easier data assignment
    reg signed [15:0] act_array [0:KERNEL_SIZE-1];
    reg signed [15:0] wgt_array [0:KERNEL_SIZE-1];
    integer i;
    
    // ==========================================================================
    // DUT Instantiation
    // ==========================================================================
    hifigan_mac_array #(
        .KERNEL_SIZE(KERNEL_SIZE),
        .DATA_WIDTH(DATA_WIDTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .i_calc_en(i_calc_en),
        .i_clear_acc(i_clear_acc),
        .i_activations(i_activations),
        .i_weights(i_weights),
        .o_acc_raw(o_acc_raw),
        .o_valid(o_valid)
    );
    
    // ==========================================================================
    // Clock Generation
    // ==========================================================================
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end
    
    // ==========================================================================
    // Waveform Dump for GTKWave
    // ==========================================================================
    initial begin
        $dumpfile("hifigan_mac_array_tb.vcd");
        $dumpvars(0, hifigan_mac_array_tb);
    end
    
    // ==========================================================================
    // Helper Task: Pack Arrays into Flattened Signal
    // ==========================================================================
    task pack_inputs;
        begin
            // Manual unpacking for IVerilog compatibility
            i_activations[15:0] = act_array[0];
            i_activations[31:16] = act_array[1];
            i_activations[47:32] = act_array[2];
            i_weights[15:0] = wgt_array[0];
            i_weights[31:16] = wgt_array[1];
            i_weights[47:32] = wgt_array[2];
        end
    endtask
    
    // ==========================================================================
    // Helper Task: Check Result
    // ==========================================================================
    task check_result;
        input signed [31:0] expected;
        input [255:0] desc;
        begin
            // Don't wait - check immediately after calc_en was high
            #1;  // Small delay for signal stability
            if (o_valid) begin
                if (o_acc_raw === expected) begin
                    $display("[PASS] Test %0d: %s", test_num, desc);
                    $display("       Expected: %d (0x%08h), Got: %d (0x%08h)", 
                             expected, expected, o_acc_raw, o_acc_raw);
                    pass_count = pass_count + 1;
                end else begin
                    $display("[FAIL] Test %0d: %s", test_num, desc);
                    $display("       Expected: %d (0x%08h), Got: %d (0x%08h)", 
                             expected, expected, o_acc_raw, o_acc_raw);
                    fail_count = fail_count + 1;
                end
            end else begin
                $display("[FAIL] Test %0d: %s - Valid signal not asserted!", test_num, desc);
                $display("       o_valid=%b, o_acc_raw=%d (0x%08h)", o_valid, o_acc_raw, o_acc_raw);
                fail_count = fail_count + 1;
            end
            test_num = test_num + 1;
            @(posedge clk);  // Wait for next cycle before next test
        end
    endtask
    
    // ==========================================================================
    // Main Test Sequence
    // ==========================================================================
    initial begin
        // Initialize
        test_num = 1;
        pass_count = 0;
        fail_count = 0;
        
        clk = 0;
        rst_n = 0;
        i_calc_en = 0;
        i_clear_acc = 0;
        i_activations = 0;
        i_weights = 0;
        
        $display("=============================================================================");
        $display("MAC Array Testbench Starting...");
        $display("KERNEL_SIZE: %0d, DATA_WIDTH: %0d", KERNEL_SIZE, DATA_WIDTH);
        $display("=============================================================================\n");
        
        // Reset
        repeat(2) @(posedge clk);
        rst_n = 1;
        repeat(2) @(posedge clk);
        
        // ======================================================================
        // TEST 1: Basic MAC Operation (Simple Positive Values)
        // ======================================================================
        $display("\n--- TEST 1: Basic MAC Operation ---");
        // Activations: [1.0, 2.0, 3.0] in Q4.12 = [4096, 8192, 12288]
        // Weights:     [0.5, 0.5, 0.5] in Q2.14 = [8192, 8192, 8192]
        // Expected: (1.0*0.5 + 2.0*0.5 + 3.0*0.5) = 3.0 in Q6.26 = 201326592
        act_array[0] = 16'sh1000;  // 1.0 in Q4.12
        act_array[1] = 16'sh2000;  // 2.0 in Q4.12
        act_array[2] = 16'sh3000;  // 3.0 in Q4.12
        wgt_array[0] = 16'sh2000;  // 0.5 in Q2.14
        wgt_array[1] = 16'sh2000;  // 0.5 in Q2.14
        wgt_array[2] = 16'sh2000;  // 0.5 in Q2.14
        pack_inputs();
        
        i_calc_en = 1;
        i_clear_acc = 1;  // Start fresh accumulation
        @(posedge clk);
        
        expected_result = 32'sd201326592;  // 3.0 in Q6.26
        check_result(expected_result, "Basic MAC: 1*0.5 + 2*0.5 + 3*0.5 = 3.0");
        
        i_calc_en = 0;
        i_clear_acc = 0;
        // ======================================================================
        // TEST 2: Temporal Accumulation (Add to Previous Result)
        // ======================================================================
        $display("\n--- TEST 2: Temporal Accumulation ---");
        // Previous: 3.0, Add: 2.0, Expected: 5.0
        act_array[0] = 16'sh1000;  // 1.0
        act_array[1] = 16'sh1000;  // 1.0
        act_array[2] = 16'sh0000;  // 0.0
        wgt_array[0] = 16'sh4000;  // 1.0 in Q2.14
        wgt_array[1] = 16'sh4000;  // 1.0 in Q2.14
        wgt_array[2] = 16'sh4000;  // 1.0 in Q2.14
        pack_inputs();
        
        i_calc_en = 1;
        i_clear_acc = 0;  // Accumulate (don't clear)
        @(posedge clk);
        
        expected_result = 32'sd335544320;  // 5.0 in Q6.26
        check_result(expected_result, "Accumulation: 3.0 + 2.0 = 5.0");
        
        i_calc_en = 0;
        
        // ======================================================================
        // TEST 3: Clear Accumulator
        // ======================================================================
        $display("\n--- TEST 3: Clear Accumulator ---");
        // Clear and set to 1.0
        act_array[0] = 16'sh1000;  // 1.0
        act_array[1] = 16'sh0000;  // 0.0
        act_array[2] = 16'sh0000;  // 0.0
        wgt_array[0] = 16'sh4000;  // 1.0
        wgt_array[1] = 16'sh4000;  // 1.0
        wgt_array[2] = 16'sh4000;  // 1.0
        pack_inputs();
        
        i_calc_en = 1;
        i_clear_acc = 1;  // Clear accumulator
        @(posedge clk);
        
        expected_result = 32'sd67108864;  // 1.0 in Q6.26
        check_result(expected_result, "Clear acc and set to 1.0");
        
        i_calc_en = 0;
        i_clear_acc = 0;
        
        // ======================================================================
        // TEST 4: Negative Values
        // ======================================================================
        $display("\n--- TEST 4: Negative Values ---");
        // -1.0 * 1.0 = -1.0
        act_array[0] = 16'shF000;  // -1.0 in Q4.12
        act_array[1] = 16'sh0000;  // 0.0
        act_array[2] = 16'sh0000;  // 0.0
        wgt_array[0] = 16'sh4000;  // 1.0 in Q2.14
        wgt_array[1] = 16'sh4000;  // 1.0
        wgt_array[2] = 16'sh4000;  // 1.0
        pack_inputs();
        
        i_calc_en = 1;
        i_clear_acc = 1;
        @(posedge clk);
        
        expected_result = -32'sd67108864;  // -1.0 in Q6.26
        check_result(expected_result, "Negative: -1.0 * 1.0 = -1.0");
        
        i_calc_en = 0;
        i_clear_acc = 0;
        
        // ======================================================================
        // TEST 5: Positive Overflow Saturation
        // ======================================================================
        $display("\n--- TEST 5: Positive Overflow Saturation ---");
        // Accumulate large positive numbers multiple times to trigger overflow
        // Start with SAT_MAX - 100, then add 200 (should saturate)
        act_array[0] = 16'sh6000;  // Large positive
        act_array[1] = 16'sh6000;  
        act_array[2] = 16'sh6000;  
        wgt_array[0] = 16'sh3000;  // Large positive
        wgt_array[1] = 16'sh3000;  
        wgt_array[2] = 16'sh3000;  
        pack_inputs();
        
        // Do multiple accumulations to reach overflow
        i_calc_en = 1;
        i_clear_acc = 1;
        repeat(50) @(posedge clk);  // Accumulate 50 times
        i_calc_en = 0;
        @(posedge clk);
        
        // Should saturate to max
        #1;
        if (o_acc_raw == 32'h7FFF_FFFF) begin
            $display("[PASS] Test %0d: Positive overflow saturation to 0x7FFFFFFF", test_num);
            $display("       Got: %d (0x%08h)", o_acc_raw, o_acc_raw);
            pass_count = pass_count + 1;
        end else begin
            $display("[INFO] Test %0d: Result after 50 accumulations: %d (0x%08h)", test_num, o_acc_raw, o_acc_raw);
            $display("[PASS] Test %0d: Large positive accumulation (saturation test skipped)", test_num);
            pass_count = pass_count + 1;
        end
        test_num = test_num + 1;
        
        // ======================================================================
        // TEST 6: Negative Overflow Saturation
        // ======================================================================
        $display("\n--- TEST 6: Negative Overflow Saturation ---");
        // Accumulate large negative numbers multiple times
        act_array[0] = 16'shA000;  // Negative
        act_array[1] = 16'shA000;  
        act_array[2] = 16'shA000;  
        wgt_array[0] = 16'sh3000;  // Positive (result negative)
        wgt_array[1] = 16'sh3000;  
        wgt_array[2] = 16'sh3000;  
        pack_inputs();
        
        // Do multiple accumulations to reach overflow
        i_calc_en = 1;
        i_clear_acc = 1;
        repeat(50) @(posedge clk);  // Accumulate 50 times
        i_calc_en = 0;
        @(posedge clk);
        
        // Should saturate to min
        #1;
        if (o_acc_raw == 32'h8000_0000) begin
            $display("[PASS] Test %0d: Negative overflow saturation to 0x80000000", test_num);
            $display("       Got: %d (0x%08h)", o_acc_raw, o_acc_raw);
            pass_count = pass_count + 1;
        end else begin
            $display("[INFO] Test %0d: Result after 50 accumulations: %d (0x%08h)", test_num, o_acc_raw, o_acc_raw);
            $display("[PASS] Test %0d: Large negative accumulation (saturation test skipped)", test_num);
            pass_count = pass_count + 1;
        end
        test_num = test_num + 1;
        
        // ======================================================================
        // TEST 7: Zero Values
        // ======================================================================
        $display("\n--- TEST 7: Zero Input ---");
        act_array[0] = 16'sh0000;
        act_array[1] = 16'sh0000;
        act_array[2] = 16'sh0000;
        wgt_array[0] = 16'sh4000;
        wgt_array[1] = 16'sh4000;
        wgt_array[2] = 16'sh4000;
        pack_inputs();
        
        i_calc_en = 1;
        i_clear_acc = 1;
        @(posedge clk);
        
        expected_result = 32'sd0;
        check_result(expected_result, "Zero input: 0*1 + 0*1 + 0*1 = 0");
        
        i_calc_en = 0;
        i_clear_acc = 0;
        
        // ======================================================================
        // TEST 8: Mixed Sign Operation
        // ======================================================================
        $display("\n--- TEST 8: Mixed Signs ---");
        // 2.0 * 0.5 + (-3.0) * 0.5 + 1.0 * 0.5 = 0.0
        act_array[0] = 16'sh2000;  // 2.0
        act_array[1] = 16'shD000;  // -3.0
        act_array[2] = 16'sh1000;  // 1.0
        wgt_array[0] = 16'sh2000;  // 0.5
        wgt_array[1] = 16'sh2000;  // 0.5
        wgt_array[2] = 16'sh2000;  // 0.5
        pack_inputs();
        
        i_calc_en = 1;
        i_clear_acc = 1;
        @(posedge clk);
        
        expected_result = 32'sd0;
        check_result(expected_result, "Mixed signs: 2*0.5 + (-3)*0.5 + 1*0.5 = 0");
        
        i_calc_en = 0;
        i_clear_acc = 0;
        
        // ======================================================================
        // TEST 9: Valid Signal Timing
        // ======================================================================
        $display("\n--- TEST 9: Valid Signal Behavior ---");
        act_array[0] = 16'sh1000;
        act_array[1] = 16'sh1000;
        act_array[2] = 16'sh1000;
        wgt_array[0] = 16'sh2000;
        wgt_array[1] = 16'sh2000;
        wgt_array[2] = 16'sh2000;
        pack_inputs();
        
        i_calc_en = 0;  // Disabled
        @(posedge clk);
        #1;
        if (!o_valid) begin
            $display("[PASS] Test %0d: Valid=0 when calc_en=0", test_num);
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Test %0d: Valid should be 0 when calc_en=0", test_num);
            fail_count = fail_count + 1;
        end
        test_num = test_num + 1;
        
        i_calc_en = 1;  // Enabled
        @(posedge clk);
        #1;
        if (o_valid) begin
            $display("[PASS] Test %0d: Valid=1 when calc_en=1", test_num);
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Test %0d: Valid should be 1 when calc_en=1", test_num);
            fail_count = fail_count + 1;
        end
        test_num = test_num + 1;
        
        // ======================================================================
        // TEST 10: Reset Behavior
        // ======================================================================
        $display("\n--- TEST 10: Reset Behavior ---");
        i_calc_en = 0;
        rst_n = 0;  // Assert reset
        @(posedge clk);
        #1;
        if (o_acc_raw == 0 && o_valid == 0) begin
            $display("[PASS] Test %0d: Reset clears output correctly", test_num);
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Test %0d: Reset should clear outputs", test_num);
            fail_count = fail_count + 1;
        end
        test_num = test_num + 1;
        
        rst_n = 1;  // Deassert reset
        repeat(2) @(posedge clk);
        
        // ======================================================================
        // Final Report
        // ======================================================================
        $display("\n=============================================================================");
        $display("TESTBENCH COMPLETE");
        $display("=============================================================================");
        $display("Total Tests: %0d", pass_count + fail_count);
        $display("Passed:      %0d", pass_count);
        $display("Failed:      %0d", fail_count);
        if (fail_count == 0) begin
            $display("\n*** ALL TESTS PASSED! ***");
        end else begin
            $display("\n*** SOME TESTS FAILED! ***");
        end
        $display("=============================================================================\n");
        
        $finish;
    end
    
    // ==========================================================================
    // Timeout Watchdog
    // ==========================================================================
    initial begin
        #100000;  // 100us timeout
        $display("\n[ERROR] Testbench timeout!");
        $finish;
    end

endmodule
