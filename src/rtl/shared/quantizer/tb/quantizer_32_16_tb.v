`timescale 1ns/1ps

//==============================================================================
// Testbench: quantizer_32_16_tb
// Purpose: Validate quantizer_32_16 module functionality
//          Tests: normal quantization, overflow saturation, edge cases
//==============================================================================

module quantizer_32_16_tb;

    // Clock and Reset
    reg clk;
    reg rst_n;
    
    // DUT Signals
    reg i_valid;
    reg signed [31:0] i_acc_raw;
    wire signed [15:0] o_data;
    wire o_valid_out;
    
    // Test Variables
    integer test_num;
    integer pass_count;
    integer fail_count;
    reg signed [15:0] expected;
    
    //==========================================================================
    // DUT Instantiation
    //==========================================================================
    quantizer_32_16 dut (
        .clk(clk),
        .rst_n(rst_n),
        .i_valid(i_valid),
        .i_acc_raw(i_acc_raw),
        .o_data(o_data),
        .o_valid_out(o_valid_out)
    );
    
    //==========================================================================
    // Clock Generation (100 MHz)
    //==========================================================================
    initial begin
        clk = 0;
        forever #5 clk = ~clk; // 10ns period
    end
    
    //==========================================================================
    // Helper Functions
    //==========================================================================
    
    // Convert Q6.26 to float (for display)
    function real q6_26_to_float;
        input signed [31:0] val;
        begin
            q6_26_to_float = $itor(val) / (2.0 ** 26);
        end
    endfunction
    
    // Convert Q4.12 to float (for display)
    function real q4_12_to_float;
        input signed [15:0] val;
        begin
            q4_12_to_float = $itor(val) / (2.0 ** 12);
        end
    endfunction
    
    // Convert float to Q6.26
    function signed [31:0] float_to_q6_26;
        input real val;
        begin
            float_to_q6_26 = $rtoi(val * (2.0 ** 26));
        end
    endfunction
    
    // Convert float to Q4.12
    function signed [15:0] float_to_q4_12;
        input real val;
        begin
            float_to_q4_12 = $rtoi(val * (2.0 ** 12));
        end
    endfunction
    
    //==========================================================================
    // Test Procedure
    //==========================================================================
    task run_test;
        input signed [31:0] input_val;
        input signed [15:0] expected_val;
        input [80*8-1:0] test_name;
        begin
            test_num = test_num + 1;
            
            // Wait for clean clock edge
            @(negedge clk);
            
            // Apply input with setup time before posedge
            i_valid = 1'b1;
            i_acc_raw = input_val;
            expected = expected_val;
            
            // Wait for posedge to register
            @(posedge clk);
            @(negedge clk);
            i_valid = 1'b0;
            
            // Wait for output
            @(posedge clk);
            
            // Check result
            if (o_valid_out && (o_data === expected)) begin
                $display("[PASS] Test %0d: %s", test_num, test_name);
                $display("       Input:    Q6.26 = 0x%08X (%f)", input_val, q6_26_to_float(input_val));
                $display("       Output:   Q4.12 = 0x%04X (%f)", o_data, q4_12_to_float(o_data));
                $display("       Expected: Q4.12 = 0x%04X (%f)", expected, q4_12_to_float(expected));
                pass_count = pass_count + 1;
            end else begin
                $display("[FAIL] Test %0d: %s", test_num, test_name);
                $display("       Input:    Q6.26 = 0x%08X (%f) [bits31:29=%b]", input_val, q6_26_to_float(input_val), input_val[31:29]);
                $display("       Output:   Q4.12 = 0x%04X (%f)", o_data, q4_12_to_float(o_data));
                $display("       Expected: Q4.12 = 0x%04X (%f)", expected, q4_12_to_float(expected));
                $display("       o_valid_out = %b", o_valid_out);
                fail_count = fail_count + 1;
            end
            $display("");
        end
    endtask
    
    //==========================================================================
    // Main Test Sequence
    //==========================================================================
    initial begin
        // Initialize
        test_num = 0;
        pass_count = 0;
        fail_count = 0;
        
        clk = 0;
        rst_n = 0;
        i_valid = 0;
        i_acc_raw = 0;
        
        // Dump waveform
        $dumpfile("quantizer_32_16_tb.vcd");
        $dumpvars(0, quantizer_32_16_tb);
        
        $display("================================================================================");
        $display("Quantizer 32-bit to 16-bit Testbench");
        $display("Input Format:  Q6.26 (32-bit signed, range: -32 to +32)");
        $display("Output Format: Q4.12 (16-bit signed, range: -8 to +7.999)");
        $display("================================================================================");
        $display("");
        
        // Reset
        #20;
        rst_n = 1;
        #10;
        
        //======================================================================
        // Test Group 1: Normal Quantization (Within Range)
        //======================================================================
        $display("--- Test Group 1: Normal Quantization ---");
        
        // Test 1: Zero
        run_test(32'h00000000, 16'h0000, "Zero input");
        
        // Test 2: Small positive (0.5)
        run_test(float_to_q6_26(0.5), float_to_q4_12(0.5), "Small positive (0.5)");
        
        // Test 3: Small negative (-0.5)
        run_test(float_to_q6_26(-0.5), float_to_q4_12(-0.5), "Small negative (-0.5)");
        
        // Test 4: Medium positive (3.75)
        run_test(float_to_q6_26(3.75), float_to_q4_12(3.75), "Medium positive (3.75)");
        
        // Test 5: Medium negative (-3.75)
        run_test(float_to_q6_26(-3.75), float_to_q4_12(-3.75), "Medium negative (-3.75)");
        
        // Test 6: Max valid positive (~7.999)
        run_test(32'h1FFFC000, 16'h7FFF, "Max valid positive (7.999)");
        
        // Test 7: Min valid negative (~-8.000)
        run_test(32'hE0004000, 16'h8001, "Min valid negative (-7.999)");
        
        //======================================================================
        // Test Group 2: Overflow Saturation (Positive)
        //======================================================================
        $display("--- Test Group 2: Positive Overflow Saturation ---");
        
        // Test 8: Slightly over max (+8.5) -> should saturate to +7.999
        run_test(float_to_q6_26(8.5), 16'h7FFF, "Overflow +8.5 -> +7.999 (MAX)");
        
        // Test 9: Large positive (+15.0)
        run_test(float_to_q6_26(15.0), 16'h7FFF, "Overflow +15.0 -> +7.999 (MAX)");
        
        // Test 10: Very large positive (+31.0)
        run_test(float_to_q6_26(31.0), 16'h7FFF, "Overflow +31.0 -> +7.999 (MAX)");
        
        // Test 11: Edge case: bits[31:29] = 001 (overflow positive)
        run_test(32'h20000000, 16'h7FFF, "Edge case 0x20000000 -> MAX");
        
        //======================================================================
        // Test Group 3: Overflow Saturation (Negative)
        //======================================================================
        $display("--- Test Group 3: Negative Overflow Saturation ---");
        
        // Test 12: Slightly under min (-8.5) -> should saturate to -8.0
        run_test(float_to_q6_26(-8.5), 16'h8000, "Overflow -8.5 -> -8.0 (MIN)");
        
        // Test 13: Large negative (-15.0)
        run_test(float_to_q6_26(-15.0), 16'h8000, "Overflow -15.0 -> -8.0 (MIN)");
        
        // Test 14: Very large negative (-31.0)
        run_test(float_to_q6_26(-31.0), 16'h8000, "Overflow -31.0 -> -8.0 (MIN)");
        
        // Test 15: Edge case: bits[31:29] = 110 (overflow negative)
        run_test(32'hC0000000, 16'h8000, "Edge case 0xC0000000 -> MIN");
        
        //======================================================================
        // Test Group 4: Boundary Cases
        //======================================================================
        $display("--- Test Group 4: Boundary Cases ---");
        
        // Test 16: Maximum positive value that fits in Q4.12 without overflow
        // Q4.12 max = +7.999755859375 = 0x7FFF
        // In Q6.26: 0x7FFF << 14 = 0x1FFFC000, but bit[29]=1 makes it negative!
        // We need bits[31:29]=000 and bit[29]=0, so max is 0x1FFFFFFF >> 2 = 0x1FFF_BFFF
        run_test(32'h1FFFBFFF, 16'h7FFE, "Boundary: max positive inside range");
        
        // Test 17: Exactly at negative boundary
        run_test(32'hE0000000, 16'h8000, "Boundary: 0xE0000000 -> 0x8000");
        
        // Test 18: Just inside positive range
        run_test(32'h1FFF0000, 16'h7FFC, "Inside: 0x1FFF0000 -> 0x7FFC");
        
        // Test 19: Just inside negative range
        run_test(32'hE0010000, 16'h8004, "Inside: 0xE0010000 -> 0x8004");
        
        //======================================================================
        // Test Group 5: Quantization Error
        //======================================================================
        $display("--- Test Group 5: Quantization Error (LSB truncation) ---");
        
        // Test 20: Value with LSBs (should be truncated)
        run_test(32'h04001FFF, 16'h1000, "Truncation: 0x04001FFF -> 0x1000");
        
        // Test 21: Value with all LSBs set
        run_test(32'h04003FFF, 16'h1000, "Truncation: 0x04003FFF -> 0x1000");
        
        // Test 22: Value just below rounding threshold
        run_test(32'h04001000, 16'h1000, "Truncation: 0x04001000 -> 0x1000");
        
        //======================================================================
        // Test Group 6: Valid Signal Propagation
        //======================================================================
        $display("--- Test Group 6: Valid Signal Propagation ---");
        
        // Test 23: Valid signal properly propagates
        @(posedge clk);
        i_valid = 1'b1;
        i_acc_raw = float_to_q6_26(1.5);
        
        @(posedge clk);
        // At this cycle, o_valid_out should be high (registered from i_valid)
        if (o_valid_out) begin
            $display("[PASS] Test %0d: Valid signal high after 1 cycle", test_num + 1);
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Test %0d: Valid signal should be high after 1 cycle", test_num + 1);
            fail_count = fail_count + 1;
        end
        test_num = test_num + 1;
        
        i_valid = 1'b0;
        @(posedge clk);
        
        // Now o_valid_out should be low
        if (!o_valid_out) begin
            $display("[PASS] Test %0d: Valid signal low after input deasserted", test_num + 1);
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Test %0d: Valid signal should be low", test_num + 1);
            fail_count = fail_count + 1;
        end
        test_num = test_num + 1;
        $display("");
        
        //======================================================================
        // Test Summary
        //======================================================================
        #100;
        $display("================================================================================");
        $display("Test Summary:");
        $display("  Total Tests: %0d", test_num);
        $display("  Passed:      %0d", pass_count);
        $display("  Failed:      %0d", fail_count);
        
        if (fail_count == 0) begin
            $display("  Result:      ALL TESTS PASSED!");
        end else begin
            $display("  Result:      SOME TESTS FAILED");
        end
        $display("================================================================================");
        
        $finish;
    end
    
    //==========================================================================
    // Timeout Watchdog
    //==========================================================================
    initial begin
        #1000000; // 1ms timeout
        $display("ERROR: Simulation timeout!");
        $finish;
    end

endmodule
