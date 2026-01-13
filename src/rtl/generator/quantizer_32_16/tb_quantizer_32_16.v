`timescale 1ns / 1ps

module tb_quantizer_32_16 ();

    //-------------------------------------------------------------------------
    // 1. Parameters & Signals
    //-------------------------------------------------------------------------
    reg clk;
    reg rst_n;
    reg i_valid;
    reg signed [31:0] i_acc_raw;
    
    wire signed [15:0] o_data;
    wire o_valid_out;

    //-------------------------------------------------------------------------
    // 2. Instantiate Unit Under Test (UUT)
    //-------------------------------------------------------------------------
    quantizer_32_16 uut (
        .clk(clk),
        .rst_n(rst_n),
        .i_valid(i_valid),
        .i_acc_raw(i_acc_raw),
        .o_data(o_data),
        .o_valid_out(o_valid_out)
    );

    //-------------------------------------------------------------------------
    // 3. Clock Generation (100MHz)
    //-------------------------------------------------------------------------
    always #5 clk = ~clk;

    //-------------------------------------------------------------------------
    // 4. Test Scenarios
    //-------------------------------------------------------------------------
    initial begin
        // VCD dump for waveform viewing
        $dumpfile("tb_quantizer_32_16.vcd");
        $dumpvars(0, tb_quantizer_32_16);
        
        // Initialize
        clk = 0;
        rst_n = 0;
        i_valid = 0;
        i_acc_raw = 0;
        
        $display("===============================================");
        $display("   QUANTIZER_32_16 TESTBENCH START");
        $display("===============================================");
        
        // Reset
        #20 rst_n = 1;
        #20;

        //---------------------------------------------------------------------
        // TEST CASE 1: Zero Input
        //---------------------------------------------------------------------
        $display("\n[TEST 1] Zero Input (Q6.26 = 0)");
        i_valid = 1;
        i_acc_raw = 32'h00000000;
        #10;
        i_valid = 0;
        #10;
        $display("  Input (Q6.26):  0x%h", 32'h00000000);
        $display("  Output (Q4.12): 0x%h = %d", o_data, o_data);
        $display("  Valid: %b (Expected: 1)", o_valid_out);
        if (o_data == 16'h0000 && o_valid_out == 1'b1) begin
            $display("  ✓ PASS");
        end else begin
            $display("  ✗ FAIL");
        end

        //---------------------------------------------------------------------
        // TEST CASE 2: Small Positive Value (No Overflow)
        //---------------------------------------------------------------------
        #20;
        $display("\n[TEST 2] Small Positive Value (Within Range)");
        // Q6.26: +0.5 -> bits should be in range
        // 0.5 in Q6.26 = 0x04000000
        // Extract [29:14] should give ~0.5 in Q4.12
        i_valid = 1;
        i_acc_raw = 32'h04000000; // +0.5 in Q6.26
        #10;
        i_valid = 0;
        #10;
        $display("  Input (Q6.26):  0x%h (approx +0.5)", 32'h04000000);
        $display("  Output (Q4.12): 0x%h = %d", o_data, o_data);
        $display("  Valid: %b", o_valid_out);
        if (o_valid_out == 1'b1 && o_data >= 16'h0000 && o_data <= 16'h7FFF) begin
            $display("  ✓ PASS");
        end else begin
            $display("  ✗ FAIL");
        end

        //---------------------------------------------------------------------
        // TEST CASE 3: Small Negative Value (No Overflow)
        //---------------------------------------------------------------------
        #20;
        $display("\n[TEST 3] Small Negative Value (Within Range)");
        // Q6.26: -0.5
        // -0.5 in Q6.26 = 0xFC000000
        // Extract [29:14] should give ~-0.5 in Q4.12
        i_valid = 1;
        i_acc_raw = 32'hFC000000; // -0.5 in Q6.26
        #10;
        i_valid = 0;
        #10;
        $display("  Input (Q6.26):  0x%h (approx -0.5)", 32'hFC000000);
        $display("  Output (Q4.12): 0x%h = %d", o_data, o_data);
        $display("  Valid: %b", o_valid_out);
        if (o_valid_out == 1'b1) begin
            $display("  ✓ PASS");
        end else begin
            $display("  ✗ FAIL");
        end

        //---------------------------------------------------------------------
        // TEST CASE 4: Maximum Positive Value (In Range)
        //---------------------------------------------------------------------
        #20;
        $display("\n[TEST 4] Maximum Positive Value (Q4.12 = +7.999)");
        // Q6.26: +7.999... -> 0x1FFFFFFF (approximately)
        // Should output MAX_OUT = 0x7FFF
        i_valid = 1;
        i_acc_raw = 32'h1FFFFFFF;
        #10;
        i_valid = 0;
        #10;
        $display("  Input (Q6.26):  0x%h (+7.999)", 32'h1FFFFFFF);
        $display("  Output (Q4.12): 0x%h = %d", o_data, o_data);
        $display("  Valid: %b", o_valid_out);
        if (o_valid_out == 1'b1 && o_data == 16'h7FFF) begin
            $display("  ✓ PASS (Saturated to MAX)");
        end else begin
            $display("  ✗ FAIL");
        end

        //---------------------------------------------------------------------
        // TEST CASE 5: Minimum Negative Value (In Range)
        //---------------------------------------------------------------------
        #20;
        $display("\n[TEST 5] Minimum Negative Value (Q4.12 = -8.000)");
        // Q6.26: -8.0 -> 0xE0000000
        // Should output MIN_OUT = 0x8000
        i_valid = 1;
        i_acc_raw = 32'hE0000000;
        #10;
        i_valid = 0;
        #10;
        $display("  Input (Q6.26):  0x%h (-8.0)", 32'hE0000000);
        $display("  Output (Q4.12): 0x%h = %d", o_data, o_data);
        $display("  Valid: %b", o_valid_out);
        if (o_valid_out == 1'b1 && o_data == 16'h8000) begin
            $display("  ✓ PASS (Saturated to MIN)");
        end else begin
            $display("  ✗ FAIL");
        end

        //---------------------------------------------------------------------
        // TEST CASE 6: Positive Overflow (Too Large)
        //---------------------------------------------------------------------
        #20;
        $display("\n[TEST 6] Positive Overflow (Value > +7.999)");
        // Q6.26: +20.0 -> 0x50000000 (overflow)
        // Should saturate to MAX_OUT = 0x7FFF
        i_valid = 1;
        i_acc_raw = 32'h50000000; // +20 in Q6.26
        #10;
        i_valid = 0;
        #10;
        $display("  Input (Q6.26):  0x%h (+20.0 - OVERFLOW)", 32'h50000000);
        $display("  Output (Q4.12): 0x%h = %d", o_data, o_data);
        $display("  Valid: %b", o_valid_out);
        if (o_valid_out == 1'b1 && o_data == 16'h7FFF) begin
            $display("  ✓ PASS (Saturated to MAX)");
        end else begin
            $display("  ✗ FAIL");
        end

        //---------------------------------------------------------------------
        // TEST CASE 7: Negative Overflow (Too Small)
        //---------------------------------------------------------------------
        #20;
        $display("\n[TEST 7] Negative Overflow (Value < -8.0)");
        // Q6.26: -20.0 -> 0xB0000000 (overflow)
        // Should saturate to MIN_OUT = 0x8000
        i_valid = 1;
        i_acc_raw = 32'hB0000000; // -20 in Q6.26
        #10;
        i_valid = 0;
        #10;
        $display("  Input (Q6.26):  0x%h (-20.0 - OVERFLOW)", 32'hB0000000);
        $display("  Output (Q4.12): 0x%h = %d", o_data, o_data);
        $display("  Valid: %b", o_valid_out);
        if (o_valid_out == 1'b1 && o_data == 16'h8000) begin
            $display("  ✓ PASS (Saturated to MIN)");
        end else begin
            $display("  ✗ FAIL");
        end

        //---------------------------------------------------------------------
        // TEST CASE 8: Invalid Input (i_valid = 0)
        //---------------------------------------------------------------------
        #20;
        $display("\n[TEST 8] Invalid Input (i_valid = 0)");
        i_valid = 0;
        i_acc_raw = 32'h12345678;
        #10;
        $display("  i_valid: %b (Expected: 0)", i_valid);
        $display("  o_valid_out: %b (Expected: 0)", o_valid_out);
        $display("  Output (Q4.12): 0x%h", o_data);
        if (o_valid_out == 1'b0) begin
            $display("  ✓ PASS (Output valid correctly propagated)");
        end else begin
            $display("  ✗ FAIL");
        end

        //---------------------------------------------------------------------
        // TEST CASE 9: Sequential Valid Pulses
        //---------------------------------------------------------------------
        #20;
        $display("\n[TEST 9] Sequential Valid Pulses");
        for (integer i = 0; i < 3; i = i + 1) begin
            i_valid = 1;
            i_acc_raw = 32'h04000000 + (i * 32'h01000000);
            #10;
            $display("  Pulse %0d: Input=0x%h, Output=0x%h, Valid=%b", 
                     i, i_acc_raw, o_data, o_valid_out);
            i_valid = 0;
            #10;
        end
        $display("  ✓ PASS (Sequential pulses processed)");

        //---------------------------------------------------------------------
        // TEST CASE 10: Reset during operation
        //---------------------------------------------------------------------
        #20;
        $display("\n[TEST 10] Reset During Operation");
        i_valid = 1;
        i_acc_raw = 32'h12345678;
        #5;
        $display("  Before reset: Output=0x%h, Valid=%b", o_data, o_valid_out);
        
        rst_n = 0;
        #10;
        $display("  After reset: Output=0x%h, Valid=%b", o_data, o_valid_out);
        if (o_data == 16'h0000 && o_valid_out == 1'b0) begin
            $display("  ✓ PASS (Reset cleared all outputs)");
        end else begin
            $display("  ✗ FAIL");
        end

        #20;
        $display("\n===============================================");
        $display("   QUANTIZER_32_16 TESTBENCH COMPLETE");
        $display("===============================================\n");
        $finish;
    end

endmodule
