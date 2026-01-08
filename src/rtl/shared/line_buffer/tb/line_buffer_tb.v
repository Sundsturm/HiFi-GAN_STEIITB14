// =======================================================================
// Testbench: line_buffer_tb
// Purpose: Verify line_buffer functionality with various dilation factors
//
// Test Cases:
//   1. Test with dilation = 1 (standard convolution)
//   2. Test with dilation = 3 (dilated convolution)
//   3. Test with dilation = 9 (highly dilated)
//   4. Test clear functionality
//   5. Test enable control
//   6. Verify sliding window behavior
//
// Simulation Tool: Icarus Verilog (iverilog)
// Waveform: GTKWave compatible VCD file
// =======================================================================

`timescale 1ns / 1ps

module line_buffer_tb;

    // ===================================================================
    // Parameters
    // ===================================================================
    parameter DATA_WIDTH   = 16;
    parameter KERNEL_SIZE  = 3;
    parameter MAX_DILATION = 9;
    parameter BUFFER_DEPTH = 64;
    parameter CLK_PERIOD   = 10;  // 10ns = 100MHz
    
    // ===================================================================
    // Signals
    // ===================================================================
    reg                             clk;
    reg                             rst_n;
    reg                             enable;
    reg  signed [DATA_WIDTH-1:0]    data_in;
    reg  [3:0]                      dilation;
    reg                             clear;
    wire signed [DATA_WIDTH*KERNEL_SIZE-1:0] window_out;  // Flattened output
    wire                            valid;
    
    // Unpacked window for easier access in testbench
    wire signed [DATA_WIDTH-1:0]    window [0:KERNEL_SIZE-1];
    
    // Unpack flattened output
    genvar g;
    generate
        for (g = 0; g < KERNEL_SIZE; g = g + 1) begin : gen_unpack
            assign window[g] = window_out[DATA_WIDTH*(g+1)-1 : DATA_WIDTH*g];
        end
    endgenerate
    
    // Test variables
    integer i, j;
    integer test_num;
    integer errors;
    
    // ===================================================================
    // DUT Instantiation
    // ===================================================================
    line_buffer #(
        .DATA_WIDTH(DATA_WIDTH),
        .KERNEL_SIZE(KERNEL_SIZE),
        .MAX_DILATION(MAX_DILATION),
        .BUFFER_DEPTH(BUFFER_DEPTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        .data_in(data_in),
        .dilation(dilation),
        .clear(clear),
        .window_out(window_out),
        .valid(valid)
    );
    
    // ===================================================================
    // Clock Generation
    // ===================================================================
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end
    
    // ===================================================================
    // VCD Dump for Waveform Viewing
    // ===================================================================
    initial begin
        $dumpfile("line_buffer_tb.vcd");
        $dumpvars(0, line_buffer_tb);
        // Dump buffer contents for debugging
        for (i = 0; i < BUFFER_DEPTH; i = i + 1) begin
            $dumpvars(1, dut.buffer[i]);
        end
        // Dump unpacked window outputs
        for (i = 0; i < KERNEL_SIZE; i = i + 1) begin
            $dumpvars(1, window[i]);
        end
    end
    
    // ===================================================================
    // Test Stimulus
    // ===================================================================
    initial begin
        // Initialize
        $display("=======================================================================");
        $display("Line Buffer Testbench Started");
        $display("=======================================================================");
        errors = 0;
        test_num = 0;
        
        rst_n = 0;
        enable = 0;
        data_in = 0;
        dilation = 1;
        clear = 0;
        
        // Reset sequence
        #(CLK_PERIOD*2);
        rst_n = 1;
        #(CLK_PERIOD*2);
        
        // ---------------------------------------------------------------
        // Test 1: Dilation = 1 (Standard Convolution)
        // ---------------------------------------------------------------
        test_num = test_num + 1;
        $display("\n--- Test %0d: Dilation = 1 (Standard Convolution) ---", test_num);
        dilation = 1;
        enable = 1;
        
        // Feed sequential data: 1, 2, 3, 4, 5, ...
        for (i = 1; i <= 10; i = i + 1) begin
            data_in = i;
            #CLK_PERIOD;
            
            // After KERNEL_SIZE samples, window should be valid
            if (i >= KERNEL_SIZE) begin
                $display("Cycle %0d: Input=%0d, Valid=%0b, Window=[%0d, %0d, %0d]", 
                         i, data_in, valid, 
                         window[0], window[1], window[2]);
                
                // Verify window contents (most recent to oldest)
                if (valid) begin
                    if (window[0] != i) begin
                        $display("ERROR: window[0] = %0d, expected %0d", window[0], i);
                        errors = errors + 1;
                    end
                    if (window[1] != i-1) begin
                        $display("ERROR: window[1] = %0d, expected %0d", window[1], i-1);
                        errors = errors + 1;
                    end
                    if (window[2] != i-2) begin
                        $display("ERROR: window[2] = %0d, expected %0d", window[2], i-2);
                        errors = errors + 1;
                    end
                end
            end
        end
        
        // ---------------------------------------------------------------
        // Test 2: Clear Buffer
        // ---------------------------------------------------------------
        test_num = test_num + 1;
        $display("\n--- Test %0d: Clear Buffer ---", test_num);
        clear = 1;
        #CLK_PERIOD;
        clear = 0;
        #CLK_PERIOD;
        
        if (!valid) begin
            $display("PASS: Buffer cleared, valid = 0");
        end else begin
            $display("ERROR: Buffer not cleared properly");
            errors = errors + 1;
        end
        
        // ---------------------------------------------------------------
        // Test 3: Dilation = 3
        // ---------------------------------------------------------------
        test_num = test_num + 1;
        $display("\n--- Test %0d: Dilation = 3 ---", test_num);
        dilation = 3;
        enable = 1;
        
        // Feed data: 10, 20, 30, 40, ...
        for (i = 1; i <= 20; i = i + 1) begin
            data_in = i * 10;
            #CLK_PERIOD;
            
            // Need (KERNEL_SIZE-1) * dilation + 1 = 2*3+1 = 7 samples for valid
            if (i >= 7) begin
                $display("Cycle %0d: Input=%0d, Valid=%0b, Window=[%0d, %0d, %0d]", 
                         i, data_in, valid,
                         window[0], window[1], window[2]);
                
                // With dilation=3, window should contain samples spaced 3 apart
                if (valid) begin
                    // Most recent, 3 samples back, 6 samples back
                    if (window[0] != i * 10) begin
                        $display("ERROR: window[0] = %0d, expected %0d", window[0], i*10);
                        errors = errors + 1;
                    end
                    if (window[1] != (i-3) * 10) begin
                        $display("ERROR: window[1] = %0d, expected %0d", window[1], (i-3)*10);
                        errors = errors + 1;
                    end
                    if (window[2] != (i-6) * 10) begin
                        $display("ERROR: window[2] = %0d, expected %0d", window[2], (i-6)*10);
                        errors = errors + 1;
                    end
                end
            end
        end
        
        // ---------------------------------------------------------------
        // Test 4: Dilation = 9 (Maximum)
        // ---------------------------------------------------------------
        test_num = test_num + 1;
        $display("\n--- Test %0d: Dilation = 9 (Maximum) ---", test_num);
        clear = 1;
        #CLK_PERIOD;
        clear = 0;
        dilation = 9;
        enable = 1;
        
        // Feed data: 100, 200, 300, ...
        for (i = 1; i <= 25; i = i + 1) begin
            data_in = i * 100;
            #CLK_PERIOD;
            
            // Need (KERNEL_SIZE-1) * dilation + 1 = 2*9+1 = 19 samples for valid
            if (i >= 19) begin
                $display("Cycle %0d: Input=%0d, Valid=%0b, Window=[%0d, %0d, %0d]", 
                         i, data_in, valid,
                         window[0], window[1], window[2]);
                
                if (valid) begin
                    // Most recent, 9 samples back, 18 samples back
                    if (window[0] != i * 100) begin
                        $display("ERROR: window[0] = %0d, expected %0d", window[0], i*100);
                        errors = errors + 1;
                    end
                    if (window[1] != (i-9) * 100) begin
                        $display("ERROR: window[1] = %0d, expected %0d", window[1], (i-9)*100);
                        errors = errors + 1;
                    end
                    if (window[2] != (i-18) * 100) begin
                        $display("ERROR: window[2] = %0d, expected %0d", window[2], (i-18)*100);
                        errors = errors + 1;
                    end
                end
            end
        end
        
        // ---------------------------------------------------------------
        // Test 5: Enable Control (Pause and Resume)
        // ---------------------------------------------------------------
        test_num = test_num + 1;
        $display("\n--- Test %0d: Enable Control ---", test_num);
        clear = 1;
        #CLK_PERIOD;
        clear = 0;
        dilation = 1;
        enable = 1;
        
        // Fill buffer
        for (i = 1; i <= 5; i = i + 1) begin
            data_in = i;
            #CLK_PERIOD;
        end
        
        // Pause (disable)
        $display("Pausing (enable=0)...");
        enable = 0;
        data_in = 999;  // This should not be written
        #(CLK_PERIOD*3);
        
        // Verify data didn't change
        if (window[0] == 999) begin
            $display("ERROR: Data was written while enable=0");
            errors = errors + 1;
        end else begin
            $display("PASS: Buffer did not change while disabled");
        end
        
        // Resume
        $display("Resuming (enable=1)...");
        enable = 1;
        data_in = 6;
        #CLK_PERIOD;
        
        if (window[0] == 6) begin
            $display("PASS: Buffer resumed correctly");
        end else begin
            $display("ERROR: Buffer did not resume correctly");
            errors = errors + 1;
        end
        
        // ---------------------------------------------------------------
        // Test 6: Circular Buffer Wrap-around
        // ---------------------------------------------------------------
        test_num = test_num + 1;
        $display("\n--- Test %0d: Circular Buffer Wrap-around ---", test_num);
        clear = 1;
        #CLK_PERIOD;
        clear = 0;
        dilation = 1;
        enable = 1;
        
        // Fill more than buffer depth to test wrap-around
        $display("Filling buffer beyond depth...");
        for (i = 1; i <= BUFFER_DEPTH + 10; i = i + 1) begin
            data_in = i;
            #CLK_PERIOD;
        end
        
        $display("Final window after wrap: [%0d, %0d, %0d]", 
                 window[0], window[1], window[2]);
        
        // Should contain most recent 3 values
        if (window[0] == BUFFER_DEPTH + 10 &&
            window[1] == BUFFER_DEPTH + 9 &&
            window[2] == BUFFER_DEPTH + 8) begin
            $display("PASS: Circular buffer wrap-around works correctly");
        end else begin
            $display("ERROR: Circular buffer wrap-around failed");
            errors = errors + 1;
        end
        
        // ---------------------------------------------------------------
        // Test Summary
        // ---------------------------------------------------------------
        #(CLK_PERIOD*5);
        $display("\n=======================================================================");
        $display("Test Summary:");
        $display("  Total Tests: %0d", test_num);
        $display("  Total Errors: %0d", errors);
        if (errors == 0) begin
            $display("  Status: ALL TESTS PASSED!");
        end else begin
            $display("  Status: TESTS FAILED!");
        end
        $display("=======================================================================");
        
        $finish;
    end
    
    // ===================================================================
    // Timeout Watchdog
    // ===================================================================
    initial begin
        #(CLK_PERIOD * 10000);
        $display("ERROR: Testbench timeout!");
        $finish;
    end

endmodule
