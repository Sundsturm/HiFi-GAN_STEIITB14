// =======================================================================
// Testbench: conv1d_engine_bram_tb
// Purpose: Verify BRAM-based Conv1D engine for Zynq integration
// =======================================================================

`timescale 1ns/1ps

module conv1d_engine_bram_tb;

    // ===================================================================
    // Parameters
    // ===================================================================
    parameter DATA_WIDTH = 16;
    parameter KERNEL_SIZE = 7;
    parameter CLK_PERIOD = 10;  // 100 MHz
    
    // ===================================================================
    // Signals
    // ===================================================================
    reg clk;
    reg rst_n;
    
    // Control
    reg start;
    wire done;
    wire busy;
    
    // Configuration
    reg [15:0] seq_length;
    reg [9:0]  in_channels;
    reg [9:0]  out_channels;
    reg [3:0]  kernel_size;
    reg [3:0]  dilation;
    
    // Input BRAM
    wire [15:0] input_addr;
    wire input_rd_en;
    reg signed [DATA_WIDTH-1:0] input_data;
    reg signed [DATA_WIDTH-1:0] input_bram [0:16383];  // 16K entries
    
    // Output BRAM
    wire [15:0] output_addr;
    wire output_wr_en;
    wire signed [DATA_WIDTH-1:0] output_data;
    reg signed [DATA_WIDTH-1:0] output_bram [0:16383];
    
    // Weight memory
    wire [20:0] weight_addr;
    reg signed [DATA_WIDTH-1:0] weight_data;
    reg signed [DATA_WIDTH-1:0] weight_mem [0:20479];  // 20K entries
    
    // Bias memory
    wire [10:0] bias_addr;
    reg signed [31:0] bias_data;
    reg signed [31:0] bias_mem [0:1023];
    
    // Test control
    integer errors;
    integer cycle_count;
    reg signed [31:0] expected_output [0:255][0:15];  // [time][channel]
    
    // ===================================================================
    // Clock Generation
    // ===================================================================
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end
    
    // ===================================================================
    // DUT Instantiation
    // ===================================================================
    conv1d_engine_bram #(
        .DATA_WIDTH(DATA_WIDTH),
        .KERNEL_SIZE(KERNEL_SIZE),
        .MAX_IN_CH(256),
        .MAX_OUT_CH(512),
        .MAX_SEQ_LEN(256),
        .ACTIVATION("NONE")
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .done(done),
        .busy(busy),
        .seq_length(seq_length),
        .in_channels(in_channels),
        .out_channels(out_channels),
        .kernel_size(kernel_size),
        .dilation(dilation),
        .input_addr(input_addr),
        .input_rd_en(input_rd_en),
        .input_data(input_data),
        .output_addr(output_addr),
        .output_wr_en(output_wr_en),
        .output_data(output_data),
        .weight_addr(weight_addr),
        .weight_data(weight_data),
        .bias_addr(bias_addr),
        .bias_data(bias_data)
    );
    
    // ===================================================================
    // BRAM Read Logic
    // ===================================================================
    always @(posedge clk) begin
        if (input_rd_en)
            input_data <= input_bram[input_addr];
        
        weight_data <= weight_mem[weight_addr];
        bias_data <= bias_mem[bias_addr];
    end
    
    // ===================================================================
    // BRAM Write Logic
    // ===================================================================
    always @(posedge clk) begin
        if (output_wr_en) begin
            output_bram[output_addr] <= output_data;
            $display("[T=%0t] Output[addr=%0d] = %h (Q4.12 = %f)", 
                     $time, output_addr, output_data, 
                     $itor(output_data) / 4096.0);
        end
    end
    
    // ===================================================================
    // Test Stimulus
    // ===================================================================
    initial begin
        // Initialize
        rst_n = 0;
        start = 0;
        errors = 0;
        cycle_count = 0;
        
        $display("\n========================================");
        $display("Conv1D BRAM Engine Testbench");
        $display("========================================\n");
        
        // Reset
        repeat(5) @(posedge clk);
        rst_n = 1;
        repeat(2) @(posedge clk);
        
        // Run tests
        run_test(3, 2, 3, 8, 1);   // Test 1: K=3, IN=2, OUT=3, SEQ=8
        run_test(5, 4, 8, 10, 1);  // Test 2: K=5, IN=4, OUT=8, SEQ=10
        run_test(3, 1, 1, 6, 1);   // Test 3: Single channel
        run_test(3, 4, 4, 8, 2);   // Test 4: With dilation
        
        // Summary
        $display("\n========================================");
        $display("TEST SUMMARY");
        $display("========================================");
        if (errors == 0) begin
            $display("ALL TESTS PASSED!");
        end else begin
            $display("TESTS FAILED: %0d errors", errors);
        end
        $display("========================================\n");
        
        $finish;
    end
    
    // ===================================================================
    // Task: Run Test
    // ===================================================================
    task run_test;
        input integer k_size;
        input integer in_ch;
        input integer out_ch;
        input integer seq_len;
        input integer dil;
        
        integer timeout;
        
        begin
            $display("\n========================================");
            $display("TEST: K=%0d, IN=%0d, OUT=%0d, SEQ=%0d, DIL=%0d", 
                     k_size, in_ch, out_ch, seq_len, dil);
            $display("========================================");
            
            // Configure
            kernel_size = k_size;
            in_channels = in_ch;
            out_channels = out_ch;
            seq_length = seq_len;
            dilation = dil;
            
            // Load test data
            load_weights(k_size, in_ch, out_ch);
            load_input(seq_len, in_ch);
            compute_expected(k_size, in_ch, out_ch, seq_len, dil);
            
            // Start processing
            cycle_count = 0;
            @(posedge clk);
            start = 1;
            @(posedge clk);
            start = 0;
            
            // Wait for completion
            timeout = 50000;
            while (!done && timeout > 0) begin
                @(posedge clk);
                timeout = timeout - 1;
                cycle_count = cycle_count + 1;
            end
            
            if (timeout == 0) begin
                $display("[ERROR] Test timeout after 50000 cycles!");
                errors = errors + 1;
            end else begin
                $display("[PASS] Completed in %0d cycles", cycle_count);
            end
            
            // Verify outputs
            check_outputs(out_ch, seq_len);
            
            repeat(10) @(posedge clk);
        end
    endtask
    
    // ===================================================================
    // Task: Load Weights
    // ===================================================================
    task load_weights;
        input integer k_size;
        input integer in_ch;
        input integer out_ch;
        
        integer o, i, k, addr;
        real weight_val;
        
        begin
            $display("[Load Weights] K=%0d, IN_CH=%0d, OUT_CH=%0d", k_size, in_ch, out_ch);
            
            addr = 0;
            for (o = 0; o < out_ch; o = o + 1) begin
                for (i = 0; i < in_ch; i = i + 1) begin
                    for (k = 0; k < k_size; k = k + 1) begin
                        // Weight pattern: (out_ch+1) * 0.1 in Q2.14
                        weight_val = (o + 1) * 0.1;
                        weight_mem[addr] = $rtoi(weight_val * 16384.0);
                        addr = addr + 1;
                    end
                end
            end
            
            // Load biases: out_ch * 0.5 in Q6.26
            for (o = 0; o < out_ch; o = o + 1) begin
                bias_mem[o] = $rtoi(o * 0.5 * 67108864.0);
            end
            
            $display("[Load Weights] Loaded %0d weights and %0d biases", addr, out_ch);
        end
    endtask
    
    // ===================================================================
    // Task: Load Input
    // ===================================================================
    task load_input;
        input integer seq_len;
        input integer in_ch;
        
        integer t, c, addr;
        real input_val;
        
        begin
            $display("[Load Input] SEQ=%0d, IN_CH=%0d", seq_len, in_ch);
            
            for (t = 0; t < seq_len; t = t + 1) begin
                for (c = 0; c < in_ch; c = c + 1) begin
                    // Input pattern: ramp 0.0, 0.1, 0.2, ... in Q4.12
                    addr = t * in_ch + c;
                    input_val = t * 0.1;
                    input_bram[addr] = $rtoi(input_val * 4096.0);
                end
            end
            
            $display("[Load Input] Loaded %0d samples", seq_len * in_ch);
        end
    endtask
    
    // ===================================================================
    // Task: Compute Expected Output (Golden Model)
    // ===================================================================
    task compute_expected;
        input integer k_size;
        input integer in_ch;
        input integer out_ch;
        input integer seq_len;
        input integer dil;
        
        integer t_out, o, i, k_idx, t_in;
        reg signed [31:0] accumulator;
        reg signed [31:0] weight_val;
        reg signed [31:0] input_val;
        reg signed [31:0] product;
        
        begin
            $display("[Compute Expected] K=%0d, IN=%0d, OUT=%0d, SEQ=%0d, DIL=%0d", 
                     k_size, in_ch, out_ch, seq_len, dil);
            
            for (t_out = 0; t_out < seq_len; t_out = t_out + 1) begin
                for (o = 0; o < out_ch; o = o + 1) begin
                    accumulator = 0;
                    
                    for (i = 0; i < in_ch; i = i + 1) begin
                        for (k_idx = 0; k_idx < k_size; k_idx = k_idx + 1) begin
                            t_in = t_out + k_idx * dil;
                            
                            if (t_in < seq_len) begin
                                // Get weight (Q2.14)
                                weight_val = $signed(weight_mem[(o * in_ch * k_size) + 
                                                               (i * k_size) + k_idx]);
                                
                                // Get input (Q4.12) - reuse same value for all channels
                                input_val = $signed(input_bram[t_in * in_ch + i]);
                                
                                // Multiply: Q4.12 * Q2.14 = Q6.26
                                product = (input_val * weight_val);
                                accumulator = accumulator + product;
                            end
                        end
                    end
                    
                    // Add bias (Q6.26)
                    accumulator = accumulator + $signed(bias_mem[o]);
                    
                    // Quantize to Q4.12
                    expected_output[t_out][o] = (accumulator + 32768) >>> 16;  // Round
                end
            end
            
            $display("[Compute Expected] Done");
        end
    endtask
    
    // ===================================================================
    // Task: Check Outputs
    // ===================================================================
    task check_outputs;
        input integer out_ch;
        input integer seq_len;
        
        integer t, c, addr;
        reg signed [DATA_WIDTH-1:0] expected;
        reg signed [DATA_WIDTH-1:0] received;
        integer diff, max_diff;
        integer mismatches;
        
        begin
            $display("[Check Outputs] Verifying %0d outputs...", seq_len * out_ch);
            
            max_diff = 0;
            mismatches = 0;
            
            for (t = 0; t < seq_len; t = t + 1) begin
                for (c = 0; c < out_ch; c = c + 1) begin
                    addr = t * out_ch + c;
                    expected = expected_output[t][c];
                    received = output_bram[addr];
                    
                    diff = (expected > received) ? (expected - received) : (received - expected);
                    
                    if (diff > max_diff)
                        max_diff = diff;
                    
                    if (diff > 100) begin  // Tolerance: ±100 LSBs (~0.024 in Q4.12)
                        $display("[MISMATCH] t=%0d, ch=%0d: expected=%h, got=%h (diff=%0d)", 
                                 t, c, expected, received, diff);
                        mismatches = mismatches + 1;
                    end
                end
            end
            
            $display("[Check Outputs] Max difference: %0d LSBs", max_diff);
            
            if (mismatches > 0) begin
                $display("[FAIL] %0d mismatches found", mismatches);
                errors = errors + 1;
            end else begin
                $display("[PASS] All outputs match within tolerance");
            end
        end
    endtask

endmodule
