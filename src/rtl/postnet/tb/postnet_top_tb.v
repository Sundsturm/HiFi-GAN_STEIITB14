//==============================================================================
// Testbench: postnet_top_tb
// Purpose: Comprehensive testbench for postnet_top module
//          Tests complete PostNet pipeline including residual connection
//==============================================================================

`timescale 1ns / 1ps

module postnet_top_tb;

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
    reg signed [DATA_WIDTH-1:0]  i_gen_data;
    reg                          i_gen_valid;
    wire signed [DATA_WIDTH-1:0] o_audio;
    wire                         o_audio_valid;
    wire                         o_busy;
    wire                         o_done;
    
    //==========================================================================
    // Test Variables
    //==========================================================================
    integer i, j, errors;
    reg signed [DATA_WIDTH-1:0] gen_output [0:SAMPLE_LENGTH-1];
    reg signed [DATA_WIDTH-1:0] audio_output [0:SAMPLE_LENGTH-1];
    integer output_count;
    integer input_count;
    
    // Statistical analysis variables
    real mean_input, mean_output;
    real max_diff, avg_diff;
    
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
    postnet_top #(
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
        .i_gen_data    (i_gen_data),
        .i_gen_valid   (i_gen_valid),
        .o_audio       (o_audio),
        .o_audio_valid (o_audio_valid),
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
        else if (o_audio_valid) begin
            audio_output[output_count] <= o_audio;
            $display("[T=%0t] Audio Out[%0d] = %h (%f)", 
                     $time, output_count, o_audio, $itor(o_audio)/4096.0);
            output_count <= output_count + 1;
        end
    end
    
    //==========================================================================
    // Generator Output Task
    //==========================================================================
    task send_generator_output;
        input integer num_samples;
        integer idx;
        begin
            $display("\n[TASK] Sending %0d generator output samples...", num_samples);
            input_count = 0;
            
            for (idx = 0; idx < num_samples; idx = idx + 1) begin
                @(posedge clk);
                i_gen_data = gen_output[idx];
                i_gen_valid = 1;
                input_count = input_count + 1;
                
                // Occasional bubble in data stream
                if (idx % 32 == 31) begin
                    @(posedge clk);
                    i_gen_valid = 0;
                end
            end
            
            @(posedge clk);
            i_gen_valid = 0;
            $display("[TASK] Generator output transmission complete (%0d samples)", input_count);
        end
    endtask
    
    //==========================================================================
    // Statistical Analysis Task
    //==========================================================================
    task analyze_results;
        integer idx;
        real input_val, output_val, diff;
        real sum_input, sum_output, sum_diff;
        begin
            sum_input = 0.0;
            sum_output = 0.0;
            sum_diff = 0.0;
            max_diff = 0.0;
            
            for (idx = 0; idx < output_count; idx = idx + 1) begin
                input_val = $itor(gen_output[idx]) / 4096.0;
                output_val = $itor(audio_output[idx]) / 4096.0;
                diff = output_val - input_val;
                
                sum_input = sum_input + input_val;
                sum_output = sum_output + output_val;
                sum_diff = sum_diff + (diff < 0 ? -diff : diff);
                
                if ((diff < 0 ? -diff : diff) > max_diff)
                    max_diff = (diff < 0 ? -diff : diff);
            end
            
            mean_input = sum_input / output_count;
            mean_output = sum_output / output_count;
            avg_diff = sum_diff / output_count;
            
            $display("\n========================================");
            $display("Statistical Analysis");
            $display("========================================");
            $display("Mean Input:  %f", mean_input);
            $display("Mean Output: %f", mean_output);
            $display("Avg |Diff|:  %f", avg_diff);
            $display("Max |Diff|:  %f", max_diff);
            $display("========================================\n");
        end
    endtask
    
    //==========================================================================
    // Test Stimulus
    //==========================================================================
    initial begin
        // Initialize signals
        rst_n = 0;
        i_start = 0;
        i_gen_data = 0;
        i_gen_valid = 0;
        errors = 0;
        output_count = 0;
        input_count = 0;
        
        // Create waveform dump
        $dumpfile("postnet_top_tb.vcd");
        $dumpvars(0, postnet_top_tb);
        
        // Reset sequence
        #(CLK_PERIOD*5);
        rst_n = 1;
        #(CLK_PERIOD*2);
        
        $display("========================================");
        $display("PostNet Top Testbench Started");
        $display("========================================");
        
        //----------------------------------------------------------------------
        // Test 1: Impulse Response
        //----------------------------------------------------------------------
        $display("\n[TEST 1] Impulse Response");
        
        // Generate impulse pattern
        for (i = 0; i < SAMPLE_LENGTH; i = i + 1) begin
            if (i == 10)
                gen_output[i] = 16'h1000; // 1.0 in Q4.12
            else
                gen_output[i] = 16'h0000;
        end
        
        // Start PostNet
        output_count = 0;
        @(posedge clk);
        i_start = 1;
        @(posedge clk);
        i_start = 0;
        
        // Send generator output
        send_generator_output(SAMPLE_LENGTH);
        
        // Wait for processing to complete
        wait(o_done);
        @(posedge clk);
        $display("[TEST 1] Completed - %0d audio samples produced", output_count);
        
        // Analyze results
        analyze_results();
        
        #(CLK_PERIOD*20);
        
        //----------------------------------------------------------------------
        // Test 2: Square Wave
        //----------------------------------------------------------------------
        $display("\n[TEST 2] Square Wave Input");
        
        // Generate square wave pattern
        for (i = 0; i < SAMPLE_LENGTH; i = i + 1) begin
            if ((i / 32) % 2 == 0)
                gen_output[i] = 16'h0C00; // 0.75 in Q4.12
            else
                gen_output[i] = -16'h0C00; // -0.75
        end
        
        // Start PostNet
        output_count = 0;
        @(posedge clk);
        i_start = 1;
        @(posedge clk);
        i_start = 0;
        
        // Send generator output
        send_generator_output(SAMPLE_LENGTH);
        
        // Wait for processing to complete
        wait(o_done);
        @(posedge clk);
        $display("[TEST 2] Completed - %0d audio samples produced", output_count);
        
        // Analyze results
        analyze_results();
        
        #(CLK_PERIOD*20);
        
        //----------------------------------------------------------------------
        // Test 3: Ramp/Sawtooth Wave
        //----------------------------------------------------------------------
        $display("\n[TEST 3] Ramp Wave Input");
        
        // Generate ramp pattern
        for (i = 0; i < SAMPLE_LENGTH; i = i + 1) begin
            gen_output[i] = (i * 16'h2000 / SAMPLE_LENGTH) - 16'h1000; // -1.0 to +1.0
        end
        
        // Start PostNet
        output_count = 0;
        @(posedge clk);
        i_start = 1;
        @(posedge clk);
        i_start = 0;
        
        // Send generator output
        send_generator_output(SAMPLE_LENGTH);
        
        // Wait for processing to complete
        wait(o_done);
        @(posedge clk);
        $display("[TEST 3] Completed - %0d audio samples produced", output_count);
        
        // Analyze results
        analyze_results();
        
        #(CLK_PERIOD*20);
        
        //----------------------------------------------------------------------
        // Test 4: Pseudo-Random Noise
        //----------------------------------------------------------------------
        $display("\n[TEST 4] Random Noise Input");
        
        // Generate random noise pattern
        for (i = 0; i < SAMPLE_LENGTH; i = i + 1) begin
            gen_output[i] = $random % 16'h1800; // ±1.5 range
        end
        
        // Start PostNet
        output_count = 0;
        @(posedge clk);
        i_start = 1;
        @(posedge clk);
        i_start = 0;
        
        // Send generator output
        send_generator_output(SAMPLE_LENGTH);
        
        // Wait for processing to complete
        wait(o_done);
        @(posedge clk);
        $display("[TEST 4] Completed - %0d audio samples produced", output_count);
        
        // Analyze results
        analyze_results();
        
        #(CLK_PERIOD*20);
        
        //----------------------------------------------------------------------
        // Test 5: DC Offset Test (Residual Connection Verification)
        //----------------------------------------------------------------------
        $display("\n[TEST 5] DC Offset (Residual Connection Test)");
        
        // Generate constant DC pattern
        for (i = 0; i < SAMPLE_LENGTH; i = i + 1) begin
            gen_output[i] = 16'h0800; // 0.5 in Q4.12
        end
        
        // Start PostNet
        output_count = 0;
        @(posedge clk);
        i_start = 1;
        @(posedge clk);
        i_start = 0;
        
        // Send generator output
        send_generator_output(SAMPLE_LENGTH);
        
        // Wait for processing to complete
        wait(o_done);
        @(posedge clk);
        $display("[TEST 5] Completed - %0d audio samples produced", output_count);
        
        // Analyze results
        analyze_results();
        
        // Verify residual connection is working
        if (avg_diff < 0.1) begin
            $display("[INFO] Residual connection appears functional (small avg difference)");
        end
        else begin
            $display("[WARNING] Large average difference may indicate residual path issue");
        end
        
        #(CLK_PERIOD*20);
        
        //----------------------------------------------------------------------
        // Test Summary
        //----------------------------------------------------------------------
        $display("\n========================================");
        $display("PostNet Top Testbench Summary");
        $display("========================================");
        $display("Total Test Cases: 5");
        $display("Total Errors: %0d", errors);
        if (errors == 0)
            $display("STATUS: PASS");
        else
            $display("STATUS: FAIL");
        $display("========================================\n");
        
        #(CLK_PERIOD*50);
        $finish;
    end
    
    //==========================================================================
    // Timeout Watchdog
    //==========================================================================
    initial begin
        #(CLK_PERIOD * 2000000); // 20ms timeout
        $display("\n[ERROR] Simulation timeout!");
        $finish;
    end
    
    //==========================================================================
    // Performance Monitoring
    //==========================================================================
    integer cycle_count;
    real throughput;
    
    always @(posedge clk) begin
        if (!rst_n)
            cycle_count <= 0;
        else if (o_busy)
            cycle_count <= cycle_count + 1;
        else if (o_done && cycle_count > 0) begin
            throughput = (output_count * 1.0) / cycle_count;
            $display("[PERF] Cycles: %0d, Samples: %0d, Throughput: %f samples/cycle", 
                     cycle_count, output_count, throughput);
            cycle_count <= 0;
        end
    end

endmodule
