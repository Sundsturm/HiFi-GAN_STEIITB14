//==============================================================================
// Module: postnet_stack
// Purpose: Sequential Conv1D layers for waveform refinement in HiFi-GAN PostNet.
//          This module implements a stack of Conv1D layers that refine the
//          raw waveform output from the Generator. Each layer applies:
//          Conv1D -> Activation (tanh for intermediate, none for last layer)
//
// Inputs:
//   - clk          : System clock
//   - rst_n        : Active-low async reset
//   - i_start      : Start signal to begin processing
//   - i_data       : Input sample data (Q4.12 format)
//   - i_data_valid : Input data valid strobe
//   - i_layer_sel  : Current layer selection (0 to NUM_LAYERS-1)
//
// Outputs:
//   - o_data       : Output sample data (Q4.12 format)
//   - o_data_valid : Output data valid strobe
//   - o_done       : Done signal when layer processing complete
//
// Fixed-point Format:
//   - Data: Q4.12 (16-bit signed, 4 integer bits, 12 fractional bits)
//   - Weights: Q2.14 (16-bit signed)
//   - Accumulator: Q6.26 (32-bit signed)
//==============================================================================

module postnet_stack #(
    parameter DATA_WIDTH      = 16,         // Q4.12 data width
    parameter ACC_WIDTH       = 32,         // Accumulator width (Q6.26)
    parameter KERNEL_SIZE     = 5,          // PostNet typically uses kernel=5
    parameter NUM_CHANNELS    = 32,         // Channel depth per layer
    parameter NUM_LAYERS      = 5,          // Number of PostNet Conv1D layers
    parameter SAMPLE_LENGTH   = 256,        // Max samples per inference
    parameter WEIGHT_DEPTH    = 1024        // Weight memory depth per layer
)(
    input wire                          clk,
    input wire                          rst_n,
    
    // Control Interface
    input wire                          i_start,
    input wire [$clog2(NUM_LAYERS)-1:0] i_layer_sel,
    
    // Data Input Interface
    input wire signed [DATA_WIDTH-1:0]  i_data,
    input wire                          i_data_valid,
    
    // Data Output Interface
    output reg signed [DATA_WIDTH-1:0]  o_data,
    output reg                          o_data_valid,
    
    // Status
    output reg                          o_busy,
    output reg                          o_done
);

    //==========================================================================
    // Local Parameters
    //==========================================================================
    localparam KERNEL_HALF = (KERNEL_SIZE - 1) / 2;
    
    // FSM States
    localparam [2:0] ST_IDLE         = 3'd0;
    localparam [2:0] ST_LOAD         = 3'd1;
    localparam [2:0] ST_FETCH_WEIGHT = 3'd2;
    localparam [2:0] ST_COMPUTE      = 3'd3;
    localparam [2:0] ST_ACTIVATE     = 3'd4;
    localparam [2:0] ST_OUTPUT       = 3'd5;
    localparam [2:0] ST_DONE         = 3'd6;

    //==========================================================================
    // Internal Registers
    //==========================================================================
    reg [2:0] state_r, state_next;
    
    // Sample counters
    reg [$clog2(SAMPLE_LENGTH)-1:0] sample_cnt_r;
    reg [$clog2(NUM_CHANNELS)-1:0]  channel_cnt_r;
    reg [$clog2(KERNEL_SIZE)-1:0]   kernel_cnt_r;
    
    // Line buffer for sliding window convolution
    reg signed [DATA_WIDTH-1:0] line_buffer [0:KERNEL_SIZE-1];
    reg [$clog2(KERNEL_SIZE)-1:0] buf_wr_ptr;
    reg buf_ready;
    
    // Shared memory interface signals
    reg [$clog2(WEIGHT_DEPTH)-1:0]   weight_addr;
    reg                               weight_rd_en;
    wire signed [DATA_WIDTH-1:0]      weight_data;
    wire                              weight_valid;
    
    reg [$clog2(NUM_LAYERS*NUM_CHANNELS)-1:0] bias_addr;
    reg                               bias_rd_en;
    wire signed [DATA_WIDTH-1:0]      bias_data;
    wire                              bias_valid;
    
    // Weight fetch buffer (for sequential reads)
    reg signed [DATA_WIDTH-1:0] weight_buffer [0:KERNEL_SIZE-1];
    reg [$clog2(KERNEL_SIZE)-1:0] weight_fetch_cnt;
    reg weight_fetch_done;
    
    // MAC interface signals
    reg                              mac_calc_en;
    reg                              mac_clear_acc;
    reg signed [KERNEL_SIZE*DATA_WIDTH-1:0] mac_activations;
    reg signed [KERNEL_SIZE*DATA_WIDTH-1:0] mac_weights;
    wire signed [ACC_WIDTH-1:0]      mac_acc_raw;
    wire                             mac_valid;
    
    // Quantizer interface signals
    reg                              quant_valid_in;
    reg signed [ACC_WIDTH-1:0]       quant_data_in;
    wire signed [DATA_WIDTH-1:0]     quant_data_out;
    wire                             quant_valid_out;
    
    // Activation interface signals
    wire signed [DATA_WIDTH-1:0]     act_out;
    reg  signed [DATA_WIDTH-1:0]     post_act_data;
    
    // Intermediate storage
    reg signed [DATA_WIDTH-1:0] conv_result_r;
    
    //==========================================================================
    // Shared Memory Instances
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
        .DEPTH(NUM_LAYERS*NUM_CHANNELS),
        .MEM_FILE("biases.mem")
    ) u_bias_mem (
        .clk     (clk),
        .rst_n   (rst_n),
        .i_addr  (bias_addr),
        .i_rd_en (bias_rd_en),
        .o_data  (bias_data),
        .o_valid (bias_valid)
    );

    //==========================================================================
    // MAC Array Instance
    //==========================================================================
    hifigan_mac_array #(
        .KERNEL_SIZE(KERNEL_SIZE),
        .DATA_WIDTH(DATA_WIDTH)
    ) u_mac_array (
        .clk          (clk),
        .rst_n        (rst_n),
        .i_calc_en    (mac_calc_en),
        .i_clear_acc  (mac_clear_acc),
        .i_activations(mac_activations),
        .i_weights    (mac_weights),
        .o_acc_raw    (mac_acc_raw),
        .o_valid      (mac_valid)
    );

    //==========================================================================
    // Quantizer Instance (32-bit to 16-bit with saturation)
    //==========================================================================
    quantizer_32_16 u_quantizer (
        .clk         (clk),
        .rst_n       (rst_n),
        .i_valid     (quant_valid_in),
        .i_acc_raw   (quant_data_in),
        .o_data      (quant_data_out),
        .o_valid_out (quant_valid_out)
    );

    //==========================================================================
    // Activation Unit Instance (tanh for intermediate layers)
    //==========================================================================
    tanh_approx_q15 u_tanh (
        .x(quant_data_out),
        .y(act_out)
    );

    //==========================================================================
    // FSM: State Register
    //==========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            state_r <= ST_IDLE;
        else
            state_r <= state_next;
    end

    //==========================================================================
    // FSM: Next State Logic
    //==========================================================================
    always @(*) begin
        state_next = state_r;
        
        case (state_r)
            ST_IDLE: begin
                if (i_start)
                    state_next = ST_LOAD;
            end
            
            ST_LOAD: begin
                // Wait until line buffer has enough samples
                if (buf_ready)
                    state_next = ST_FETCH_WEIGHT;
            end
            
            ST_FETCH_WEIGHT: begin
                // Fetch all kernel weights sequentially
                if (weight_fetch_done)
                    state_next = ST_COMPUTE;
            end
            
            ST_COMPUTE: begin
                // MAC computation with fetched weights
                if (mac_valid)
                    state_next = ST_ACTIVATE;
            end
            
            ST_ACTIVATE: begin
                // Wait for quantization and activation
                if (quant_valid_out)
                    state_next = ST_OUTPUT;
            end
            
            ST_OUTPUT: begin
                if (sample_cnt_r >= SAMPLE_LENGTH - 1)
                    state_next = ST_DONE;
                else
                    state_next = ST_LOAD;
            end
            
            ST_DONE: begin
                state_next = ST_IDLE;
            end
            
            default: state_next = ST_IDLE;
        endcase
    end

    //==========================================================================
    // FSM: Output and Control Logic
    //==========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            o_busy       <= 1'b0;
            o_done       <= 1'b0;
            o_data       <= {DATA_WIDTH{1'b0}};
            o_data_valid <= 1'b0;
            
            sample_cnt_r  <= 0;
            channel_cnt_r <= 0;
            kernel_cnt_r  <= 0;
            buf_wr_ptr    <= 0;
            buf_ready     <= 1'b0;
            
            mac_calc_en   <= 1'b0;
            mac_clear_acc <= 1'b0;
            quant_valid_in <= 1'b0;
            
            weight_addr   <= 0;
            weight_rd_en  <= 1'b0;
            bias_addr     <= 0;
            bias_rd_en    <= 1'b0;
            conv_result_r <= 0;
            
            weight_fetch_cnt <= 0;
            weight_fetch_done <= 1'b0;
        end
        else begin
            // Default pulse signals
            o_done         <= 1'b0;
            o_data_valid   <= 1'b0;
            mac_calc_en    <= 1'b0;
            mac_clear_acc  <= 1'b0;
            quant_valid_in <= 1'b0;
            weight_rd_en   <= 1'b0;
            bias_rd_en     <= 1'b0;
            
            case (state_r)
                ST_IDLE: begin
                    o_busy <= 1'b0;
                    if (i_start) begin
                        o_busy            <= 1'b1;
                        sample_cnt_r      <= 0;
                        channel_cnt_r     <= 0;
                        kernel_cnt_r      <= 0;
                        buf_wr_ptr        <= 0;
                        buf_ready         <= 1'b0;
                        weight_fetch_cnt  <= 0;
                        weight_fetch_done <= 1'b0;
                    end
                end
                
                ST_LOAD: begin
                    // Load input samples into line buffer
                    if (i_data_valid) begin
                        line_buffer[buf_wr_ptr] <= i_data;
                        
                        if (buf_wr_ptr >= KERNEL_SIZE - 1) begin
                            buf_ready  <= 1'b1;
                            buf_wr_ptr <= KERNEL_HALF; // Keep center for sliding
                        end
                        else begin
                            buf_wr_ptr <= buf_wr_ptr + 1;
                        end
                    end
                end
                
                ST_FETCH_WEIGHT: begin
                    // Sequential weight fetch from shared memory
                    if (!weight_fetch_done) begin
                        // Calculate weight address
                        // Address = layer_offset + channel_offset + kernel_pos
                        weight_addr <= (i_layer_sel * NUM_CHANNELS * KERNEL_SIZE) + 
                                       (channel_cnt_r * KERNEL_SIZE) + weight_fetch_cnt;
                        weight_rd_en <= 1'b1;
                        
                        // Store fetched weight in buffer
                        if (weight_valid) begin
                            weight_buffer[weight_fetch_cnt - 1] <= weight_data;
                            
                            if (weight_fetch_cnt >= KERNEL_SIZE) begin
                                weight_fetch_done <= 1'b1;
                            end
                            else begin
                                weight_fetch_cnt <= weight_fetch_cnt + 1;
                            end
                        end
                        else if (weight_rd_en) begin
                            weight_fetch_cnt <= weight_fetch_cnt + 1;
                        end
                    end
                end
                
                ST_COMPUTE: begin
                    // Pack line buffer into MAC input
                    mac_activations <= {line_buffer[4], line_buffer[3], 
                                        line_buffer[2], line_buffer[1], 
                                        line_buffer[0]};
                    
                    // Pack fetched weights into MAC input
                    mac_weights <= {weight_buffer[4], weight_buffer[3],
                                    weight_buffer[2], weight_buffer[1],
                                    weight_buffer[0]};
                    
                    // Start MAC computation
                    mac_calc_en   <= 1'b1;
                    mac_clear_acc <= 1'b1;  // Single MAC operation per channel
                    
                    // Reset weight fetch for next iteration
                    weight_fetch_cnt  <= 0;
                    weight_fetch_done <= 1'b0;
                end
                
                ST_ACTIVATE: begin
                    // Read bias from shared memory
                    bias_addr  <= (i_layer_sel * NUM_CHANNELS) + channel_cnt_r;
                    bias_rd_en <= 1'b1;
                    
                    // Add bias to accumulator when valid
                    if (bias_valid) begin
                        // Extend bias to 32-bit and add to accumulator
                        quant_data_in  <= mac_acc_raw + {{16{bias_data[15]}}, bias_data};
                        quant_valid_in <= 1'b1;
                    end
                    
                    // Apply activation based on layer
                    if (quant_valid_out) begin
                        // Last layer: no activation (linear)
                        // Intermediate layers: tanh activation
                        if (i_layer_sel == NUM_LAYERS - 1)
                            post_act_data <= quant_data_out;
                        else
                            post_act_data <= act_out;
                    end
                end
                
                ST_OUTPUT: begin
                    // Output the processed sample
                    o_data       <= post_act_data;
                    o_data_valid <= 1'b1;
                    
                    // Shift line buffer for next sample
                    line_buffer[0] <= line_buffer[1];
                    line_buffer[1] <= line_buffer[2];
                    line_buffer[2] <= line_buffer[3];
                    line_buffer[3] <= line_buffer[4];
                    
                    sample_cnt_r <= sample_cnt_r + 1;
                    kernel_cnt_r <= 0;
                end
                
                ST_DONE: begin
                    o_busy <= 1'b0;
                    o_done <= 1'b1;
                end
            endcase
        end
    end

endmodule