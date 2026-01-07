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
    localparam [2:0] ST_IDLE       = 3'd0;
    localparam [2:0] ST_LOAD       = 3'd1;
    localparam [2:0] ST_COMPUTE    = 3'd2;
    localparam [2:0] ST_ACTIVATE   = 3'd3;
    localparam [2:0] ST_OUTPUT     = 3'd4;
    localparam [2:0] ST_DONE       = 3'd5;

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
    
    // Weight and bias memory (inferred BRAM)
    // Organized as: [layer][channel_out][channel_in][kernel_pos]
    reg signed [DATA_WIDTH-1:0] weight_mem [0:WEIGHT_DEPTH-1];
    reg signed [DATA_WIDTH-1:0] bias_mem   [0:NUM_LAYERS*NUM_CHANNELS-1];
    
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
    
    // Weight address calculation
    reg [$clog2(WEIGHT_DEPTH)-1:0]   weight_addr;
    reg [$clog2(NUM_LAYERS*NUM_CHANNELS)-1:0] bias_addr;
    
    // Intermediate storage
    reg signed [DATA_WIDTH-1:0] conv_result_r;
    
    //==========================================================================
    // Memory Initialization
    //==========================================================================
    initial begin
        $readmemh("postnet_weights.mem", weight_mem);
        $readmemh("postnet_bias.mem", bias_mem);
    end

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
                    state_next = ST_COMPUTE;
            end
            
            ST_COMPUTE: begin
                // Process all kernel elements and channels
                if (mac_valid && kernel_cnt_r == KERNEL_SIZE - 1)
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
            bias_addr     <= 0;
            conv_result_r <= 0;
        end
        else begin
            // Default pulse signals
            o_done       <= 1'b0;
            o_data_valid <= 1'b0;
            mac_calc_en  <= 1'b0;
            mac_clear_acc <= 1'b0;
            quant_valid_in <= 1'b0;
            
            case (state_r)
                ST_IDLE: begin
                    o_busy <= 1'b0;
                    if (i_start) begin
                        o_busy        <= 1'b1;
                        sample_cnt_r  <= 0;
                        channel_cnt_r <= 0;
                        kernel_cnt_r  <= 0;
                        buf_wr_ptr    <= 0;
                        buf_ready     <= 1'b0;
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
                
                ST_COMPUTE: begin
                    // Pack line buffer into MAC input
                    mac_activations <= {line_buffer[4], line_buffer[3], 
                                        line_buffer[2], line_buffer[1], 
                                        line_buffer[0]};
                    
                    // Calculate weight address
                    // Address = layer_offset + channel_offset + kernel_pos
                    weight_addr <= (i_layer_sel * NUM_CHANNELS * KERNEL_SIZE) + 
                                   (channel_cnt_r * KERNEL_SIZE) + kernel_cnt_r;
                    
                    // Load weights from memory
                    mac_weights <= weight_mem[weight_addr];
                    
                    // Start MAC computation
                    mac_calc_en <= 1'b1;
                    mac_clear_acc <= (kernel_cnt_r == 0);
                    
                    if (mac_valid) begin
                        if (kernel_cnt_r < KERNEL_SIZE - 1) begin
                            kernel_cnt_r <= kernel_cnt_r + 1;
                        end
                    end
                end
                
                ST_ACTIVATE: begin
                    // Add bias and prepare for quantization
                    bias_addr <= (i_layer_sel * NUM_CHANNELS) + channel_cnt_r;
                    
                    // Extend bias to 32-bit and add to accumulator
                    quant_data_in  <= mac_acc_raw + {{16{bias_mem[bias_addr][15]}}, 
                                                      bias_mem[bias_addr]};
                    quant_valid_in <= 1'b1;
                    
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