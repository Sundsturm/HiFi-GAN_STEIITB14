//==============================================================================
// Module: postnet_top
// Purpose: Top-level wrapper for HiFi-GAN PostNet module.
//          Integrates the postnet_stack (Conv1D layers) with the postnet_fsm
//          (control sequencer) and performs residual summation between the
//          Generator output and the PostNet refinement output.
//
//          Signal flow: Generator Output -> PostNet Stack -> Residual Add -> Audio Output
//
// Inputs:
//   - clk            : System clock
//   - rst_n          : Active-low async reset
//   - i_start        : Start signal from top-level FSM
//   - i_gen_data     : Generator output waveform samples (Q4.12)
//   - i_gen_valid    : Generator output valid strobe
//
// Outputs:
//   - o_audio        : Final refined audio output (Q4.12)
//   - o_audio_valid  : Audio output valid strobe
//   - o_busy         : PostNet is processing
//   - o_done         : PostNet processing complete
//
// Fixed-point Format:
//   - All data paths: Q4.12 (16-bit signed, 4 integer bits, 12 fractional bits)
//==============================================================================

module postnet_top #(
    parameter DATA_WIDTH      = 16,         // Q4.12 data width
    parameter ACC_WIDTH       = 32,         // Accumulator width (Q6.26)
    parameter KERNEL_SIZE     = 5,          // PostNet Conv1D kernel size
    parameter NUM_CHANNELS    = 32,         // Channels per layer
    parameter NUM_LAYERS      = 5,          // Number of Conv1D layers
    parameter SAMPLE_LENGTH   = 256,        // Max samples per inference
    parameter WEIGHT_DEPTH    = 1024        // Weight memory depth
)(
    input wire                          clk,
    input wire                          rst_n,
    
    // Control Interface (from top_fsm)
    input wire                          i_start,
    
    // Generator Output Interface
    input wire signed [DATA_WIDTH-1:0]  i_gen_data,
    input wire                          i_gen_valid,
    
    // Audio Output Interface
    output wire signed [DATA_WIDTH-1:0] o_audio,
    output wire                         o_audio_valid,
    
    // Status
    output wire                         o_busy,
    output wire                         o_done
);

    //==========================================================================
    // Internal Wires and Registers
    //==========================================================================
    
    // FSM -> Stack interface
    wire                                fsm_stack_start;
    wire [$clog2(NUM_LAYERS)-1:0]       fsm_layer_sel;
    wire                                fsm_busy;
    wire                                fsm_done;
    
    // Stack output signals
    wire signed [DATA_WIDTH-1:0]        stack_data_out;
    wire                                stack_data_valid;
    wire                                stack_busy;
    wire                                stack_done;
    
    // Input buffer for storing Generator output (for residual add)
    reg signed [DATA_WIDTH-1:0] gen_buffer [0:SAMPLE_LENGTH-1];
    reg [$clog2(SAMPLE_LENGTH)-1:0] gen_wr_ptr;
    reg [$clog2(SAMPLE_LENGTH)-1:0] gen_rd_ptr;
    reg gen_buffer_ready;
    
    // Intermediate layer buffer (ping-pong between layers)
    reg signed [DATA_WIDTH-1:0] layer_buffer [0:SAMPLE_LENGTH-1];
    reg [$clog2(SAMPLE_LENGTH)-1:0] layer_wr_ptr;
    reg [$clog2(SAMPLE_LENGTH)-1:0] layer_rd_ptr;
    
    // Current data feeding into stack
    reg signed [DATA_WIDTH-1:0]  stack_data_in;
    reg                          stack_data_in_valid;
    
    // Residual addition signals
    reg signed [DATA_WIDTH-1:0]  residual_sum;
    reg                          residual_valid;
    
    // Layer processing state
    reg [$clog2(NUM_LAYERS)-1:0] current_layer;
    reg                          processing_first_layer;
    reg                          processing_last_layer;
    
    //==========================================================================
    // Saturation Logic for Residual Addition
    //==========================================================================
    // Q4.12 max = 0x7FFF (+7.9999...), min = 0x8000 (-8.0)
    localparam signed [DATA_WIDTH-1:0] MAX_SAT = 16'sh7FFF;
    localparam signed [DATA_WIDTH-1:0] MIN_SAT = 16'sh8000;
    
    // Extended precision for addition (17-bit to catch overflow)
    wire signed [DATA_WIDTH:0] residual_extended;
    wire overflow_pos, overflow_neg;
    reg signed [DATA_WIDTH-1:0] residual_saturated;
    
    assign residual_extended = {stack_data_out[DATA_WIDTH-1], stack_data_out} + 
                               {gen_buffer[gen_rd_ptr][DATA_WIDTH-1], gen_buffer[gen_rd_ptr]};
    assign overflow_pos = (~residual_extended[DATA_WIDTH] && residual_extended[DATA_WIDTH-1]);
    assign overflow_neg = (residual_extended[DATA_WIDTH] && ~residual_extended[DATA_WIDTH-1]);
    
    always @(*) begin
        if (overflow_pos)
            residual_saturated = MAX_SAT;
        else if (overflow_neg)
            residual_saturated = MIN_SAT;
        else
            residual_saturated = residual_extended[DATA_WIDTH-1:0];
    end

    //==========================================================================
    // PostNet FSM Instance
    //==========================================================================
    postnet_fsm #(
        .NUM_LAYERS    (NUM_LAYERS),
        .SAMPLE_LENGTH (SAMPLE_LENGTH)
    ) u_postnet_fsm (
        .clk           (clk),
        .rst_n         (rst_n),
        .i_start       (i_start),
        .i_stack_done  (stack_done),
        .o_stack_start (fsm_stack_start),
        .o_layer_sel   (fsm_layer_sel),
        .o_busy        (fsm_busy),
        .o_done        (fsm_done)
    );

    //==========================================================================
    // PostNet Stack Instance (Conv1D Layers)
    //==========================================================================
    postnet_stack #(
        .DATA_WIDTH    (DATA_WIDTH),
        .ACC_WIDTH     (ACC_WIDTH),
        .KERNEL_SIZE   (KERNEL_SIZE),
        .NUM_CHANNELS  (NUM_CHANNELS),
        .NUM_LAYERS    (NUM_LAYERS),
        .SAMPLE_LENGTH (SAMPLE_LENGTH),
        .WEIGHT_DEPTH  (WEIGHT_DEPTH)
    ) u_postnet_stack (
        .clk           (clk),
        .rst_n         (rst_n),
        .i_start       (fsm_stack_start),
        .i_layer_sel   (fsm_layer_sel),
        .i_data        (stack_data_in),
        .i_data_valid  (stack_data_in_valid),
        .o_data        (stack_data_out),
        .o_data_valid  (stack_data_valid),
        .o_busy        (stack_busy),
        .o_done        (stack_done)
    );

    //==========================================================================
    // Generator Output Buffer (stores input for residual skip connection)
    //==========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            gen_wr_ptr <= 0;
            gen_buffer_ready <= 1'b0;
        end
        else begin
            if (i_start) begin
                gen_wr_ptr <= 0;
                gen_buffer_ready <= 1'b0;
            end
            else if (i_gen_valid) begin
                gen_buffer[gen_wr_ptr] <= i_gen_data;
                gen_wr_ptr <= gen_wr_ptr + 1;
                
                if (gen_wr_ptr == SAMPLE_LENGTH - 1)
                    gen_buffer_ready <= 1'b1;
            end
        end
    end

    //==========================================================================
    // Layer Processing Control
    //==========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_layer <= 0;
            processing_first_layer <= 1'b1;
            processing_last_layer <= 1'b0;
        end
        else begin
            if (i_start) begin
                current_layer <= 0;
                processing_first_layer <= 1'b1;
                processing_last_layer <= 1'b0;
            end
            else if (stack_done) begin
                if (current_layer < NUM_LAYERS - 1) begin
                    current_layer <= current_layer + 1;
                    processing_first_layer <= 1'b0;
                    processing_last_layer <= (current_layer == NUM_LAYERS - 2);
                end
            end
        end
    end

    //==========================================================================
    // Layer Buffer Management (ping-pong for intermediate results)
    //==========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            layer_wr_ptr <= 0;
            layer_rd_ptr <= 0;
        end
        else begin
            if (fsm_stack_start) begin
                layer_wr_ptr <= 0;
                layer_rd_ptr <= 0;
            end
            else begin
                // Write stack output to layer buffer (for next layer input)
                if (stack_data_valid && !processing_last_layer) begin
                    layer_buffer[layer_wr_ptr] <= stack_data_out;
                    layer_wr_ptr <= layer_wr_ptr + 1;
                end
                
                // Read from layer buffer for next layer input
                if (stack_data_in_valid && !processing_first_layer) begin
                    layer_rd_ptr <= layer_rd_ptr + 1;
                end
            end
        end
    end

    //==========================================================================
    // Stack Input Mux (first layer uses gen_buffer, others use layer_buffer)
    //==========================================================================
    always @(*) begin
        if (processing_first_layer) begin
            stack_data_in = gen_buffer[gen_rd_ptr];
        end
        else begin
            stack_data_in = layer_buffer[layer_rd_ptr];
        end
    end

    // Stack input valid generation (simplified - driven by FSM)
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            stack_data_in_valid <= 1'b0;
            gen_rd_ptr <= 0;
        end
        else begin
            stack_data_in_valid <= 1'b0;
            
            if (fsm_stack_start) begin
                gen_rd_ptr <= 0;
            end
            else if (stack_busy && gen_buffer_ready) begin
                // Feed samples to stack during processing
                stack_data_in_valid <= 1'b1;
                if (processing_first_layer && stack_data_valid) begin
                    // Advance read pointer during first layer for residual add later
                end
            end
        end
    end

    //==========================================================================
    // Residual Addition (final layer output + original generator input)
    //==========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            residual_sum <= 0;
            residual_valid <= 1'b0;
            gen_rd_ptr <= 0;
        end
        else begin
            residual_valid <= 1'b0;
            
            if (i_start) begin
                gen_rd_ptr <= 0;
            end
            else if (processing_last_layer && stack_data_valid) begin
                // Perform residual addition on last layer output
                residual_sum <= residual_saturated;
                residual_valid <= 1'b1;
                gen_rd_ptr <= gen_rd_ptr + 1;
            end
        end
    end

    //==========================================================================
    // Output Assignments
    //==========================================================================
    assign o_audio       = residual_sum;
    assign o_audio_valid = residual_valid;
    assign o_busy        = fsm_busy;
    assign o_done        = fsm_done;

endmodule