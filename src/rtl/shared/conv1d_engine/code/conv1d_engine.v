// =======================================================================
// Module: conv1d_engine_bram
// Purpose: Simplified 1-D convolution engine for Zynq BRAM interface
// 
// Description:
//   Multi-channel Conv1D accelerator for HiFi-GAN vocoder.
//   Reads input from BRAM, processes all channels, writes output to BRAM.
//   Designed for Zynq PS-PL integration with external FSM control.
//
// Architecture:
//   - Batch processing: reads complete sequence from BRAM
//   - Multi-channel: handles IN_CHANNELS → OUT_CHANNELS mapping
//   - Hardware sharing: one engine reused across all layers
//   - Memory interface: simple read/write to BRAM (no AXI complexity)
//
// Processing Flow:
//   For each output timestep t:
//     For each output channel o:
//       acc = 0
//       For each input channel i:
//         For each kernel position k:
//           acc += input[t+k*dilation][i] * weight[o][i][k]
//       output[t][o] = quantize(acc + bias[o])
//
// Notes:
//   - Pure Verilog-2001, synthesizable for Zynq FPGA
//   - Fixed-point: Q4.12 (data), Q2.14 (weights), Q6.26 (accumulator)
//   - ~250 lines vs 547 in streaming version
// =======================================================================

module conv1d_engine_bram #(
    parameter DATA_WIDTH      = 16,     // Q4.12 fixed-point
    parameter KERNEL_SIZE     = 3,      // Max kernel size
    parameter MAX_IN_CH       = 256,    // Max input channels
    parameter MAX_OUT_CH      = 512,    // Max output channels
    parameter MAX_SEQ_LEN     = 256,    // Max sequence length
    parameter ACTIVATION      = "NONE"  // "LEAKY_RELU", "TANH", "NONE"
)(
    input  wire                           clk,
    input  wire                           rst_n,
    
    // Control interface
    input  wire                           start,
    output reg                            done,
    output reg                            busy,
    
    // Configuration (set before start)
    input  wire [15:0]                    seq_length,
    input  wire [9:0]                     in_channels,
    input  wire [9:0]                     out_channels,
    input  wire [3:0]                     kernel_size,
    input  wire [3:0]                     dilation,
    
    // Input BRAM interface (read-only)
    output reg  [15:0]                    input_addr,    // Address: [time][channel]
    output reg                            input_rd_en,
    input  wire signed [DATA_WIDTH-1:0]  input_data,
    
    // Output BRAM interface (write-only)
    output reg  [15:0]                    output_addr,   // Address: [time][channel]
    output reg                            output_wr_en,
    output reg  signed [DATA_WIDTH-1:0]  output_data,
    
    // Weight memory interface (read-only)
    output reg  [20:0]                    weight_addr,   // Address: [out_ch][in_ch][k]
    input  wire signed [DATA_WIDTH-1:0]  weight_data,
    
    // Bias memory interface (read-only)
    output reg  [10:0]                    bias_addr,     // Address: [out_ch]
    input  wire signed [31:0]             bias_data
);

    // ===================================================================
    // FSM States
    // ===================================================================
    localparam IDLE          = 3'd0;
    localparam INIT_OUT_CH   = 3'd1;
    localparam LOAD_WEIGHTS  = 3'd2;
    localparam COMPUTE_MAC   = 3'd3;
    localparam ADD_BIAS      = 3'd4;
    localparam WRITE_OUTPUT  = 3'd5;
    localparam DONE_STATE    = 3'd6;
    
    reg [2:0] state, next_state;
    
    // ===================================================================
    // Iteration Counters
    // ===================================================================
    reg [15:0] time_idx;         // Current output timestep (0 to seq_length-1)
    reg [9:0]  out_ch;           // Current output channel (0 to out_channels-1)
    reg [9:0]  in_ch;            // Current input channel (0 to in_channels-1)
    reg [3:0]  k_idx;            // Current kernel position (0 to kernel_size-1)
    
    // ===================================================================
    // MAC and Accumulation
    // ===================================================================
    reg signed [31:0] channel_acc;      // Accumulator across input channels (Q6.26)
    reg signed [31:0] mac_result;       // Single MAC result (Q6.26)
    reg signed [DATA_WIDTH-1:0] weight_buffer [0:KERNEL_SIZE-1];
    reg signed [DATA_WIDTH-1:0] input_buffer  [0:KERNEL_SIZE-1];
    
    // ===================================================================
    // Memory Access State Machine
    // ===================================================================
    reg [3:0] mem_state;  // Sub-state for memory reads
    localparam MEM_IDLE        = 4'd0;
    localparam MEM_LOAD_INPUT  = 4'd1;
    localparam MEM_LOAD_WEIGHT = 4'd2;
    localparam MEM_COMPUTE     = 4'd3;
    localparam MEM_NEXT_INCH   = 4'd4;
    localparam MEM_DONE        = 4'd5;
    
    reg [2:0] load_counter;  // Counter for loading kernel weights/inputs
    
    // Quantizer signals
    reg signed [31:0] quant_input;
    reg quant_valid_in;
    wire signed [DATA_WIDTH-1:0] quant_output;
    wire quant_valid_out;
    
    // ===================================================================
    // Instantiate Quantizer (Q6.26 → Q4.12)
    // ===================================================================
    quantizer_32_16 u_quantizer (
        .clk(clk),
        .rst_n(rst_n),
        .i_valid(quant_valid_in),
        .i_acc_raw(quant_input),
        .o_data(quant_output),
        .o_valid_out(quant_valid_out)
    );
    
    // ===================================================================
    // Optional Activation Function
    // ===================================================================
    wire signed [DATA_WIDTH-1:0] activated_output;
    
    generate
        if (ACTIVATION == "LEAKY_RELU") begin : gen_leaky_relu
            leaky_relu_q15 u_activation (
                .clk(clk),
                .rst_n(rst_n),
                .x(quant_output),
                .y(activated_output)
            );
        end else if (ACTIVATION == "TANH") begin : gen_tanh
            tanh_approx_q15 u_activation (
                .clk(clk),
                .rst_n(rst_n),
                .x(quant_output),
                .y(activated_output)
            );
        end else begin : gen_bypass
            assign activated_output = quant_output;
        end
    endgenerate
    
    // ===================================================================
    // FSM: State Register
    // ===================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            state <= IDLE;
        else
            state <= next_state;
    end
    
    // ===================================================================
    // FSM: Next State Logic
    // ===================================================================
    always @(*) begin
        next_state = state;
        
        case (state)
            IDLE: begin
                if (start)
                    next_state = INIT_OUT_CH;
            end
            
            INIT_OUT_CH: begin
                next_state = LOAD_WEIGHTS;
            end
            
            LOAD_WEIGHTS: begin
                if (mem_state == MEM_DONE)
                    next_state = COMPUTE_MAC;
            end
            
            COMPUTE_MAC: begin
                if (in_ch >= in_channels - 1)
                    next_state = ADD_BIAS;
                else
                    next_state = LOAD_WEIGHTS;  // Next input channel
            end
            
            ADD_BIAS: begin
                next_state = WRITE_OUTPUT;
            end
            
            WRITE_OUTPUT: begin
                // Check if all output channels done for this timestep
                if (out_ch >= out_channels - 1) begin
                    // Move to next timestep
                    if (time_idx >= seq_length - 1)
                        next_state = DONE_STATE;
                    else
                        next_state = INIT_OUT_CH;
                end else begin
                    // More output channels for this timestep
                    next_state = INIT_OUT_CH;
                end
            end
            
            DONE_STATE: begin
                next_state = IDLE;
            end
            
            default: next_state = IDLE;
        endcase
    end
    
    // ===================================================================
    // FSM: Datapath and Memory Control
    // ===================================================================
    integer i;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            done <= 1'b0;
            busy <= 1'b0;
            
            time_idx <= 0;
            out_ch <= 0;
            in_ch <= 0;
            k_idx <= 0;
            
            channel_acc <= 0;
            mac_result <= 0;
            
            input_addr <= 0;
            input_rd_en <= 1'b0;
            output_addr <= 0;
            output_wr_en <= 1'b0;
            output_data <= 0;
            weight_addr <= 0;
            bias_addr <= 0;
            
            mem_state <= MEM_IDLE;
            load_counter <= 0;
            quant_input <= 0;
            quant_valid_in <= 1'b0;
            
            for (i = 0; i < KERNEL_SIZE; i = i + 1) begin
                weight_buffer[i] <= 0;
                input_buffer[i] <= 0;
            end
            
        end else begin
            // Default assignments
            input_rd_en <= 1'b0;
            output_wr_en <= 1'b0;
            done <= 1'b0;
            quant_valid_in <= 1'b0;
            
            case (state)
                IDLE: begin
                    busy <= 1'b0;
                    if (start) begin
                        busy <= 1'b1;
                        time_idx <= 0;
                        out_ch <= 0;
                        in_ch <= 0;
                    end
                end
                
                INIT_OUT_CH: begin
                    busy <= 1'b1;
                    channel_acc <= 0;
                    in_ch <= 0;
                    mem_state <= MEM_IDLE;
                end
                
                LOAD_WEIGHTS: begin
                    busy <= 1'b1;
                    
                    // Memory access state machine for loading kernel
                    case (mem_state)
                        MEM_IDLE: begin
                            load_counter <= 0;
                            k_idx <= 0;
                            mem_state <= MEM_LOAD_INPUT;
                        end
                        
                        MEM_LOAD_INPUT: begin
                            // Calculate input address: input[time + k*dilation][in_ch]
                            // Address layout: [time * in_channels + in_ch]
                            if (load_counter < kernel_size) begin
                                input_addr <= (time_idx + load_counter * dilation) * in_channels + in_ch;
                                input_rd_en <= 1'b1;
                                
                                // Capture data from previous cycle
                                if (load_counter > 0)
                                    input_buffer[load_counter - 1] <= input_data;
                                
                                load_counter <= load_counter + 1;
                            end else begin
                                // Capture last data
                                input_buffer[kernel_size - 1] <= input_data;
                                load_counter <= 0;
                                mem_state <= MEM_LOAD_WEIGHT;
                            end
                        end
                        
                        MEM_LOAD_WEIGHT: begin
                            // Calculate weight address: weight[out_ch][in_ch][k]
                            if (load_counter < kernel_size) begin
                                weight_addr <= (out_ch * in_channels * kernel_size) + 
                                             (in_ch * kernel_size) + load_counter;
                                
                                // Capture data from previous cycle
                                if (load_counter > 0)
                                    weight_buffer[load_counter - 1] <= weight_data;
                                
                                load_counter <= load_counter + 1;
                            end else begin
                                // Capture last data
                                weight_buffer[kernel_size - 1] <= weight_data;
                                mem_state <= MEM_COMPUTE;
                            end
                        end
                        
                        MEM_COMPUTE: begin
                            mem_state <= MEM_DONE;
                        end
                        
                        MEM_DONE: begin
                            // Hold until FSM advances
                        end
                    endcase
                end
                
                COMPUTE_MAC: begin
                    busy <= 1'b1;
                    
                    // Perform MAC: sum(input[k] * weight[k]) for k=0 to kernel_size-1
                    mac_result = 0;
                    for (i = 0; i < KERNEL_SIZE; i = i + 1) begin
                        if (i < kernel_size) begin
                            // Q4.12 * Q2.14 = Q6.26
                            mac_result = mac_result + (input_buffer[i] * weight_buffer[i]);
                        end
                    end
                    
                    // Accumulate across input channels
                    channel_acc <= channel_acc + mac_result;
                    
                    // Move to next input channel
                    in_ch <= in_ch + 1;
                    mem_state <= MEM_IDLE;
                end
                
                ADD_BIAS: begin
                    busy <= 1'b1;
                    
                    // Read bias for current output channel
                    bias_addr <= out_ch[10:0];
                    
                    // Add bias and send to quantizer
                    quant_input <= channel_acc + bias_data;
                    quant_valid_in <= 1'b1;
                end
                
                WRITE_OUTPUT: begin
                    busy <= 1'b1;
                    
                    // Write quantized and activated output to BRAM
                    // Address layout: [time * out_channels + out_ch]
                    output_addr <= time_idx * out_channels + out_ch;
                    output_data <= activated_output;
                    output_wr_en <= 1'b1;
                    
                    // Update counters for next iteration
                    if (out_ch >= out_channels - 1) begin
                        // All output channels done, move to next timestep
                        out_ch <= 0;
                        time_idx <= time_idx + 1;
                    end else begin
                        // More output channels to process
                        out_ch <= out_ch + 1;
                    end
                end
                
                DONE_STATE: begin
                    busy <= 1'b0;
                    done <= 1'b1;
                end
            endcase
        end
    end

endmodule
