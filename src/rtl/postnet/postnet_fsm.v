//==============================================================================
// Module: postnet_fsm
// Purpose: FSM controller for sequencing PostNet Conv1D layers.
//          Controls the order of layer execution and coordinates with
//          postnet_stack for iterating through all NUM_LAYERS layers.
//
// Inputs:
//   - clk           : System clock
//   - rst_n         : Active-low async reset
//   - i_start       : Start signal from postnet_top
//   - i_stack_done  : Done signal from postnet_stack
//
// Outputs:
//   - o_stack_start : Start signal to postnet_stack
//   - o_layer_sel   : Current layer selection (0 to NUM_LAYERS-1)
//   - o_busy        : FSM is processing layers
//   - o_done        : All layers processed
//
// Fixed-point Format:
//   - N/A (control module only)
//==============================================================================

module postnet_fsm #(
    parameter NUM_LAYERS    = 5,            // Number of PostNet Conv1D layers
    parameter SAMPLE_LENGTH = 256           // Max samples per inference
)(
    input wire                          clk,
    input wire                          rst_n,
    
    // Control Interface (from postnet_top)
    input wire                          i_start,
    input wire                          i_stack_done,
    
    // Stack Control Interface
    output reg                          o_stack_start,
    output reg [$clog2(NUM_LAYERS)-1:0] o_layer_sel,
    
    // Status
    output reg                          o_busy,
    output reg                          o_done
);

    //==========================================================================
    // FSM State Definitions
    //==========================================================================
    localparam [2:0] ST_IDLE        = 3'd0;  // Waiting for start
    localparam [2:0] ST_INIT        = 3'd1;  // Initialize for new inference
    localparam [2:0] ST_START_LAYER = 3'd2;  // Start current layer processing
    localparam [2:0] ST_WAIT_LAYER  = 3'd3;  // Wait for layer to complete
    localparam [2:0] ST_NEXT_LAYER  = 3'd4;  // Advance to next layer
    localparam [2:0] ST_DONE        = 3'd5;  // All layers complete

    //==========================================================================
    // Internal Registers
    //==========================================================================
    reg [2:0] state_r, state_next;
    reg [$clog2(NUM_LAYERS)-1:0] layer_cnt_r;
    
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
                    state_next = ST_INIT;
            end
            
            ST_INIT: begin
                // One-cycle initialization
                state_next = ST_START_LAYER;
            end
            
            ST_START_LAYER: begin
                // Issue start pulse, move to wait
                state_next = ST_WAIT_LAYER;
            end
            
            ST_WAIT_LAYER: begin
                // Wait for stack to complete current layer
                if (i_stack_done) begin
                    if (layer_cnt_r >= NUM_LAYERS - 1)
                        state_next = ST_DONE;
                    else
                        state_next = ST_NEXT_LAYER;
                end
            end
            
            ST_NEXT_LAYER: begin
                // Increment layer counter and start next
                state_next = ST_START_LAYER;
            end
            
            ST_DONE: begin
                // Signal completion and return to idle
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
            o_stack_start <= 1'b0;
            o_layer_sel   <= 0;
            o_busy        <= 1'b0;
            o_done        <= 1'b0;
            layer_cnt_r   <= 0;
        end
        else begin
            // Default pulse signals
            o_stack_start <= 1'b0;
            o_done        <= 1'b0;
            
            case (state_r)
                ST_IDLE: begin
                    o_busy <= 1'b0;
                    if (i_start) begin
                        o_busy <= 1'b1;
                    end
                end
                
                ST_INIT: begin
                    // Reset layer counter for new inference
                    layer_cnt_r <= 0;
                    o_layer_sel <= 0;
                end
                
                ST_START_LAYER: begin
                    // Issue start pulse to stack
                    o_stack_start <= 1'b1;
                    o_layer_sel   <= layer_cnt_r;
                end
                
                ST_WAIT_LAYER: begin
                    // Wait for stack to finish - no action needed
                end
                
                ST_NEXT_LAYER: begin
                    // Advance to next layer
                    layer_cnt_r <= layer_cnt_r + 1;
                end
                
                ST_DONE: begin
                    o_busy <= 1'b0;
                    o_done <= 1'b1;
                end
            endcase
        end
    end

endmodule