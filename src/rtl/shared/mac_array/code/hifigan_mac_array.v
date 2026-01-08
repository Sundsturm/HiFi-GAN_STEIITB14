// =============================================================================
// MODULE: hifigan_mac_array
// PURPOSE: Fixed-point MAC (Multiply-Accumulate) array for 1D convolution
//          with saturation logic to prevent overflow during accumulation.
//
// DESCRIPTION:
//   Performs spatial convolution (across kernel) and temporal accumulation 
//   (across channels) using fixed-point arithmetic with overflow protection.
//
// FIXED-POINT FORMATS:
//   - i_activations: Q4.12 format (4 integer bits, 12 fractional bits)
//   - i_weights:     Q2.14 format (2 integer bits, 14 fractional bits)
//   - Multiplication: Q6.26 format (6 integer bits, 26 fractional bits)
//   - Accumulation:   Q6.26 with saturation (prevents overflow)
//
// OPERATION:
//   1. Spatial Sum: Sum(act[i] * wgt[i]) for i=0 to KERNEL_SIZE-1
//   2. Temporal Accumulation: acc = acc + spatial_sum (with saturation)
//   3. Saturation: Clamps result to [-2^31, 2^31-1] range
//
// PARAMETERS:
//   KERNEL_SIZE: Number of elements in convolution kernel (default: 3)
//   DATA_WIDTH:  Bit width of input data (default: 16)
// =============================================================================

module hifigan_mac_array #(
    parameter KERNEL_SIZE = 3,
    parameter DATA_WIDTH  = 16 
)(
    input wire clk,
    input wire rst_n,
    
    // Control Signals
    input wire i_calc_en,      // Enable calculation
    input wire i_clear_acc,    // Clear accumulator (start new accumulation)
    
    // Data Inputs (Flattened)
    input wire signed [(KERNEL_SIZE*DATA_WIDTH)-1:0] i_activations, // Q4.12
    input wire signed [(KERNEL_SIZE*DATA_WIDTH)-1:0] i_weights,     // Q2.14
    
    // Output
    output reg signed [31:0] o_acc_raw,  // Accumulated result Q6.26 with saturation
    output reg o_valid                    // Output valid flag
);

    // ==========================================================================
    // 1. INPUT UNPACKING
    // Unpack flattened input arrays into individual elements
    // ==========================================================================
    wire signed [DATA_WIDTH-1:0] act_elem [0:KERNEL_SIZE-1];
    wire signed [DATA_WIDTH-1:0] wgt_elem [0:KERNEL_SIZE-1];
    
    genvar g;
    generate
        for (g=0; g<KERNEL_SIZE; g=g+1) begin : unpack
            assign act_elem[g] = i_activations[(g+1)*DATA_WIDTH-1 : g*DATA_WIDTH];
            assign wgt_elem[g] = i_weights[(g+1)*DATA_WIDTH-1 : g*DATA_WIDTH];
        end
    endgenerate

    // ==========================================================================
    // 2. MULTIPLICATION STAGE
    // Each element: Q4.12 * Q2.14 = Q6.26 (32-bit result)
    // ==========================================================================
    wire signed [31:0] mult_res [0:KERNEL_SIZE-1];
    
    generate
        for (g=0; g<KERNEL_SIZE; g=g+1) begin : mult_stage
            qmult u_mult (
                .i_act(act_elem[g]),  // Q4.12
                .i_wgt(wgt_elem[g]),  // Q2.14
                .o_res(mult_res[g])   // Q6.26
            );
        end
    endgenerate

    // ==========================================================================
    // 3. SPATIAL SUMMATION (Across Kernel)
    // Sum all multiplication results in combinational logic
    // ==========================================================================
    reg signed [31:0] sum_spatial;
    integer i;
    
    always @(*) begin
        sum_spatial = 0;
        for (i=0; i<KERNEL_SIZE; i=i+1) begin
            sum_spatial = sum_spatial + mult_res[i];
        end
    end

    // ==========================================================================
    // 4. TEMPORAL ACCUMULATION WITH SATURATION
    // Accumulate spatial sums across channels with overflow protection
    // Uses 33-bit intermediate for overflow detection
    // ==========================================================================
    
    // Saturation constants for Q6.26 in 32-bit signed
    localparam signed [31:0] SAT_MAX = 32'h7FFF_FFFF;  // +2^31 - 1
    localparam signed [31:0] SAT_MIN = 32'h8000_0000;  // -2^31
    
    // Internal accumulator with extra bit for overflow detection
    reg signed [32:0] acc_temp;  // 33-bit to detect overflow
    reg signed [31:0] acc_saturated;
    
    // Saturation logic (combinational)
    always @(*) begin
        // Perform addition in 33-bit space
        if (i_calc_en) begin
            if (i_clear_acc)
                acc_temp = {sum_spatial[31], sum_spatial};  // Sign extend to 33-bit
            else
                acc_temp = {o_acc_raw[31], o_acc_raw} + {sum_spatial[31], sum_spatial};
        end else begin
            acc_temp = {o_acc_raw[31], o_acc_raw};
        end
        
        // Saturate to 32-bit range
        // Check bit 32 (sign extension bit) against bit 31 (MSB of 32-bit result)
        // If they differ, overflow occurred
        if (acc_temp[32] != acc_temp[31]) begin
            // Overflow detected
            if (acc_temp[32] == 1'b1) begin
                // Negative overflow (result wrapped from negative to positive)
                // Should saturate to max negative
                acc_saturated = SAT_MIN;
            end else begin
                // Positive overflow (result wrapped from positive to negative)
                // Should saturate to max positive
                acc_saturated = SAT_MAX;
            end
        end else begin
            // No overflow
            acc_saturated = acc_temp[31:0];
        end
    end
    
    // Register the saturated result
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            o_acc_raw <= 32'sd0;
            o_valid   <= 1'b0;
        end else begin
            o_valid <= 1'b0;
            if (i_calc_en) begin
                o_acc_raw <= acc_saturated;
                o_valid   <= 1'b1;
            end
        end
    end

endmodule