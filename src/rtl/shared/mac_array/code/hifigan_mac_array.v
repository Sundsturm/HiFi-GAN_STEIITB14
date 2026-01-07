module hifigan_mac_array #(
    parameter KERNEL_SIZE = 3,
    parameter DATA_WIDTH  = 16 
)(
    input wire clk,
    input wire rst_n,
    
    // Control
    input wire i_calc_en,
    input wire i_clear_acc,
    
    // Inputs (Flattened)
    // i_activations diasumsikan Q4.12
    input wire signed [(KERNEL_SIZE*DATA_WIDTH)-1:0] i_activations, 
    // i_weights diasumsikan Q2.14
    input wire signed [(KERNEL_SIZE*DATA_WIDTH)-1:0] i_weights,
    
    // Output (Raw Accumulator Q6.26)
    // Kita keluarkan 32-bit penuh ke Saturator
    output reg signed [31:0] o_acc_raw, 
    output reg o_valid
);

    // --- 1. Unpacking ---
    wire signed [DATA_WIDTH-1:0] act_elem   [0:KERNEL_SIZE-1];
    wire signed [DATA_WIDTH-1:0] wgt_elem   [0:KERNEL_SIZE-1];
    
    genvar g;
    generate
        for (g=0; g<KERNEL_SIZE; g=g+1) begin : unpack
            assign act_elem[g] = i_activations[(g+1)*DATA_WIDTH-1 : g*DATA_WIDTH];
            assign wgt_elem[g] = i_weights[(g+1)*DATA_WIDTH-1 : g*DATA_WIDTH];
        end
    endgenerate

    // --- 2. Hybrid Multiplication ---
    wire signed [31:0] mult_res [0:KERNEL_SIZE-1];
    
    generate
        for (g=0; g<KERNEL_SIZE; g=g+1) begin : mult_stage
            qmult u_mult (
                .i_act(act_elem[g]), // Q4.12
                .i_wgt(wgt_elem[g]), // Q2.14
                .o_res(mult_res[g])  // Result: Q6.26
            );
        end
    endgenerate

    // --- 3. Spatial Summation ---
    reg signed [31:0] sum_spatial;
    integer i;
    
    always @(*) begin
        sum_spatial = 0;
        for (i=0; i<KERNEL_SIZE; i=i+1) begin
            sum_spatial = sum_spatial + mult_res[i];
        end
    end

    // --- 4. Temporal Accumulation ---
    // Format Accumulator: Q6.26 (32-bit)
    // Catatan: Jika menjumlahkan sangat banyak channel (misal > 256), 
    // Q6.26 mungkin butuh extension ke 40-bit untuk safety integer part.
    // Tapi untuk standard GAN, 32-bit seringkali cukup.
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            o_acc_raw <= 0;
            o_valid <= 0;
        end else begin
            o_valid <= 0;
            if (i_calc_en) begin
                if (i_clear_acc)
                    o_acc_raw <= sum_spatial;
                else
                    o_acc_raw <= o_acc_raw + sum_spatial;
                
                o_valid <= 1;
            end
        end
    end

endmodule