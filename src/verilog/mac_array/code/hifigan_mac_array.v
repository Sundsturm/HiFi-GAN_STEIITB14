module hifigan_mac_array #(
    parameter KERNEL_SIZE = 3,
    parameter DATA_WIDTH  = 16  // Tetap 16-bit
)(
    input wire clk,
    input wire rst_n,
    
    // --- Flow Control ---
    input wire i_calc_en,      // 1 = Lakukan perhitungan
    input wire i_clear_acc,    // 1 = Reset akumulator (mulai output sample baru)
    
    // --- Data Inputs (Flattened) ---
    // Input Window: [Sample n, Sample n-1, Sample n-2]
    // Format data harus sudah dikonversi ke Q2.14 sebelumnya (di Python/MATLAB)
    input wire signed [(KERNEL_SIZE*DATA_WIDTH)-1:0] i_data_window, 
    input wire signed [(KERNEL_SIZE*DATA_WIDTH)-1:0] i_weights,
    
    // --- Output ---
    output reg signed [DATA_WIDTH-1:0] o_mac_result,
    output reg o_valid
);

    // --- 1. Unpacking Array ---
    wire signed [DATA_WIDTH-1:0] data_elem   [0:KERNEL_SIZE-1];
    wire signed [DATA_WIDTH-1:0] weight_elem [0:KERNEL_SIZE-1];
    
    genvar g;
    generate
        for (g=0; g<KERNEL_SIZE; g=g+1) begin : unpack
            assign data_elem[g]   = i_data_window[(g+1)*DATA_WIDTH-1 : g*DATA_WIDTH];
            assign weight_elem[g] = i_weights[(g+1)*DATA_WIDTH-1 : g*DATA_WIDTH];
        end
    endgenerate

    // --- 2. Parallel Multiplication (Menggunakan Q2.14) ---
    wire signed [DATA_WIDTH-1:0] mult_out [0:KERNEL_SIZE-1];
    
    generate
        for (g=0; g<KERNEL_SIZE; g=g+1) begin : mult_stage
            // Instansiasi Multiplier Q2.14
            qmult_q2_14 u_qmult (
                .i_a(data_elem[g]),
                .i_b(weight_elem[g]),
                .o_res(mult_out[g])
            );
        end
    endgenerate

    // --- 3. Adder Tree (Sum of Products) ---
    // Menjumlahkan hasil perkalian dalam satu window
    reg signed [DATA_WIDTH-1:0] sum_products;
    integer i;
    
    always @(*) begin
        sum_products = 0;
        for (i=0; i<KERNEL_SIZE; i=i+1) begin
            sum_products = sum_products + mult_out[i];
        end
    end

    // --- 4. Accumulator & Saturation ---
    // Gunakan 32-bit untuk akumulasi agar tidak overflow saat menjumlahkan 128 channel
    reg signed [31:0] acc_reg;
    
    // Batas Saturasi untuk 16-bit Signed
    // Max Positive: 0111...1 (0x7FFF) -> +1.999...
    // Max Negative: 1000...0 (0x8000) -> -2.000...
    localparam signed [31:0] MAX_VAL = 32'h00007FFF;
    localparam signed [31:0] MIN_VAL = 32'hFFFF8000;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc_reg <= 0;
            o_mac_result <= 0;
            o_valid <= 0;
        end else begin
            o_valid <= 0;
            
            if (i_calc_en) begin
                if (i_clear_acc) begin
                    // Load nilai awal (channel pertama)
                    acc_reg <= sum_products; 
                end else begin
                    // Akumulasi (channel berikutnya)
                    acc_reg <= acc_reg + sum_products; 
                end
                
                // --- Logika Saturasi Output ---
                // Meskipun internal model Anda aman (Q2.14), akumulasi ratusan channel
                // KADANG bisa menembus batas sebentar. Saturasi ini adalah pengaman.
                
                if (acc_reg > MAX_VAL) 
                    o_mac_result <= 16'h7FFF; // Mentok Positif
                else if (acc_reg < MIN_VAL) 
                    o_mac_result <= 16'h8000; // Mentok Negatif
                else 
                    o_mac_result <= acc_reg[15:0]; // Data Aman, ambil 16 bit bawah
                
                o_valid <= 1;
            end
        end
    end

endmodule