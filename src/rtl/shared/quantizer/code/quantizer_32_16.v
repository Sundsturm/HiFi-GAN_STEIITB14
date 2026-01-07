module quantizer_32_16 (
    input wire clk,
    input wire rst_n,
    input wire i_valid,
    input wire signed [31:0] i_acc_raw, // Input Q6.26
    
    output reg signed [15:0] o_data,    // Output Q4.12
    output reg o_valid_out
);

    // --- Parameter Batas Q4.12 ---
    // Max: +7.999... (0x7FFF)
    // Min: -8.000... (0x8000)
    localparam signed [15:0] MAX_OUT = 16'h7FFF;
    localparam signed [15:0] MIN_OUT = 16'h8000;

    // --- Wires untuk Pengecekan Overflow ---
    // Kita ingin mengambil slice bits [29:14].
    // Bit [31] adalah sign bit asli.
    // Bit [30] dan [29] adalah bit integer yang "terancam" terbuang/berubah.
    // Jika i_acc_raw[31:29] tidak seragam (tidak semua 0 atau tidak semua 1),
    // berarti nilai integer terlalu besar untuk muat di 4 bit.
    
    wire [2:0] check_bits;
    assign check_bits = i_acc_raw[31:29];
    
    // Flags
    wire is_overflow_pos;
    wire is_overflow_neg;

    // Overflow Positif: Jika sign bit 0, tapi ada bit 1 di sisa integer atas
    // Contoh: 010... (Nilai positif besar) -> Harus disaturasi ke MAX
    assign is_overflow_pos = (i_acc_raw[31] == 0) && (check_bits != 3'b000);

    // Overflow Negatif: Jika sign bit 1, tapi ada bit 0 di sisa integer atas
    // Contoh: 101... (Nilai negatif besar) -> Harus disaturasi ke MIN
    assign is_overflow_neg = (i_acc_raw[31] == 1) && (check_bits != 3'b111);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            o_data <= 0;
            o_valid_out <= 0;
        end else begin
            o_valid_out <= i_valid;
            
            if (i_valid) begin
                if (is_overflow_pos) begin
                    o_data <= MAX_OUT; // Clamp ke +7.99
                end
                else if (is_overflow_neg) begin
                    o_data <= MIN_OUT; // Clamp ke -8.00
                end
                else begin
                    // Safe Zone: Ambil bit [29:14]
                    // Ini membuang 14 bit pecahan terbawah (Quantization)
                    // Dan mengambil 4 bit integer terbawah + 12 bit pecahan
                    o_data <= i_acc_raw[29:14];
                end
            end
        end
    end

endmodule