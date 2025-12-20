module quantizer_32_16 #(
    parameter IN_WIDTH  = 32, // Lebar data dari Akumulator MAC
    parameter OUT_WIDTH = 16  // Lebar data target (Q2.14)
)(
    input wire clk,
    input wire rst_n,
    
    // Data Flow
    input wire i_valid,                  // Sinyal valid dari MAC
    input wire signed [IN_WIDTH-1:0] i_data, // Input 32-bit (Q18.14)
    
    // Output
    output reg signed [OUT_WIDTH-1:0] o_data, // Output 16-bit (Q2.14)
    output reg o_valid,                  // Data output valid
    output reg o_overflow                // Indikator jika saturasi terjadi (untuk debug)
);

    // --- 1. Definisikan Batas Minimum dan Maksimum 16-bit ---
    // Max Positive untuk 16-bit: 0000...0111111111111111 (0x7FFF)
    // Min Negative untuk 16-bit: 1111...1000000000000000 (0x8000 dalam 16bit, diperluas ke 32bit)
    
    localparam signed [IN_WIDTH-1:0] MAX_VAL = 32'h0000_7FFF; 
    localparam signed [IN_WIDTH-1:0] MIN_VAL = 32'hFFFF_8000;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            o_data <= 0;
            o_valid <= 0;
            o_overflow <= 0;
        end else begin
            // Default: tidak valid sampai ada input valid
            o_valid <= 0;
            o_overflow <= 0;

            if (i_valid) begin
                o_valid <= 1;

                // --- LOGIKA SATURASI ---
                if (i_data > MAX_VAL) begin
                    // Kasus 1: Terlalu Positif (Overflow)
                    o_data <= 16'h7FFF; // Mentok di +1.999
                    o_overflow <= 1;
                end 
                else if (i_data < MIN_VAL) begin
                    // Kasus 2: Terlalu Negatif (Underflow)
                    o_data <= 16'h8000; // Mentok di -2.000
                    o_overflow <= 1;
                end 
                else begin
                    // Kasus 3: Aman (Dalam Range)
                    // Ambil 16 bit terbawah saja (Truncation/Quantization)
                    o_data <= i_data[OUT_WIDTH-1:0];
                    o_overflow <= 0;
                end
            end
        end
    end

endmodule