module qmult_q2_14 (
    input  signed [15:0] i_a,
    input  signed [15:0] i_b,
    output signed [15:0] o_res
);
    wire signed [31:0] temp_mult;
    
    // 1. Lakukan perkalian normal
    assign temp_mult = i_a * i_b;
    
    // 2. SHIFTING LOGIC untuk Q2.14
    // ---------------------------------------------------------
    // Input A: Q2.14 (2 bit integer, 14 bit fractional)
    // Input B: Q2.14
    // Hasil Sementara (temp_mult): Q4.28 (4 bit integer, 28 bit fractional)
    //
    // Kita ingin output kembali ke format Q2.14 (14 bit fractional).
    // Maka:
    // - Buang 14 bit terbawah (bit 0 sampai 13).
    // - Ambil 14 bit berikutnya sebagai fractional baru (bit 14 sampai 27).
    // - Ambil 2 bit berikutnya sebagai integer (bit 28 sampai 29).
    //
    // Total slice: [29:14]
    // ---------------------------------------------------------
    
    assign o_res = temp_mult[29:14]; 

endmodule