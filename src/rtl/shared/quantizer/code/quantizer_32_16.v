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
    // Q6.26 input range: -32 to +31.999 (6 bit integer)
    // Q4.12 output range: -8 to +7.999 (4 bit integer) 
    //
    // Kita mengambil bits [29:14] dari input, yang akan menjadi [15:0] output
    // Bits [31:30] adalah 2 MSB yang akan dibuang
    //
    // Untuk tidak overflow:
    // - Jika input positif (bit31=0): bits[31:29] harus 000 (nilai 0 sampai +7.999)
    // - Jika input negatif (bit31=1): bits[31:29] harus 111 (nilai -8.0 sampai -0.xxx)
    //
    // Jika bits[31:29] bukan 000 atau 111, berarti overflow
    
    wire [2:0] top_bits;
    assign top_bits = i_acc_raw[31:29];
    
    // Check if value in range
    wire in_range;
    assign in_range = (top_bits == 3'b000) || (top_bits == 3'b111);
    
    // Overflow flag
    wire is_overflow;
    assign is_overflow = !in_range;
    
    // Determine saturation direction based on MSB
    wire saturate_positive;
    assign saturate_positive = (i_acc_raw[31] == 1'b0);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            o_data <= 0;
            o_valid_out <= 0;
        end else begin
            o_valid_out <= i_valid;
            
            if (i_valid) begin
                if (is_overflow) begin
                    // Saturate based on original sign
                    if (saturate_positive) begin
                        o_data <= MAX_OUT; // Positive overflow -> +MAX
                    end else begin
                        o_data <= MIN_OUT; // Negative overflow -> MIN
                    end
                end
                else begin
                    // Safe Zone: Ambil bit [29:14]
                    // Truncate 14 LSB (lose fractional precision)
                    // Keep 4 MSB integer + 12 fractional bits
                    o_data <= i_acc_raw[29:14];
                end
            end
        end
    end

endmodule