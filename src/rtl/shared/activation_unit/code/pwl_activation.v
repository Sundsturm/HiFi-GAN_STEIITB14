module pwl_activation(
    input  wire signed [15:0] d_in,   // Input Q4.12 Signed
    output reg  signed [15:0] d_out   // Output Q4.12 Signed (Range: -1.0 s.d 1.0)
);
    localparam signed [15:0] VAL_ONE       =  16'sd4096; //  1.0
    localparam signed [15:0] VAL_MINUS_ONE = -16'sd4096; // -1.0

    
    localparam signed [15:0] UPPER_THRESH =  16'sd2048; //  0.5
    localparam signed [15:0] LOWER_THRESH = -16'sd2048; // -0.5

    always @(*) begin
        if (d_in <= LOWER_THRESH) begin
            // Segmen 1: Saturasi Bawah (-1.0)
            d_out = VAL_MINUS_ONE;
        end 
        else if (d_in >= UPPER_THRESH) begin
            // Segmen 3: Saturasi Atas (+1.0)
            d_out = VAL_ONE;
        end 
        else begin
            // Segmen 2: Daerah Linear
            // Rumus: y = 2x
            // Logic shift left (<<< 1) tetap valid untuk Q4.12
            // Contoh: Input 0.25 (1024) -> Output 1024 << 1 = 2048 (0.5) -> Benar.
            
            d_out = d_in <<< 1;
        end
    end

endmodule