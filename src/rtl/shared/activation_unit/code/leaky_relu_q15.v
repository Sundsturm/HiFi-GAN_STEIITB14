// Leaky ReLU - Q1.15
// y = x            , x >= 0
// y = x / 8 (shift), x <  0

module leaky_relu_q15 (
    input  wire signed [15:0] x,
    output reg  signed [15:0] y
);
    always @* begin
        if (x >= 0)
            y = x;
        else
            y = x >>> 3;   // divide by 8
    end
endmodule