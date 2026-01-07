// Sigmoid Approx - Q1.15
// if x <= -0.5 -> 0.25
// if x >= +0.5 -> 0.75
// else -> 0.5 + x/2

module sigmoid_approx_q15 (
    input  wire signed [15:0] x,
    output reg  signed [15:0] y
);

    localparam signed [15:0] HALF   = 16'sh4000; // 0.5
    localparam signed [15:0] QUART  = 16'sh2000; // 0.25
    localparam signed [15:0] THREE4 = 16'sh6000; // 0.75

    always @* begin
        if (x <= -HALF)
            y = QUART;
        else if (x >= HALF)
            y = THREE4;
        else
            y = HALF + (x >>> 1);
    end
endmodule
