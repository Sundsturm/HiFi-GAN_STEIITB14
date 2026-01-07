// Tanh Approx - Q1.15
// Very light piecewise approx:
// small |x| -> linear
// medium -> ~0.8
// large  -> ~0.99

module tanh_approx_q15 (
    input  wire signed [15:0] x,
    output reg  signed [15:0] y
);

    localparam signed [15:0] TH_MID  = 16'sh4000; // 0.5
    localparam signed [15:0] TH_HIGH = 16'sh7333; // ~0.9
    localparam signed [15:0] Y_MID   = 16'sh6666; // ~0.8
    localparam signed [15:0] Y_HIGH  = 16'sh7EC8; // ~0.99

    wire signed [15:0] a = (x[15]) ? -x : x; // abs

    always @* begin
        if (a >= TH_HIGH)
            y = x[15] ? -Y_HIGH : Y_HIGH;
        else if (a >= TH_MID)
            y = x[15] ? -Y_MID  : Y_MID;
        else
            y = x;
    end
endmodule
