//dipakai untuk glu_activation
module pwl_sigmoid(
    input  wire signed [15:0] d_in,  
    output reg  signed [15:0] d_out   
);

    localparam signed [15:0] VAL_ONE  = 16'sd4096; // 1.0
    localparam signed [15:0] VAL_ZERO = 16'sd0;    // 0.0

    localparam signed [15:0] UPPER_THRESH =  16'sd8192; //  +2.0
    localparam signed [15:0] LOWER_THRESH = -16'sd8192; //  -2.0

    localparam signed [15:0] BIAS_HALF = 16'sd2048;

    always @(*) begin
        if (d_in <= LOWER_THRESH) begin
            d_out = VAL_ZERO;
        end 
        else if (d_in >= UPPER_THRESH) begin
            d_out = VAL_ONE;
        end 
        else begin
            // daerah linier
            d_out = (d_in >>> 2) + BIAS_HALF;
        end
    end

endmodule