// GAU - Q1.15
// y = tanh(in_feat) * sigmoid(in_gate)

module gau_q15 (
    input  wire signed [15:0] in_feat,
    input  wire signed [15:0] in_gate,
    output wire signed [15:0] y
);

    wire signed [15:0] t, s;
    wire signed [31:0] prod;

    tanh_approx_q15    u_tanh (.x(in_feat), .y(t));
    sigmoid_approx_q15 u_sig  (.x(in_gate), .y(s));

    assign prod = t * s;
    assign y    = prod[30:15];   // scale back to Q1.15
endmodule
