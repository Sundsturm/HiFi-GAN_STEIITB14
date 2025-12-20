module qmult (
    input  signed [15:0] i_act,   // Q4.12 (Activation)
    input  signed [15:0] i_wgt,   // Q2.14 (Weight)
    output signed [31:0] o_res    // Output Q6.26 (Full Precision)
);
    // Matematika Fixed Point:
    // (Q4.12) * (Q2.14) 
    // Integer bits: 4 + 2 = 6 bits
    // Fractional bits: 12 + 14 = 26 bits
    // Total: 32 bits (Q6.26)
    
    assign o_res = i_act * i_wgt;

endmodule