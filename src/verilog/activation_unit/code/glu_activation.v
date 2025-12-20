// Simpan sebagai glu_activation.v
`timescale 1ns/1ps
`include "pwl_sigmoid.v" 

module glu_activation (
    input  wire signed [15:0] d_in,   
    output wire signed [15:0] d_out   
);

    wire signed [15:0] sigmoid_val; 

    wire signed [31:0] mult_temp;

    pwl_sigmoid u_sigmoid (
        .d_in (d_in),
        .d_out(sigmoid_val)
    );

    assign mult_temp = d_in * sigmoid_val;
    //geser bit 12 ke kanan
    assign d_out = mult_temp >>> 12;

endmodule