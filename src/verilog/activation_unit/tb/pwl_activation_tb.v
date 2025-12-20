`timescale 1ns/1ps

// Pastikan nama file ini sesuai dengan tempat Anda menyimpan modul PWL Q4.12
`include "pwl_activation.v"

module pwl_activation_tb;

    // --- Deklarasi Signal ---
    reg signed [15:0] d_in;
    wire signed [15:0] d_out;

    integer i;

    real SCALE = 4096.0;

    pwl_activation uut (
        .d_in (d_in),
        .d_out(d_out)
    );

    initial begin
        $dumpfile("pwl_activation_tb.vcd");
        $dumpvars(0, pwl_activation_tb);

        $display("=====================================================================");
        $display("TESTBENCH PWL ACTIVATION (Q4.12 Bipolar)");
        $display("Range: -1.0 s.d 1.0 | Slope: 2x | Threshold: +/- 0.5");
        $display("Scale Factor: 1.0 = 4096");
        $display("=====================================================================");
        $display(" Time | In (Raw) | In (Float) | Out (Raw)| Out (Float)| Status");
        $display("------+----------+------------+----------+------------+--------------");

        check_output("Initial");
        d_in = 16'd0; //  input 0.0, out = 0.0
        #10;
        
        d_in = 16'd1024;
        #10;
        check_output("Linear (+)"); // input 0.25 , out = 0.5

        d_in = 16'd2048;
        #10;
        check_output("Threshold (+)"); // input 0.5 , out = 1.0

        d_in = 16'd3276; 
        #10;
        check_output("Saturation (+)"); // input 0.8 , out = 1.0

        d_in = -16'd1024;
        #10;
        check_output("Linear (-)"); // input -0.25 , out = -0.5

        d_in = -16'd2048;
        #10;
        check_output("Threshold (-)"); //input -0.5 , out = -1.0

        d_in = -16'd6144; 
        #10;
        check_output("Saturation (-)"); // input -1.5 , out = -1.0

    end 
endmodule