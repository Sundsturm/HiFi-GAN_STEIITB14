`timescale 1ns/1ps
`include "glu_activation.v"

module glu_tb;

    reg signed [15:0] d_in;
    wire signed [15:0] d_out;
    real SCALE = 4096.0;
    integer i;

    glu_activation uut (.d_in(d_in), .d_out(d_out));

    initial begin
        $dumpfile("glu_activation_tb.vcd");
        $dumpvars(0, glu_tb);
        
        $display("==========================================================");
        $display(" GLU / SiLU Activation Test (Q4.12)");
        $display(" Formula: Output = Input * PWL_Sigmoid(Input)");
        $display("==========================================================");
        $display(" Time | Input (Float) | Sigmoid (Est) | Output (Float)");
        $display("------+---------------+---------------+---------------");

        check_point(-8192); //-2.0 ,out = 0

        check_point(-4096); // -1 , out = -0.25
        check_point(0); // 0.5, out = 0
        check_point(4096); // 1.0 , out = 0.75
        check_point(8192); // 2.0 , out = 2.0

        $display("----------------------------------------------------------");
        $display("--- SWEEP TEST (-4.0 to +4.0) ---");
        for (i = -16384; i <= 16384; i = i + 2048) begin
             d_in = i[15:0];
             #5;
             $display(" %4d | %2.4f        | ...           | %2.4f", 
                      d_in, d_in/SCALE, d_out/SCALE);
        end

        $finish;
    end

    task check_point;
        input signed [15:0] val;
        begin
            d_in = val;
            #10;
            $display(" %4t | %2.4f        | ~%2.2f          | %2.4f", 
                     $time, d_in/SCALE, (d_out/SCALE)/(d_in/SCALE + 0.0001), d_out/SCALE);
        end
    endtask

endmodule