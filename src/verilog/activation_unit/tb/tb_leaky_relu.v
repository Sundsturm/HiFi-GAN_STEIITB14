`timescale 1ns/1ps

module tb_leaky_relu;

    reg  signed [15:0] x;
    wire signed [15:0] y;

    leaky_relu_q15 dut (
        .x(x),
        .y(y)
    );

    initial begin
        $dumpfile("leaky_relu.vcd");
        $dumpvars(0, tb_leaky_relu);

        $display("Testing Leaky ReLU");
        $display("   x (dec)    ->    y (dec)");

        x = 16'sh0000;  #10;  $display("%d -> %d", x, y);   // 0
        x = 16'sh2000;  #10;  $display("%d -> %d", x, y);   // +0.25
        x = 16'sh4000;  #10;  $display("%d -> %d", x, y);   // +0.5
        x = 16'sh7FFF;  #10;  $display("%d -> %d", x, y);   // ~1
        x = -16'sh2000; #10;  $display("%d -> %d", x, y);   // -0.25
        x = -16'sh4000; #10;  $display("%d -> %d", x, y);   // -0.5
        x = -16'sh7FFF; #10;  $display("%d -> %d", x, y);   // -1

        $finish;
    end

endmodule
