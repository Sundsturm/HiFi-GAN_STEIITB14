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
        $display("   x (dec)    ->    y (dec)     Expected");

        // Test zero
        x = 16'sh0000;  #10;  $display("%d -> %d     (0)", x, y);

        // Test positive values (should pass through)
        x = 16'sh1000;  #10;  $display("%d -> %d     (4096)", x, y);
        x = 16'sh2000;  #10;  $display("%d -> %d     (8192)", x, y);
        x = 16'sh4000;  #10;  $display("%d -> %d     (16384)", x, y);
        x = 16'sh7FFF;  #10;  $display("%d -> %d     (32767)", x, y);

        // Test negative values (should be divided by 8)
        x = -16'sh0800; #10;  $display("%d -> %d     (-256)", x, y);   // -2048 / 8 = -256
        x = -16'sh1000; #10;  $display("%d -> %d     (-512)", x, y);   // -4096 / 8 = -512
        x = -16'sh2000; #10;  $display("%d -> %d     (-1024)", x, y);  // -8192 / 8 = -1024
        x = -16'sh4000; #10;  $display("%d -> %d     (-2048)", x, y);  // -16384 / 8 = -2048
        x = -16'sh7FFF; #10;  $display("%d -> %d     (-4096)", x, y);  // -32767 / 8 = -4095.875 ≈ -4096

        $finish;
    end

endmodule
