`timescale 1ns/1ps

module tb_sigmoid;

    reg  signed [15:0] x;
    wire signed [15:0] y;

    sigmoid_approx_q15 dut (
        .x(x),
        .y(y)
    );

    initial begin
        $dumpfile("sigmoid.vcd");
        $dumpvars(0, tb_sigmoid);

        $display("Testing Sigmoid Approx");
        $display("   x (dec)    ->    y (dec)");

        x = -16'sh6000; #10; $display("%d -> %d", x, y);   // << -0.5
        x = -16'sh4000; #10; $display("%d -> %d", x, y);   // ~ -0.5
        x =  16'sh0000; #10; $display("%d -> %d", x, y);   // 0
        x =  16'sh2000; #10; $display("%d -> %d", x, y);   // +0.25
        x =  16'sh4000; #10; $display("%d -> %d", x, y);   // +0.5
        x =  16'sh6000; #10; $display("%d -> %d", x, y);   // >> 0.5

        $finish;
    end

endmodule
