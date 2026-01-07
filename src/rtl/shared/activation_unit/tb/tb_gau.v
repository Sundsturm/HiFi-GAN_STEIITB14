`timescale 1ns/1ps

module tb_gau;

    reg  signed [15:0] feat;
    reg  signed [15:0] gate;
    wire signed [15:0] y;

    gau_q15 dut (
        .in_feat(feat),
        .in_gate(gate),
        .y(y)
    );

    initial begin
        $dumpfile("gau.vcd");
        $dumpvars(0, tb_gau);

        $display("Testing GAU");
        $display(" feat   gate   ->   out");

        feat = 16'sh4000; gate = 16'sh4000; #10;   // +0.5, +0.5
        $display("%d   %d  -> %d", feat, gate, y);

        feat = 16'sh7FFF; gate = 16'sh4000; #10;   // 1 * 0.75
        $display("%d   %d  -> %d", feat, gate, y);

        feat = 16'sh2000; gate = -16'sh4000; #10;  // 0.25 * ~0.25
        $display("%d   %d  -> %d", feat, gate, y);

        feat = -16'sh4000; gate = 16'sh4000; #10;  // -0.5 * 0.75
        $display("%d   %d  -> %d", feat, gate, y);

        feat = -16'sh7FFF; gate = -16'sh7FFF; #10;
        $display("%d   %d  -> %d", feat, gate, y);

        $finish;
    end

endmodule
