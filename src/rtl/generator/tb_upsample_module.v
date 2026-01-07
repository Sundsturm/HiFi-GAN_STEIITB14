module tb_upsample_module;
    // 1. Deklarasi Sinyal (Input = reg, Output = wire)
    reg clk, rst;
    reg [7:0] data_in;
    wire [7:0] data_out;

    // 2. Instansiasi Unit (DUT)
    modul_anda uut (
        .clk(clk),
        .rst(rst),
        .in(data_in),
        .out(data_out)
    );

    // 3. Clock Generation
    always #5 clk = ~clk;

    // 4. Stimulus (Skenario Uji)
    initial begin
        clk = 0; rst = 1; data_in = 0;
        #10 rst = 0; // Lepas reset
        
        #10 data_in = 8'd10; // Kasus 1
        #10 data_in = 8'd255; // Kasus batas (Max Value)
        
        #50 $finish;
    end
endmodule