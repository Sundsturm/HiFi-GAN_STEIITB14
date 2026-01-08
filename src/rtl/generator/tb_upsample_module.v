`timescale 1ns / 1ps
`include "upsample_module.v"
module tb_upsample;

    // --- Parameter ---
    localparam DATA_WIDTH    = 16;
    localparam ADDR_WIDTH    = 10;
    localparam UPSAMPLE_RATE = 2; // Kita tes 2x lipat

    // --- Signals ---
    reg                   clk;
    reg                   rst_n;
    reg                   start;
    wire                  done;
    wire                  busy;
    reg  [ADDR_WIDTH-1:0] input_len;
    
    // Interface ke Memory Input (Kita simulasikan sebagai Slave)
    wire [ADDR_WIDTH-1:0] in_mem_addr;
    reg  [DATA_WIDTH-1:0] in_mem_data;
    
    // Interface ke Memory Output (Kita monitor)
    wire [ADDR_WIDTH-1:0] out_mem_addr;
    wire [DATA_WIDTH-1:0] out_mem_data;
    wire                  out_mem_we;

    // --- Instansiasi DUT (Design Under Test) ---
    upsample_module #(
        .DATA_WIDTH(DATA_WIDTH),
        .ADDR_WIDTH(ADDR_WIDTH),
        .UPSAMPLE_RATE(UPSAMPLE_RATE)
    ) uut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .done(done),
        .busy(busy),
        .input_len(input_len),
        .in_mem_addr(in_mem_addr),
        .in_mem_data(in_mem_data),
        .out_mem_addr(out_mem_addr),
        .out_mem_data(out_mem_data),
        .out_mem_we(out_mem_we)
    );

    // --- Clock Gen ---
    always #5 clk = ~clk; // 100MHz clock

    // --- Simulasi Input RAM (Simple Look-up Table) ---
    // Modul Anda mengharapkan data tersedia 1 cycle setelah address diminta
    always @(posedge clk) begin
        case (in_mem_addr)
            10'd0: in_mem_data <= 16'hAAAA; // Data ke-0
            10'd1: in_mem_data <= 16'hBBBB; // Data ke-1
            10'd2: in_mem_data <= 16'hCCCC; // Data ke-2
            10'd3: in_mem_data <= 16'hDDDD; // Data ke-3
            default: in_mem_data <= 16'h0000;
        endcase
    end

    // --- Main Stimulus ---
    initial begin
        // 1. Inisialisasi
        clk = 0;
        rst_n = 0;
        start = 0;
        input_len = 4; // Kita ingin memproses 4 data -> Harusnya jadi 8 output
        
        // 2. Reset Sequence
        #20 rst_n = 1;
        
        // 3. Start Process
        #10 start = 1;
        #10 start = 0; // Pulse start

        // 4. Tunggu sampai DONE naik
        wait(done == 1);
        
        // 5. Selesai
        #20;
        $display("Simulasi Selesai. Cek Waveform.");
        $finish;
    end

    // --- Monitor Output di Console ---
    always @(posedge clk) begin
        if (out_mem_we) begin
            $display("Time: %t | Write Addr: %d | Data: %h", $time, out_mem_addr, out_mem_data);
        end
    end
endmodule