//////////////////////////////////////////////////////////////////////////////////
// Module: upsample_module
// Project: HiFi-GAN Generator Hardware
// 
// Purpose:
//   Melakukan upsampling pada sinyal input menggunakan metode "Repeat" 
//   (Nearest Neighbor). Membaca dari input RAM dan menulis ke output RAM.
//
// Inputs:
//   - Input Data (dari Memory/Reg Array)
//   - Konfigurasi panjang input
//
// Outputs:
//   - Output Data (ke Memory/Buffer berikutnya)
//   - Sinyal kontrol Write Enable & Address
//
// Fixed-Point Assumptions:
//   - Data bersifat agnostik (hanya menyalin bit).
//   - Tidak ada operasi aritmatika (multiply/add), hanya data movement.
//////////////////////////////////////////////////////////////////////////////////

module upsample_module #(
    parameter DATA_WIDTH    = 16,  // Lebar bit data (misal Q1.15)
    parameter ADDR_WIDTH    = 10,  // Lebar alamat memori
    parameter UPSAMPLE_RATE = 2    // Faktor upsampling (misal 2, 4, 8)
)(
    input  wire                     clk,
    input  wire                     rst_n,
    
    // Control Interface
    input  wire                     start,      // Sinyal memulai proses
    output reg                      done,       // Sinyal selesai
    output reg                      busy,       // Status busy
    
    // Configuration
    input  wire [ADDR_WIDTH-1:0]    input_len,  // Jumlah sampel input yg akan diproses
    
    // Input Memory Interface (Read Port)
    output reg  [ADDR_WIDTH-1:0]    in_mem_addr,
    input  wire [DATA_WIDTH-1:0]    in_mem_data, // Data masuk 1 cycle setelah addr diset
    
    // Output Memory Interface (Write Port)
    output reg  [ADDR_WIDTH-1:0]    out_mem_addr,
    output reg  [DATA_WIDTH-1:0]    out_mem_data,
    output reg                      out_mem_we
);

    //-------------------------------------------------------------------------
    // FSM States
    //-------------------------------------------------------------------------
    localparam S_IDLE       = 3'd0;
    localparam S_READ_ADDR  = 3'd1; // Set alamat baca
    localparam S_READ_WAIT  = 3'd2; // Tunggu data RAM valid (latency)
    localparam S_WRITE_LOOP = 3'd3; // Tulis berulang (repeat)
    localparam S_CHECK_DONE = 3'd4; // Cek apakah semua input sudah diproses
    localparam S_DONE       = 3'd5;

    reg [2:0] state, next_state;

    //-------------------------------------------------------------------------
    // Internal Counters & Registers
    //-------------------------------------------------------------------------
    reg [ADDR_WIDTH-1:0] cnt_in;      // Counter index input
    reg [ADDR_WIDTH-1:0] cnt_out;     // Counter index output
    reg [3:0]            cnt_repeat;  // Counter untuk pengulangan (asumsi max rate 16)
    reg [DATA_WIDTH-1:0] data_buffer; // Menyimpan data yang sedang dibaca

    //-------------------------------------------------------------------------
    // 1. State Register Logic
    //-------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
        end else begin
            state <= next_state;
        end
    end

    //-------------------------------------------------------------------------
    // 2. Next State Logic
    //-------------------------------------------------------------------------
    always @(*) begin
        next_state = state;
        case (state)
            S_IDLE: begin
                if (start)
                    next_state = S_READ_ADDR;
            end

            S_READ_ADDR: begin
                // Langsung pindah ke wait untuk memberi waktu RAM merespon
                next_state = S_READ_WAIT;
            end

            S_READ_WAIT: begin
                // Asumsi 1 cycle read latency selesai di sini
                next_state = S_WRITE_LOOP;
            end

            S_WRITE_LOOP: begin
                // Tulis data sebanyak UPSAMPLE_RATE kali
                // Jika counter repeat mencapai batas - 1, pindah ke cek
                if (cnt_repeat >= (UPSAMPLE_RATE - 1))
                    next_state = S_CHECK_DONE;
                else
                    next_state = S_WRITE_LOOP;
            end

            S_CHECK_DONE: begin
                // Jika index input sudah mencapai panjang yang diminta
                if (cnt_in >= (input_len - 1))
                    next_state = S_DONE;
                else
                    next_state = S_READ_ADDR;
            end

            S_DONE: begin
                // Handshake: tunggu start turun (opsional) atau langsung idle
                next_state = S_IDLE;
            end

            default: next_state = S_IDLE;
        endcase
    end

    //-------------------------------------------------------------------------
    // 3. Output & Datapath Logic
    //-------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cnt_in       <= 0;
            cnt_out      <= 0;
            cnt_repeat   <= 0;
            in_mem_addr  <= 0;
            out_mem_addr <= 0;
            out_mem_data <= 0;
            out_mem_we   <= 0;
            done         <= 0;
            busy         <= 0;
            data_buffer  <= 0;
        end else begin
            // Default signals
            out_mem_we <= 1'b0; 
            done       <= 1'b0;

            case (state)
                S_IDLE: begin
                    busy       <= 1'b0;
                    cnt_in     <= 0;
                    cnt_out    <= 0;
                    cnt_repeat <= 0;
                    if (start) busy <= 1'b1;
                end

                S_READ_ADDR: begin
                    // Setup alamat input
                    in_mem_addr <= cnt_in;
                end

                S_READ_WAIT: begin
                    // Tidak melakukan apa-apa, menunggu data valid
                    // (Data akan tersedia di in_mem_data pada akhir cycle ini)
                end

                S_WRITE_LOOP: begin
                    // Ambil data (latch) pada cycle pertama loop
                    if (cnt_repeat == 0) begin
                        data_buffer <= in_mem_data; 
                        out_mem_data <= in_mem_data; // Tulis langsung
                    end else begin
                        out_mem_data <= data_buffer; // Tulis dari buffer
                    end

                    // Lakukan penulisan
                    out_mem_addr <= cnt_out;
                    out_mem_we   <= 1'b1;
                    
                    // Increment counters
                    cnt_out    <= cnt_out + 1;
                    cnt_repeat <= cnt_repeat + 1;
                end

                S_CHECK_DONE: begin
                    // Reset repeat counter untuk input berikutnya
                    cnt_repeat <= 0;
                    
                    // Pindah ke input berikutnya jika belum selesai
                    if (cnt_in < input_len) begin
                        cnt_in <= cnt_in + 1;
                    end
                end

                S_DONE: begin
                    done <= 1'b1;
                    busy <= 1'b0;
                end
            endcase
        end
    end

endmodule