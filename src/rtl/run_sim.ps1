# =========================================================
# HiFi-GAN RTL Simulation Script for Vivado (XSim)
# =========================================================

# 1. Konfigurasi Path File (Sesuaikan nama file jika berbeda)
# ---------------------------------------------------------
# Asumsi: tb_residual_block.v ada di root atau folder sim
$TB_FILE      = ".\tb_residual_block.v" 

# File RTL Target (Berdasarkan struktur folder gambar Anda)
$RTL_FILES    = @(
    ".\rtl\generator\residual_block.v",
    ".\rtl\shared\activation_unit\activation_unit.v",
    ".\rtl\shared\conv1d_engine\code\conv1d_engine.v"
)

# Nama Top Module di dalam Testbench
$TOP_MODULE   = "tb_residual_block"
$SNAPSHOT     = "tb_snapshot"

# 2. Cek Vivado Environment
# ---------------------------------------------------------
if (-not (Get-Command "xvlog" -ErrorAction SilentlyContinue)) {
    Write-Host "Error: Vivado CLI tools (xvlog, xelab, xsim) tidak ditemukan di PATH." -ForegroundColor Red
    Write-Host "Tips: Jalankan script ini dari 'Vivado Command Prompt' atau tambahkan bin Vivado ke Environment Variable."
    exit
}

# 3. Bersihkan File Temporary (Opsional)
# ---------------------------------------------------------
Write-Host "--- Cleaning previous run ---" -ForegroundColor Cyan
if (Test-Path "xsim.dir") { Remove-Item -Recurse -Force "xsim.dir" }
if (Test-Path "*.log")    { Remove-Item "*.log" }
if (Test-Path "*.pb")     { Remove-Item "*.pb" }
if (Test-Path "*.jou")    { Remove-Item "*.jou" }

# 4. Step 1: Parsing (XVLOG)
# ---------------------------------------------------------
Write-Host "`n--- Step 1: Parsing Verilog Files ---" -ForegroundColor Cyan

# Parse file RTL dan TB
# Menggabungkan array file menjadi string argumen
$compile_cmd = "xvlog $TB_FILE " + ($RTL_FILES -join " ")
Invoke-Expression $compile_cmd

if ($LASTEXITCODE -ne 0) {
    Write-Host "Compilation Failed!" -ForegroundColor Red
    exit
}

# 5. Step 2: Elaboration (XELAB)
# ---------------------------------------------------------
Write-Host "`n--- Step 2: Elaboration ---" -ForegroundColor Cyan
# -debug typical diperlukan agar kita bisa melihat waveform
# -s menentukan nama snapshot simulasi
xelab -debug typical -top $TOP_MODULE -snapshot $SNAPSHOT

if ($LASTEXITCODE -ne 0) {
    Write-Host "Elaboration Failed!" -ForegroundColor Red
    exit
}

# 6. Step 3: Simulation (XSIM)
# ---------------------------------------------------------
Write-Host "`n--- Step 3: Running Simulation ---" -ForegroundColor Green
# -R artinya Run All (sampai $finish)
xsim $SNAPSHOT -R

# Jika ingin membuka GUI untuk melihat waveform, ganti baris di atas dengan:
# xsim $SNAPSHOT -gui