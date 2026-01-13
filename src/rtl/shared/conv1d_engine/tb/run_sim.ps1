# PowerShell script to build and run conv1d_engine_bram testbench

$ErrorActionPreference = "Stop"

# Directories
$TB_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$CODE_DIR = Join-Path $TB_DIR "..\code"
$SHARED_DIR = Join-Path $TB_DIR "..\.."

# Output files
$VVP_FILE = Join-Path $TB_DIR "conv1d_engine_bram_tb.vvp"
$VCD_FILE = Join-Path $TB_DIR "conv1d_engine_bram_tb.vcd"

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "Building Conv1D BRAM Engine Testbench" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan

# Build iverilog command
$sources = @(
    "$CODE_DIR\conv1d_engine.v",
    "$SHARED_DIR\quantizer\code\quantizer_32_16.v",
    "$SHARED_DIR\mac_array\code\hifigan_mac_array.v",
    "$SHARED_DIR\mac_array\code\qmult.v",
    "$SHARED_DIR\activation_unit\code\leaky_relu_q15.v",
    "$SHARED_DIR\activation_unit\code\tanh_approx_q15.v",
    "$TB_DIR\conv1d_engine_tb.v"
)

try {
    # Compile
    & iverilog -g2005 -o $VVP_FILE @sources
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "Build successful! Running simulation..." -ForegroundColor Green
        Write-Host "=========================================" -ForegroundColor Cyan
        Write-Host ""
        
        # Run simulation
        & vvp $VVP_FILE
        
        Write-Host ""
        Write-Host "=========================================" -ForegroundColor Cyan
        Write-Host "Waveform saved to: $VCD_FILE" -ForegroundColor Green
        Write-Host "View with: gtkwave $VCD_FILE" -ForegroundColor Yellow
        Write-Host "=========================================" -ForegroundColor Cyan
    } else {
        throw "Compilation failed!"
    }
} catch {
    Write-Host ""
    Write-Host "[ERROR] $_" -ForegroundColor Red
    exit 1
}
