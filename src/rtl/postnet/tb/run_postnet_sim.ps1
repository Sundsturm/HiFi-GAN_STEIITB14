# ==============================================================================
# PowerShell Simulation Script for PostNet Components (Windows)
# Purpose: Run simulations using Vivado xsim or ModelSim on Windows
# Usage: .\run_postnet_sim.ps1 -Module [stack|top|all] -Simulator [vivado|modelsim]
# ==============================================================================

param(
    [Parameter(Mandatory=$false)]
    [ValidateSet("stack", "top", "all")]
    [string]$Module = "all",
    
    [Parameter(Mandatory=$false)]
    [ValidateSet("vivado", "modelsim")]
    [string]$Simulator = "vivado"
)

# Directory setup
$RTL_DIR = ".."
$SHARED_DIR = "..\..\shared"
$TB_DIR = "."
$SIM_DIR = ".\sim_output"

# Create simulation output directory
if (-not (Test-Path $SIM_DIR)) {
    New-Item -ItemType Directory -Path $SIM_DIR | Out-Null
}

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "PostNet Simulation Script (Windows)" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Module: $Module" -ForegroundColor White
Write-Host "Simulator: $Simulator" -ForegroundColor White
Write-Host "Output Directory: $SIM_DIR" -ForegroundColor White
Write-Host "==========================================" -ForegroundColor Cyan

# ==============================================================================
# Function: Run Vivado xsim Simulation
# ==============================================================================
function Run-VivadoSim {
    param(
        [string]$Testbench,
        [string]$ModuleName
    )
    
    Write-Host "`n[Vivado xsim] Compiling $ModuleName..." -ForegroundColor Yellow
    
    Push-Location $SIM_DIR
    
    # Compile Verilog files
    $CompileFiles = @(
        "..\$Testbench.v",
        "..\..\postnet_stack.v",
        "..\..\postnet_top.v",
        "..\..\postnet_fsm.v",
        "..\..\..\shared\activation_unit\code\tanh_approx_q15.v",
        "..\..\..\shared\activation_unit\code\leaky_relu_q15.v",
        "..\..\..\shared\activation_unit\code\pwl_activation.v",
        "..\..\..\shared\mac_array\code\hifigan_mac_array.v",
        "..\..\..\shared\mac_array\code\qmult.v",
        "..\..\..\shared\memory\weight_mem.v",
        "..\..\..\shared\memory\bias_mem.v",
        "..\..\..\shared\quantizer\code\quantizer_32_16.v"
    )
    
    foreach ($file in $CompileFiles) {
        xvlog --sv $file 2>&1 | Tee-Object -Append -FilePath "${ModuleName}_compile.log"
        if ($LASTEXITCODE -ne 0) {
            Write-Host "[ERROR] Compilation failed for $file" -ForegroundColor Red
            Pop-Location
            return $false
        }
    }
    
    Write-Host "[Vivado xsim] Compilation successful" -ForegroundColor Green
    
    # Elaborate design
    Write-Host "[Vivado xsim] Elaborating design..." -ForegroundColor Yellow
    xelab -debug typical -top $Testbench -snapshot ${ModuleName}_snapshot 2>&1 | Tee-Object -Append -FilePath "${ModuleName}_elab.log"
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Elaboration failed" -ForegroundColor Red
        Pop-Location
        return $false
    }
    
    Write-Host "[Vivado xsim] Elaboration successful" -ForegroundColor Green
    
    # Run simulation
    Write-Host "[Vivado xsim] Running simulation..." -ForegroundColor Yellow
    xsim ${ModuleName}_snapshot -runall 2>&1 | Tee-Object -FilePath "${ModuleName}_sim.log"
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[Vivado xsim] Simulation completed" -ForegroundColor Green
        
        # Check for waveform database
        if (Test-Path "${ModuleName}.wdb") {
            Write-Host "Waveform saved: $SIM_DIR\${ModuleName}.wdb" -ForegroundColor Green
        }
        Pop-Location
        return $true
    }
    else {
        Write-Host "[ERROR] Simulation failed" -ForegroundColor Red
        Pop-Location
        return $false
    }
}

# ==============================================================================
# Function: Run ModelSim Simulation
# ==============================================================================
function Run-ModelsimSim {
    param(
        [string]$Testbench,
        [string]$ModuleName
    )
    
    Write-Host "`n[ModelSim] Compiling $ModuleName..." -ForegroundColor Yellow
    
    Push-Location $SIM_DIR
    
    # Create work library
    if (-not (Test-Path "work")) {
        vlib work
    }
    
    # Compile all source files
    $CompileCmd = @(
        "vlog",
        "-work", "work",
        "+incdir+$RTL_DIR",
        "+incdir+$SHARED_DIR",
        "..\$Testbench.v",
        "..\..\postnet_stack.v",
        "..\..\postnet_top.v",
        "..\..\postnet_fsm.v",
        "..\..\..\shared\activation_unit\code\tanh_approx_q15.v",
        "..\..\..\shared\activation_unit\code\leaky_relu_q15.v",
        "..\..\..\shared\activation_unit\code\pwl_activation.v",
        "..\..\..\shared\mac_array\code\hifigan_mac_array.v",
        "..\..\..\shared\mac_array\code\qmult.v",
        "..\..\..\shared\memory\weight_mem.v",
        "..\..\..\shared\memory\bias_mem.v",
        "..\..\..\shared\quantizer\code\quantizer_32_16.v"
    )
    
    & $CompileCmd[0] $CompileCmd[1..($CompileCmd.Length-1)] 2>&1 | Tee-Object -FilePath "${ModuleName}_compile.log"
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Compilation failed" -ForegroundColor Red
        Pop-Location
        return $false
    }
    
    Write-Host "[ModelSim] Compilation successful" -ForegroundColor Green
    Write-Host "[ModelSim] Running simulation..." -ForegroundColor Yellow
    
    # Run simulation
    vsim -c -do "run -all; quit" work.$Testbench 2>&1 | Tee-Object -FilePath "${ModuleName}_sim.log"
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[ModelSim] Simulation completed" -ForegroundColor Green
        Pop-Location
        return $true
    }
    else {
        Write-Host "[ERROR] Simulation failed" -ForegroundColor Red
        Pop-Location
        return $false
    }
}

# ==============================================================================
# Main Simulation Flow
# ==============================================================================
$Errors = 0

switch ($Module) {
    "stack" {
        Write-Host "`n===== Running PostNet Stack Simulation =====" -ForegroundColor Yellow
        if ($Simulator -eq "vivado") {
            $result = Run-VivadoSim -Testbench "postnet_stack_tb" -ModuleName "postnet_stack_tb"
        }
        else {
            $result = Run-ModelsimSim -Testbench "postnet_stack_tb" -ModuleName "postnet_stack_tb"
        }
        if (-not $result) { $Errors++ }
    }
    
    "top" {
        Write-Host "`n===== Running PostNet Top Simulation =====" -ForegroundColor Yellow
        if ($Simulator -eq "vivado") {
            $result = Run-VivadoSim -Testbench "postnet_top_tb" -ModuleName "postnet_top_tb"
        }
        else {
            $result = Run-ModelsimSim -Testbench "postnet_top_tb" -ModuleName "postnet_top_tb"
        }
        if (-not $result) { $Errors++ }
    }
    
    "all" {
        Write-Host "`n===== Running All PostNet Simulations =====" -ForegroundColor Yellow
        
        # Stack simulation
        Write-Host "`n--- PostNet Stack ---" -ForegroundColor Yellow
        if ($Simulator -eq "vivado") {
            $result = Run-VivadoSim -Testbench "postnet_stack_tb" -ModuleName "postnet_stack_tb"
        }
        else {
            $result = Run-ModelsimSim -Testbench "postnet_stack_tb" -ModuleName "postnet_stack_tb"
        }
        if (-not $result) { $Errors++ }
        
        # Top simulation
        Write-Host "`n--- PostNet Top ---" -ForegroundColor Yellow
        if ($Simulator -eq "vivado") {
            $result = Run-VivadoSim -Testbench "postnet_top_tb" -ModuleName "postnet_top_tb"
        }
        else {
            $result = Run-ModelsimSim -Testbench "postnet_top_tb" -ModuleName "postnet_top_tb"
        }
        if (-not $result) { $Errors++ }
    }
}

# ==============================================================================
# Summary
# ==============================================================================
Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Simulation Summary" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

if ($Errors -eq 0) {
    Write-Host "All simulations PASSED" -ForegroundColor Green
    Write-Host "Log files: $SIM_DIR\*_sim.log"
}
else {
    Write-Host "Some simulations FAILED (Errors: $Errors)" -ForegroundColor Red
    Write-Host "Check log files in: $SIM_DIR\"
}

Write-Host "==========================================" -ForegroundColor Cyan

exit $Errors
