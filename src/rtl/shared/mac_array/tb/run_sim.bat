@echo off
REM =============================================================================
REM Script: run_sim.bat
REM Purpose: Compile and run MAC array testbench with IVerilog (Windows)
REM Usage: run_sim.bat
REM =============================================================================

echo =============================================================================
echo MAC Array Simulation Script (Windows)
echo =============================================================================

REM Clean previous build
echo Cleaning previous build...
if exist mac_array_sim.exe del mac_array_sim.exe
if exist hifigan_mac_array_tb.vcd del hifigan_mac_array_tb.vcd

REM Compile with IVerilog
echo Compiling with IVerilog...
iverilog -g2001 -o mac_array_sim.exe -I..\code ..\code\qmult.v ..\code\hifigan_mac_array.v hifigan_mac_array_tb.v

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Compilation failed!
    exit /b 1
)

echo [SUCCESS] Compilation complete
echo.

REM Run simulation
echo Running simulation...
echo -----------------------------------------------------------------------------
vvp mac_array_sim.exe
echo -----------------------------------------------------------------------------

REM Check if VCD file was generated
if exist hifigan_mac_array_tb.vcd (
    echo [SUCCESS] VCD file generated: hifigan_mac_array_tb.vcd
    echo.
    echo To view waveforms, run:
    echo   gtkwave hifigan_mac_array_tb.vcd
) else (
    echo [ERROR] VCD file not generated
    exit /b 1
)

echo.
echo =============================================================================
echo Simulation Complete
echo =============================================================================
pause
