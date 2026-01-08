@echo off
REM =======================================================================
REM Script: run_sim.bat
REM Purpose: Run line_buffer testbench simulation using Icarus Verilog
REM Usage: run_sim.bat
REM =======================================================================

echo =======================================================================
echo Line Buffer Testbench Simulation
echo =======================================================================

REM Check if iverilog is installed
where iverilog >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Error: Icarus Verilog ^(iverilog^) not found!
    echo Please install it from: http://bleyer.org/icarus/
    echo Or add it to your PATH
    pause
    exit /b 1
)

REM Clean previous build
echo Cleaning previous build...
if exist line_buffer_tb.vvp del line_buffer_tb.vvp
if exist line_buffer_tb.vcd del line_buffer_tb.vcd

REM Compile the design
echo Compiling design...
iverilog -o line_buffer_tb.vvp -I ..\code ..\code\line_buffer.v line_buffer_tb.v

if %ERRORLEVEL% NEQ 0 (
    echo Compilation failed!
    pause
    exit /b 1
)

echo Compilation successful!

REM Run simulation
echo Running simulation...
vvp line_buffer_tb.vvp

if %ERRORLEVEL% NEQ 0 (
    echo Simulation failed!
    pause
    exit /b 1
)

echo Simulation completed!

REM Check if VCD file was generated
if exist line_buffer_tb.vcd (
    echo Waveform file generated: line_buffer_tb.vcd
    echo View with: gtkwave line_buffer_tb.vcd
) else (
    echo Warning: VCD file not generated
)

echo =======================================================================
echo Simulation Complete!
echo =======================================================================
pause
