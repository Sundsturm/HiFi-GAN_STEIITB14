@echo off
REM Simulation script for quantizer_32_16 testbench using IVerilog + GTKWave (Windows)

echo ========================================
echo Quantizer 32-16 Simulation
echo ========================================

REM Clean previous simulation files
if exist quantizer_32_16_tb.vcd del quantizer_32_16_tb.vcd
if exist quantizer_sim.exe del quantizer_sim.exe

REM Compile with IVerilog
echo Compiling with IVerilog...
iverilog -o quantizer_sim.exe -I..\code ..\code\quantizer_32_16.v quantizer_32_16_tb.v

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Compilation failed!
    exit /b 1
)

echo Compilation successful!

REM Run simulation
echo.
echo Running simulation...
vvp quantizer_sim.exe

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Simulation failed!
    exit /b 1
)

echo.
echo Simulation complete!

REM Open GTKWave if VCD file exists
if exist quantizer_32_16_tb.vcd (
    echo.
    echo Opening GTKWave...
    start gtkwave quantizer_32_16_tb.vcd
) else (
    echo WARNING: VCD file not generated!
)

echo Done!
pause
