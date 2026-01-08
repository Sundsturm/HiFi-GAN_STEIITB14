@echo off
REM Run conv1d_engine testbench simulation

echo =======================================================================
echo Conv1D Engine Testbench Simulation
echo =======================================================================

REM Clean
if exist conv1d_engine_tb.vvp del conv1d_engine_tb.vvp
if exist conv1d_engine_tb.vcd del conv1d_engine_tb.vcd

REM Compile
echo Compiling...
iverilog -o conv1d_engine_tb.vvp ^
    -I ..\..\line_buffer\code ^
    -I ..\..\mac_array\code ^
    -I ..\..\quantizer\code ^
    ..\..\line_buffer\code\line_buffer.v ^
    ..\..\mac_array\code\hifigan_mac_array.v ^
    ..\..\mac_array\code\qmult.v ^
    ..\..\quantizer\code\quantizer_32_16.v ^
    ..\code\conv1d_engine_simple.v ^
    conv1d_engine_tb.v

if %ERRORLEVEL% NEQ 0 (
    echo Compilation failed!
    pause
    exit /b 1
)

echo Compilation successful!

REM Run
echo Running simulation...
vvp conv1d_engine_tb.vvp

if %ERRORLEVEL% NEQ 0 (
    echo Simulation failed!
    pause
    exit /b 1
)

echo Simulation completed!
if exist conv1d_engine_tb.vcd (
    echo Waveform: conv1d_engine_tb.vcd
    echo View with: gtkwave conv1d_engine_tb.vcd
)

echo =======================================================================
pause
