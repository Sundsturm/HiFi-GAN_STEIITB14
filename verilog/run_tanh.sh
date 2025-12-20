#!/bin/bash
# Test runner for tanh_approx_q15 testbench

iverilog -o tb tb/tb_tanh.v src/tanh_approx_q15.v
vvp tb
