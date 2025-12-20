#!/bin/bash
# Test runner for gau_q15 testbench (uses tanh + sigmoid)

iverilog -o tb tb/tb_gau.v src/gau_q15.v src/tanh_approx_q15.v src/sigmoid_approx_q15.v
vvp tb
