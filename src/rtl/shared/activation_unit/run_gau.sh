#!/bin/bash
# Test runner for gau_q15 testbench (uses tanh + sigmoid)

iverilog -o tb tb/tb_gau.v code/gau_q15.v code/tanh_approx_q15.v code/sigmoid_approx_q15.v
vvp tb
