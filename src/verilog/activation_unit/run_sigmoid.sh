#!/bin/bash
# Test runner for sigmoid_approx_q15 testbench

iverilog -o tb tb/tb_sigmoid.v code/sigmoid_approx_q15.v
vvp tb
