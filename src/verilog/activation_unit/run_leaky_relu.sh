#!/bin/bash
# Test runner for leaky_relu_q15 testbench

iverilog -o tb tb/tb_leaky_relu.v code/leaky_relu_q15.v
vvp tb
