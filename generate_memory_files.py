#!/usr/bin/env python3
"""
HiFi-GAN Memory File Generator
Parses weight CSV files and generates:
1. Verilog memory initialization files (.mem)
2. Address map header file (.vh)
3. Memory configuration files
"""

import csv
import numpy as np
import os
import sys
from pathlib import Path

class HiFiGANMemoryGenerator:
    def __init__(self, weights_csv, addr_map_csv, output_dir=".", bit_width=16):
        self.weights_csv = weights_csv
        self.addr_map_csv = addr_map_csv
        self.output_dir = Path(output_dir)
        self.bit_width = bit_width
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Data structures
        self.weights = {}
        self.addr_map = {}
        self.layer_info = {}
        
    def parse_address_map(self):
        """Parse the address map CSV file"""
        print("Parsing address map...")
        
        with open(self.addr_map_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                layer_name = row['Layer Name']
                self.addr_map[layer_name] = {
                    'start': int(row['Start Address']),
                    'end': int(row['End Address']),
                    'count': int(row['Count']),
                    'shape': row['Shape']
                }
                
        print(f"  Loaded {len(self.addr_map)} layer entries")
        return self.addr_map
    
    def parse_weights(self):
        """Parse the weights CSV file - handles two data parts per row"""
        print("Parsing weights...")
        
        with open(self.weights_csv, 'r') as f:
            reader = csv.DictReader(f)
            addr = 0
            
            for row in reader:
                # Each row has Data_Part_1 and Data_Part_2
                for part_key in ['Data_Part_1', 'Data_Part_2']:
                    if part_key in row and row[part_key]:
                        try:
                            value = float(row[part_key])
                            
                            # Find which layer this address belongs to
                            layer_name = self.find_layer_for_address(addr)
                            
                            if layer_name not in self.weights:
                                self.weights[layer_name] = {}
                            
                            self.weights[layer_name][addr] = value
                            addr += 1
                        except ValueError:
                            continue
        
        total_weights = sum(len(v) for v in self.weights.values())
        print(f"  Loaded {total_weights} weights from {len(self.weights)} layers")
        return self.weights
    
    def find_layer_for_address(self, addr):
        """Find which layer a given address belongs to"""
        for layer_name, info in self.addr_map.items():
            if info['start'] <= addr <= info['end']:
                return layer_name
        return 'unknown'
    
    def float_to_fixed(self, value, int_bits=2, frac_bits=14):
        """Convert float to fixed-point Q format (2's complement)"""
        # Q2.14 for weights, Q4.12 for biases
        scale = 2 ** frac_bits
        fixed = int(round(value * scale))
        
        # Saturate to bit width
        max_val = (2 ** (int_bits + frac_bits - 1)) - 1
        min_val = -(2 ** (int_bits + frac_bits - 1))
        
        if fixed > max_val:
            fixed = max_val
        elif fixed < min_val:
            fixed = min_val
        
        # Convert to unsigned representation for hex output
        if fixed < 0:
            fixed = (1 << self.bit_width) + fixed
        
        return fixed
    
    def generate_unified_mem_file(self):
        """Generate unified memory file with all weights"""
        print("\nGenerating unified memory file...")
        
        output_file = self.output_dir / "hifigan_weights.mem"
        
        # Create array for all addresses
        max_addr = max(info['end'] for info in self.addr_map.values())
        mem_array = [0] * (max_addr + 1)
        
        # Fill memory array from weights
        for layer_name, addresses in self.weights.items():
            for addr, value in addresses.items():
                # Determine if bias or weight for proper Q format
                if 'bias' in layer_name.lower():
                    fixed_val = self.float_to_fixed(value, int_bits=4, frac_bits=12)
                else:
                    fixed_val = self.float_to_fixed(value, int_bits=2, frac_bits=14)
                
                mem_array[addr] = fixed_val
        
        # Write to file in hex format
        with open(output_file, 'w') as f:
            for addr, value in enumerate(mem_array):
                f.write(f"{value:04x}\n")
        
        print(f"  Written {len(mem_array)} entries to {output_file}")
        return output_file
    
    def generate_separated_mem_files(self):
        """Generate separate weight and bias memory files"""
        print("\nGenerating separated memory files...")
        
        # Separate weights and biases
        weights_dict = {}
        biases_dict = {}
        
        for layer_name, info in self.addr_map.items():
            if 'bias' in layer_name.lower():
                biases_dict[layer_name] = info
            else:
                weights_dict[layer_name] = info
        
        # Generate weight memory file
        weight_file = self.output_dir / "weights.mem"
        weight_entries = []
        
        for layer_name, info in sorted(weights_dict.items(), key=lambda x: x[1]['start']):
            if layer_name in self.weights:
                for addr in range(info['start'], info['end'] + 1):
                    value = self.weights[layer_name].get(addr, 0.0)
                    fixed_val = self.float_to_fixed(value, int_bits=2, frac_bits=14)
                    weight_entries.append(fixed_val)
        
        with open(weight_file, 'w') as f:
            for value in weight_entries:
                f.write(f"{value:04x}\n")
        
        print(f"  Written {len(weight_entries)} weight entries to {weight_file}")
        
        # Generate bias memory file
        bias_file = self.output_dir / "biases.mem"
        bias_entries = []
        
        for layer_name, info in sorted(biases_dict.items(), key=lambda x: x[1]['start']):
            if layer_name in self.weights:
                for addr in range(info['start'], info['end'] + 1):
                    value = self.weights[layer_name].get(addr, 0.0)
                    fixed_val = self.float_to_fixed(value, int_bits=4, frac_bits=12)
                    bias_entries.append(fixed_val)
        
        with open(bias_file, 'w') as f:
            for value in bias_entries:
                f.write(f"{value:04x}\n")
        
        print(f"  Written {len(bias_entries)} bias entries to {bias_file}")
        
        return weight_file, bias_file
    
    def generate_address_map_header(self):
        """Generate Verilog header file with address map constants"""
        print("\nGenerating address map header...")
        
        output_file = self.output_dir / "hifigan_addr_map.vh"
        
        with open(output_file, 'w') as f:
            f.write("//==============================================================================\n")
            f.write("// HiFi-GAN Address Map\n")
            f.write("// Auto-generated from CSV files\n")
            f.write("// DO NOT EDIT MANUALLY\n")
            f.write("//==============================================================================\n\n")
            
            f.write("`ifndef HIFIGAN_ADDR_MAP_VH\n")
            f.write("`define HIFIGAN_ADDR_MAP_VH\n\n")
            
            # Total memory size
            max_addr = max(info['end'] for info in self.addr_map.values())
            f.write(f"// Total parameter memory size\n")
            f.write(f"localparam HIFIGAN_TOTAL_PARAMS = {max_addr + 1};\n")
            f.write(f"localparam HIFIGAN_ADDR_WIDTH   = {(max_addr + 1).bit_length()};\n\n")
            
            # Generate constants for each layer
            f.write("// Layer address ranges\n")
            for layer_name, info in sorted(self.addr_map.items(), key=lambda x: x[1]['start']):
                # Convert layer name to valid Verilog identifier
                verilog_name = layer_name.replace('.', '_').replace('[', '_').replace(']', '').upper()
                
                f.write(f"localparam {verilog_name}_START = {info['start']:6d};\n")
                f.write(f"localparam {verilog_name}_END   = {info['end']:6d};\n")
                f.write(f"localparam {verilog_name}_COUNT = {info['count']:6d}; // Shape: {info['shape']}\n")
                f.write("\n")
            
            # Generate organized sections
            f.write("//==============================================================================\n")
            f.write("// Organized by functional blocks\n")
            f.write("//==============================================================================\n\n")
            
            # Conv Pre
            f.write("// Pre-convolution layer\n")
            f.write(f"localparam CONV_PRE_START = {self.addr_map['conv_pre.bias']['start']};\n")
            f.write(f"localparam CONV_PRE_END   = {self.addr_map['conv_pre.weight_v']['end']};\n\n")
            
            # Upsamplers
            f.write("// Upsampler blocks\n")
            f.write(f"localparam UPS_START = {self.addr_map['ups.0.bias']['start']};\n")
            f.write(f"localparam UPS_END   = {self.addr_map['ups.2.weight_v']['end']};\n\n")
            
            # Residual blocks
            f.write("// Residual blocks\n")
            f.write(f"localparam RESBLOCKS_START = {self.addr_map['resblocks.0.convs.0.bias']['start']};\n")
            f.write(f"localparam RESBLOCKS_END   = {self.addr_map['resblocks.8.convs.1.weight_v']['end']};\n\n")
            
            # Conv Post
            f.write("// Post-convolution layer\n")
            f.write(f"localparam CONV_POST_START = {self.addr_map['conv_post.bias']['start']};\n")
            f.write(f"localparam CONV_POST_END   = {self.addr_map['conv_post.weight_v']['end']};\n\n")
            
            f.write("`endif // HIFIGAN_ADDR_MAP_VH\n")
        
        print(f"  Written address map to {output_file}")
        return output_file
    
    def generate_memory_config(self):
        """Generate memory configuration summary"""
        print("\nGenerating memory configuration...")
        
        output_file = self.output_dir / "memory_config.txt"
        
        with open(output_file, 'w') as f:
            f.write("HiFi-GAN Memory Configuration\n")
            f.write("=" * 80 + "\n\n")
            
            # Statistics
            total_params = sum(info['count'] for info in self.addr_map.values())
            total_weights = sum(info['count'] for name, info in self.addr_map.items() if 'bias' not in name.lower())
            total_biases = sum(info['count'] for name, info in self.addr_map.items() if 'bias' in name.lower())
            
            f.write(f"Total Parameters: {total_params:,}\n")
            f.write(f"Total Weights:    {total_weights:,}\n")
            f.write(f"Total Biases:     {total_biases:,}\n")
            f.write(f"Bit Width:        {self.bit_width}\n")
            f.write(f"Total Memory:     {total_params * self.bit_width / 8 / 1024:.2f} KB\n\n")
            
            # Layer breakdown
            f.write("Layer Breakdown:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Layer Name':<40} {'Start':>10} {'End':>10} {'Count':>10}\n")
            f.write("-" * 80 + "\n")
            
            for layer_name, info in sorted(self.addr_map.items(), key=lambda x: x[1]['start']):
                f.write(f"{layer_name:<40} {info['start']:>10} {info['end']:>10} {info['count']:>10}\n")
            
            # Functional blocks
            f.write("\n" + "=" * 80 + "\n")
            f.write("Functional Block Summary:\n")
            f.write("-" * 80 + "\n")
            
            blocks = {
                'conv_pre': [],
                'ups': [],
                'resblocks': [],
                'conv_post': []
            }
            
            for layer_name, info in self.addr_map.items():
                if layer_name.startswith('conv_pre'):
                    blocks['conv_pre'].append(info['count'])
                elif layer_name.startswith('ups'):
                    blocks['ups'].append(info['count'])
                elif layer_name.startswith('resblocks'):
                    blocks['resblocks'].append(info['count'])
                elif layer_name.startswith('conv_post'):
                    blocks['conv_post'].append(info['count'])
            
            for block_name, counts in blocks.items():
                total = sum(counts)
                f.write(f"{block_name:<20} {total:>10} params ({total*self.bit_width/8/1024:.2f} KB)\n")
        
        print(f"  Written configuration to {output_file}")
        return output_file
    
    def run(self):
        """Execute full memory generation pipeline"""
        print("=" * 80)
        print("HiFi-GAN Memory File Generator")
        print("=" * 80)
        
        try:
            # Parse inputs
            self.parse_address_map()
            
            # Check if weights CSV exists, if not create dummy data
            if not os.path.exists(self.weights_csv):
                print(f"\nWarning: Weights file not found: {self.weights_csv}")
                print("Generating dummy zero weights for structure validation...")
                self.generate_dummy_weights()
            else:
                self.parse_weights()
            
            # Generate outputs
            self.generate_address_map_header()
            self.generate_separated_mem_files()
            self.generate_unified_mem_file()
            self.generate_memory_config()
            
            print("\n" + "=" * 80)
            print("Memory file generation complete!")
            print("=" * 80)
            print(f"\nGenerated files in: {self.output_dir}")
            print("  - hifigan_addr_map.vh    (Verilog address constants)")
            print("  - weights.mem            (Weight memory initialization)")
            print("  - biases.mem             (Bias memory initialization)")
            print("  - hifigan_weights.mem    (Unified memory)")
            print("  - memory_config.txt      (Configuration summary)")
            
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        return True
    
    def generate_dummy_weights(self):
        """Generate dummy zero weights for testing structure"""
        for layer_name, info in self.addr_map.items():
            self.weights[layer_name] = {}
            for addr in range(info['start'], info['end'] + 1):
                self.weights[layer_name][addr] = 0.0

def main():
    # File paths
    script_dir = Path(__file__).parent
    weights_csv = script_dir / "HiFiGAN_Weights All Weights.csv"
    addr_map_csv = script_dir / "HiFiGAN_Weights- Address Map.csv"
    output_dir = script_dir
    
    # Allow command line overrides
    if len(sys.argv) > 1:
        weights_csv = sys.argv[1]
    if len(sys.argv) > 2:
        addr_map_csv = sys.argv[2]
    if len(sys.argv) > 3:
        output_dir = sys.argv[3]
    
    # Generate memory files
    generator = HiFiGANMemoryGenerator(weights_csv, addr_map_csv, output_dir)
    success = generator.run()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
