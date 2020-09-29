import glob
import os
import argparse


parser = argparse.ArgumentParser(
        description='Convert CASIA-SURF labels into LGSC labels for only RGB images.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('input_file', type=str, help='Input CASIA-SURF labels.')
parser.add_argument('output_file', type=str, help='Output labels file.')

args = parser.parse_args()

with open(args.input_file, 'r') as file:
    new_file = open(args.output_file, "a")

    lines = file.readlines()
    for line in lines:
        item = line.strip().split(' ')
        result = item[0] + " " + item[-1] + "\n"
        new_file.write(result)
