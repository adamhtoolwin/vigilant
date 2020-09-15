import glob
import os
import pandas as pd
import argparse

"""
Helper file to convert txt labels into csv labels. Use convert_labels.py first to convert to txt files 
    with only RGB files.
"""

parser = argparse.ArgumentParser(
        description='Convert txt labels into csv labels.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('input_file', type=str, help='Input .txt file.')
parser.add_argument('output_file', type=str, help='Output .csv file.')

args = parser.parse_args()

df = pd.read_csv(args.input_file, sep=" ")
df.to_csv(args.output_file, index=False, header=['path', 'target'])
