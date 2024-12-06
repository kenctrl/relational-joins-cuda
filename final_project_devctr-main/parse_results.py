#!/usr/bin/env python3

import os
import glob
import argparse
from collections import defaultdict

def parse_output_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
        lines = content.split('\n')
        
        # Initialize result dictionary
        result = {}
        
        # Parse output file name
        for line in lines:
            if 'Output file:' in line:
                result['output_name'] = line.split('/')[-1].strip()
                break
                
        # Parse join algorithm
        for line in lines:
            if 'Join algorithm:' in line:
                result['join_algorithm'] = line.split(':')[1].strip()
                break
        
        # Parse timing statistics
        for line in lines:
            if 'Sort:' in line:
                result['sort'] = float(line.split(':')[1].strip().split()[0])
            elif 'Merge:' in line:
                result['merge'] = float(line.split(':')[1].strip().split()[0])
            elif 'Materialize:' in line:
                result['materialize'] = float(line.split(':')[1].strip().split()[0])
        
        # Store all lines for comparison
        result['all_lines'] = set(lines)
        
        return result

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Parse join experiment results')
    parser.add_argument('yaml_count', type=int, help='Number of YAML files processed')
    args = parser.parse_args()
    
    # Get the last N job directories based on yaml_count
    telerun_dir = 'telerun-out'
    job_dirs = sorted(glob.glob(os.path.join(telerun_dir, '*')))[-args.yaml_count:]
    
    # Parse each output file
    results = []
    for job_dir in job_dirs:
        output_file = os.path.join(job_dir, 'output.txt')
        if os.path.exists(output_file):
            result = parse_output_file(output_file)
            results.append(result)
    
    # Print results
    for i, result in enumerate(results):
        print(f"\nExperiment {i+1}:")
        print(f"Output name: {result.get('output_name', 'N/A')}")
        print(f"Join algorithm: {result.get('join_algorithm', 'N/A')}")
        print(f"Sort: {result.get('sort', 'N/A')} ms")
        print(f"Merge: {result.get('merge', 'N/A')} ms")
        print(f"Materialize: {result.get('materialize', 'N/A')} ms")

if __name__ == "__main__":
    main()
