#!/bin/bash

# Parse command line arguments
run_initial=false
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -c|--compile) run_initial=true ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Run initial run to compile all executables only if flag is set
if [ "$run_initial" = true ]; then
    echo "Running initial run to compile all executables"
    chmod +x initial_run.sh
    ./initial_run.sh
fi

# Copy over the template build script into src/
cp build-template.sh src/build.sh

# Run process_yamls.sh and capture the number of YAMLs processed
yaml_count=$(./process_yamls.sh)

# Parse yaml_count to ensure it's an integer
yaml_count=${yaml_count//[^0-9]/}

# Check if yaml_count is a valid number
if ! [[ "$yaml_count" =~ ^[0-9]+$ ]]; then
    echo "Error: Invalid YAML count received from process_yamls.sh"
    exit 1
fi

# echo "Processed $yaml_count YAML files"

# Run parse_results.py with the yaml count and save output to output.txt
python3 parse_results.py "$yaml_count" > output.txt

echo "Results have been saved to output.txt"
