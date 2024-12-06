#!/bin/bash

# Directory containing YAML files
YAML_DIR="src/exp"
TEMPLATE_FILE="run-template.sh"

# Initialize counter
count=0

# Process each YAML file
for yaml_file in "$YAML_DIR"/*.yaml; do
    echo "Running $yaml_file..." >&2

    # Get the filename without path and extension
    filename=$(basename "$yaml_file" .yaml)
    
    # Create run.sh with the correct yaml path
    sed "s|{{ join_runs_path }}|exp/${filename}.yaml|g" "$TEMPLATE_FILE" > "src/run.sh"
    
    # Make run.sh executable
    # chmod +x src/run.sh
    
    # Build the project
    ./devtool build_project
    
    # Submit the build
    python3 telerun.py submit build.tar
    
    # Increment counter
    ((count++))
done

# Print only the counter
echo $count
