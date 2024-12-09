#!/bin/bash

# Directory containing YAML files
YAML_DIR="src/exp"
TEMPLATE_FILE="run-template.sh"

# Initialize counter
count=0

# Process each YAML file
for yaml_file in "$YAML_DIR"/*.yaml; do
    # Skip if no yaml files found
    [[ -e "$yaml_file" ]] || { echo "0" >&2; exit 0; }
    
    echo "Running $yaml_file..." >&2

    # Get the filename without path and extension
    filename=$(basename "$yaml_file" .yaml)
    
    # Create run.sh with the correct yaml path
    sed "s|{{ join_runs_path }}|exp/${filename}.yaml|g" "$TEMPLATE_FILE" > "src/run.sh" 2>&1
    
    # Make run.sh executable
    # chmod +x src/run.sh
    
    # Build the project
    ./devtool build_project >&2
    wait # Wait for build to complete
    
    # Submit the build
    python3 telerun.py submit build.tar >&2
    wait # Wait for submit to complete
    
    # Increment counter
    ((count++))
done

# Print only the counter to stdout
echo "$count"
