#!/bin/bash

# Directory containing graph files
DATASET_DIR="/home/abhijeet/datasets/medium_datasets/ecl_graphs"

# Path to executable
EXEC="./main"

# Check if executable exists
if [[ ! -f "$EXEC" ]]; then
    echo "Error: Executable $EXEC not found."
    exit 1
fi

# Loop through each file in the dataset directory
for file in "$DATASET_DIR"/*; do
    if [[ -f "$file" ]]; then
        echo "Running Gconn on: $file"
        "$EXEC" "$file"
        echo "------------------------------------------------------------"
    fi
done

echo "All runs completed."

