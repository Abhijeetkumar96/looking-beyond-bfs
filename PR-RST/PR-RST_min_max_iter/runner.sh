#!/bin/bash

# Base directory where the original datasets are located
base_dir="/raid/graphwork/datasets/new_graphs/txt"

# Log directory (make sure this exists or create it)
log_dir="/home/graphwork/cs22s501/spanning_tree/batch-dynamic-spanning-tree/PR-RST_min_max_iter/log_files"
err_dir="/home/graphwork/cs22s501/spanning_tree/batch-dynamic-spanning-tree/PR-RST_min_max_iter/err_files"

# Ensure log and error directories exist
mkdir -p "$log_dir"
mkdir -p "$err_dir"

# Loop through all files in the base directory
for file in "$base_dir"/*; do
    # Extract the basename of the file
    filename=$(basename "$file")
    filename="${filename%.*}"  # Removes the extension from filename
    log_file="$log_dir/${filename}.log"
    error_file="$err_dir/${filename}.err"

    # Display the command that will be run (useful for debugging)
    echo "Running command: build/pr_rst $file >> $log_file 2>> $error_file"

    # If you want to execute the command uncomment the following line:
    build/pr_rst "$file" >> "$log_file" 2>> "$error_file"
done
