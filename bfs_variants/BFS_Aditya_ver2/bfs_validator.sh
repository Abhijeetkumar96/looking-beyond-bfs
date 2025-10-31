#!/bin/bash

# Directory containing the .txt files
directory="/raid/graphwork/datasets/new_graphs/txt"

# Command executable
executable="./bfs_executable"

# Loop through all .txt files in the specified directory
for file in "$directory"/*.txt; do
    echo "Processing file: $file"

    # Execute the command with the current file as an argument
    $executable "$file" >> bfs_out.log

    # Check the return value of the last command executed
    if [ $? -eq 0 ]; then
        echo "Successfully processed file: $file"
    else
        echo "Failed to process file: $file"
        # Optionally, you can add a break to stop the loop on failure
        # break
    fi
done
