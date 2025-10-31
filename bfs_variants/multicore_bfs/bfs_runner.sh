#!/bin/bash

# Path to the input files and output directory
# input_path="/home/graphwork/cs22s501/datasets/txt/large_graphs"
input_path="/home/graphwork/cs22s501/datasets/txt/large_graphs/modified"
# input_path="/home/graphwork/cs22s501/datasets/txt/small_graphs"
# input_path="/raid/graphwork/new/modified"

mkdir -p bfs_results

# Loop through all .txt files in the specified input directory
for file in "${input_path}"/*.txt; do
    # Inform the user which file is being processed
    echo -e "\nProcessing $file..."

    # Extract the base filename without extension
    filename=$(basename -- "$file")
    filename="${filename%.txt}"

    # Execute the parallel bcc
    ./bfs_test "$file" >> bfs_results/par_out.log

    # Assign the exit status to res
    res=$?

    # Check the value of res
    if [ $res -eq 0 ]; then
        echo "Program exited successfully for $file"
    else
        echo "Program execution failed for $file"
    fi

    echo "Processing complete."
done
