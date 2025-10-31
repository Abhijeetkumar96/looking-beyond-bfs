#!/bin/bash

# Base directory where the files are located
# base_dir="/home/graphwork/cs22s501/spanning_tree/batch-dynamic-spanning-tree/main_variants/SG_PR_ET/datasets/connected_datasets"
# base_dir="/raid/graphwork/datasets/new_graphs/txt"
base_dir="/home/graphwork/cs22s501/small_datasets/txt"

# Loop through all files in the base directory
for file in "$base_dir"/*; do
    echo "Executing for $file"
    ./main $file >> pr_out.log
done
