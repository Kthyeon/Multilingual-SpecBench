#!/bin/bash
data_path="./dataset/train/wmt/wmt16_train_ruen_1m.json"
output_prefix="dataset/train/wmt_selfdistill/wmt16_ruen_1m"
num_threads=256
max_tokens=1024
base_command="python ./data_generation/generate.py" 
SEQ="0.0 0.3 0.7 1.0"

# Ensure directories exist
mkdir -p $(dirname "$data_path")
mkdir -p $(dirname "$output_prefix")

# Loop for temperatures 
for temp in $SEQ; do
    output_path="${output_prefix}_${temp}.json"
    command="${base_command} --data_path ${data_path} --output_path ${output_path} --num_threads ${num_threads} --max_tokens ${max_tokens} --temperature ${temp}"

    echo "Running command: ${command}"
    $command  # Execute the command
done


# # convert to train format
python data_generation/convert_to_train.py