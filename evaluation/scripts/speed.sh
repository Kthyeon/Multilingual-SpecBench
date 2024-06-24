#!/bin/bash
GPU_DEVICES=0

Tokenizer_PATH=lmsys/vicuna-7b-v1.3
Bench_NAME=fr     d
en_bench #angmu_bench # or spec_bench or deen_bench

basemodel_id=vicuna_7b_v1.3-vanilla-temp
model_id=vicuna_7b_v1.3-vicuna68m_wmt14_fren_1m_finetune_again-temp

model_path_prefix=./dataset/eval/${Bench_NAME}/model_answer
output_path_prefix=./dataset/eval/${Bench_NAME}/speed

temperature_array=(0.0 1.0) #0.7 0.8 0.9 1.0) 

for temp in "${temperature_array[@]}"
do
    echo "Start calculate speed up model ${Target_NAME} draft ${Draft_NAME} in temperature ${temp}"
    CUDA_VISIBLE_DEVICES=${GPU_DEVICES} \
    python evaluation/speed.py \
        --bench-name $Bench_NAME \
        --file-path ${model_path_prefix}/${model_id}-${temp}.jsonl \
        --base-path ${model_path_prefix}/${basemodel_id}-${temp}.jsonl \
        --tokenizer-path $Tokenizer_PATH \
        --result-path ${output_path_prefix}/${model_id}-${temp}.csv
    echo "Finish speed.py in temperature ${temp}"
done