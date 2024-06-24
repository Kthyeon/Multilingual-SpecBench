# evaluation
GPU_DEVICES=0
Vicuna_PATH=lmsys/vicuna-7b-v1.3
Drafter_PATH=double7/vicuna-68m #./model/vicuna68m_wmt16_deen_160k_pretrained
MODEL_NAME=vicuna_7b_v1.3
Draft_NAME=vicuna68m #_wmt16_deen_160k_pretrained

Bench_NAME=spec_bench # deen_bench, fren_bench, jaen_bench, ruen_bench, zhen_bench

temperature_array=(0.0) #(0.0 0.7 0.8 0.9 1.0)
num_choice_array=(1)

# # Firtly, run the baseline model's evaluation, which is standart of speedup ratio
# for temp in "${temperature_array[@]}"
# do
#     echo "Start eval baseline model ${MODEL_NAME} w. temperature ${temp}"
#     CUDA_VISIBLE_DEVICES=${GPU_DEVICES} \
#     python -m evaluation.inference_baseline \
#         --model-path $Vicuna_PATH  \
#         --model-id ${MODEL_NAME}-vanilla-temp-${temp} \
#         --bench-name ${Bench_NAME} \
#         --temperature $temp
#     echo "Finish eval model ${MODEL_NAME} w. temperature ${temp}"
# done

# # Run vanilla SpecDec
# for temp in "${temperature_array[@]}"
# do
#     for num_choice in "${num_choice_array[@]}"
#     do
#         echo "Start eval model ${MODEL_NAME} drafter ${Draft_NAME} w. temperature ${temp}"  
#         CUDA_VISIBLE_DEVICES=${GPU_DEVICES} \
#         python -m evaluation.inference_sps \
#             --model-path $Vicuna_PATH \
#             --drafter-path $Drafter_PATH \
#             --model-id ${MODEL_NAME}-${Draft_NAME}-temp-${temp} \
#             --bench-name ${Bench_NAME} \
#             --num-choices ${num_choice} \
#             --temperature $temp
#         echo "Finish eval model ${MODEL_NAME} w. temperature ${temp}"
#     done
# done

# # Run the specific drafter's evlauation.
# Drafter_PATH=./model/vicuna68m_wmt16_deen_160k_scratch
# Draft_NAME=vicuna68m_wmt16_deen_160k_scratch_again
# for temp in "${temperature_array[@]}"
# do
#     for num_choice in "${num_choice_array[@]}"
#     do
#         echo "Start eval model ${MODEL_NAME} drafter ${Draft_NAME} w. temperature ${temp}"  
#         CUDA_VISIBLE_DEVICES=${GPU_DEVICES} \
#         python -m evaluation.inference_sps \
#             --model-path $Vicuna_PATH \
#             --drafter-path $Drafter_PATH \
#             --model-id ${MODEL_NAME}-${Draft_NAME}-temp-${temp} \
#             --bench-name ${Bench_NAME} \
#             --num-choices ${num_choice} \
#             --temperature $temp
#         echo "Finish eval model ${MODEL_NAME} w. temperature ${temp}"
#     done
# done

# Run the specific drafter's evlauation.
Bench_NAME=deen_bench
for temp in "${temperature_array[@]}"
do
    for num_choice in "${num_choice_array[@]}"
    do
        echo "Start eval model ${MODEL_NAME} drafter ${Draft_NAME} w. temperature ${temp}"  
        CUDA_VISIBLE_DEVICES=${GPU_DEVICES} \
        python -m evaluation.inference_sps \
            --model-path $Vicuna_PATH \
            --drafter-path $Drafter_PATH \
            --model-id ${MODEL_NAME}-${Draft_NAME}-temp-${temp} \
            --bench-name ${Bench_NAME} \
            --num-choices ${num_choice} \
            --temperature $temp
        echo "Finish eval model ${MODEL_NAME} w. temperature ${temp}"
    done
done

# # eagle
# for temp in "${temperature_array[@]}"
# do
#     echo "Start eval eagle model ${MODEL_NAME} w. temperature ${temp}"  
#     CUDA_VISIBLE_DEVICES=${GPU_DEVICES} \
#     python -m evaluation.inference_eagle \
#         --ea-model-path yuhuili/EAGLE-Vicuna-7B-v1.3 \
#         --base-model-path $Vicuna_PATH \
#         --model-id eagle-${MODEL_NAME} \
#         --bench-name ${Bench_NAME} \
#         --temperature $temp
#     echo "Finish eval model ${MODEL_NAME} w. temperature ${temp}"
# done

# # medusa
# for temp in "${temperature_array[@]}"
# do
#     echo "Start eval medusa model ${MODEL_NAME} w. temperature ${temp}"  
#     CUDA_VISIBLE_DEVICES=${GPU_DEVICES} \
#     python -m evaluation.inference_medusa \
#         --model-path FasterDecoding/medusa-vicuna-7b-v1.3 \
#         --base-model $Vicuna_PATH \
#         --model-id medusa-${MODEL_NAME} \
#         --bench-name ${Bench_NAME} \
#         --temperature $temp
#     echo "Finish eval model ${MODEL_NAME} w. temperature ${temp}"
# done

# # Lookahead
# for temp in "${temperature_array[@]}"
# do
#     for num_choice in "${num_choice_array[@]}"
#     do
#         echo "Start eval medusa model ${MODEL_NAME} w. temperature ${temp}"  
#         CUDA_VISIBLE_DEVICES=${GPU_DEVICES} \
#         USE_LADE=1 python -m evaluation.inference_lookahead \
#             --model-path $Vicuna_PATH \
#             --model-id LaH-${MODEL_NAME}-lade-level-5-win-7-guess-7-temp-${temp} \
#             --level 5 \
#             --window 7 \
#             --guess 7 \
#             --bench-name ${Bench_NAME}
#         echo "Finish eval model ${MODEL_NAME} w. temperature ${temp}"
#     done
# done


# temperature_array=(0.0)

# # PLD
# for temp in "${temperature_array[@]}"
# do
#     for num_choice in "${num_choice_array[@]}"
#     do
#         echo "Start eval eagle model ${MODEL_NAME} w. temperature ${temp}"  
#         CUDA_VISIBLE_DEVICES=${GPU_DEVICES} \
#         python -m evaluation.inference_pld \
#             --model-path $Vicuna_PATH \
#             --model-id pld-${MODEL_NAME} \
#             --bench-name ${Bench_NAME}
#         echo "Finish eval model ${MODEL_NAME} w. temperature ${temp}"
#     done
# done