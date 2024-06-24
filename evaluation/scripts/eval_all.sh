Vicuna_PATH= # Path to the Vicuna model
Eagle_PATH= # Path to the Eagle model
Medusa_PATH= # Path to the Medusa model
Drafter_PATH= # Path to the public (former) Drafter model
OUR_PATH= # Path to the Our Drafter model
question_path=./dataset/eval/spec_bench/specbench_question.jsonl
model_answer_path=./dataset/eval/deen
datastore_PATH= # Path to the datastore
MODEL_NAME=vicuna-7b-v1.3
TEMP=0.0
GPU_DEVICES=0

bench_NAME="spec_bench"
torch_dtype="float16" # ["float32", "float64", "float16", "bfloat16"]


CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_baseline --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-vanilla-${torch_dtype}-temp-${TEMP} --temperature $TEMP --dtype $torch_dtype --answer-folder $model_answer_path --bench-name ${Bench_NAME}
CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_sps --model-path $Vicuna_PATH --drafter-path $Drafter_PATH --model-id ${MODEL_NAME}-sps-68m-${torch_dtype}-temp-${TEMP}  --temperature $TEMP --dtype $torch_dtype --answer-folder $model_answer_path --bench-name ${Bench_NAME}
CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_sps --model-path $Vicuna_PATH --drafter-path $Our_PATH --model-id ${MODEL_NAME}-ours-68m-${torch_dtype}-temp-${TEMP}  --temperature $TEMP --dtype $torch_dtype --answer-folder $model_answer_path --bench-name ${Bench_NAME}
CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_medusa --model-path $Medusa_PATH --base-model $Vicuna_PATH --model-id ${MODEL_NAME}-medusa-${torch_dtype}  --temperature $TEMP --dtype $torch_dtype --answer-folder $model_answer_path --bench-name ${Bench_NAME}
CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_eagle --ea-model-path $Eagle_PATH --base-model-path $Vicuna_PATH --model-id ${MODEL_NAME}-eagle-${torch_dtype}  --temperature $TEMP --dtype $torch_dtype --answer-folder $model_answer_path --bench-name ${Bench_NAME}
CUDA_VISIBLE_DEVICES=${GPU_DEVICES} USE_LADE=1 python -m evaluation.inference_lookahead --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-lade-level-5-win-7-guess-7-${torch_dtype} --level 5 --window 7 --guess 7 --dtype $torch_dtype --answer-folder $model_answer_path --bench-name ${Bench_NAME}
CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_pld --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-pld-${torch_dtype} --dtype $torch_dtype --answer-folder $model_answer_path --bench-name ${Bench_NAME}
