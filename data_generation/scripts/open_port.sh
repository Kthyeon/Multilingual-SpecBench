# # open server 
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --model google/gemma-1.1-7b-it --port 8000 --max-model-len 2048
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server --model google/gemma-1.1-7b-it --port 8001 --max-model-len 2048
CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server --model google/gemma-1.1-7b-it --port 8002 --max-model-len 2048
CUDA_VISIBLE_DEVICES=3 python -m vllm.entrypoints.openai.api_server --model google/gemma-1.1-7b-it --port 8003 --max-model-len 2048