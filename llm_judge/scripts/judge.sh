export OPENAI_API_KEY=${openai_api_key} # set the OpenAI API key

python llm_judge/gen_judgment.py \
    --bench-name deen_bench \
    --judge-model gpt-4o \
    --model-list vicuna_7b_v1.3-vanilla-temp-0.8

python llm_judge/show_result.py \
    --bench-name zhen_bench \
    --judge-model gpt-4o \
    --model-list vicuna_7b_v1.3-vanilla-temp-0.8

    