# Generate self-distilled data
We use vLLM to enable batched generation. We refer to the [Medusa GitHub Repository](https://github.com/FasterDecoding/Medusa).

## Shortcut
```bash
bash ./data_generation/scripts/open_port.sh
bash ./data_generation/scripts/self_distill_generation.sh
```

## Start server
```bash
python -m vllm.entrypoints.openai.api_server \
    --model YOUR_MODEL_NAME --port 8000
# Refer to `./data_generation/scripts/open_port.sh`
```

You can also start multiple servers with different ports to enable parallel generation. In `generate.py`, we scan the ports from 8000 to 8009 to find available servers. You can modify the code to use other ports. `--model` would hugging face model repo or your local model repo path.


## Generate data
The following command will let the model to continue the first prompt from each sample in `DATA_PATH`, this is suitable for models that can play both roles in a conversation (e.g., Zephyr 7B). If you want to use all prompts in each sample to repeatedly talk to the model, use `--chat` instead. `--chat` mode works for more models but may take longer time to generate due to repeated computation (welcome to contribute a better implementation).

```bash
python generate.py --data_path YOUR_DATA_PATH --output_path YOUR_OUTPUT_PATH --num_threads NUM_THREADS --max_tokens YOUR_MAX_TOKENS --temperature YOUR_TEMPERATURE
```

## (Optional) Format data and count dataset token
When you want to use generated data as training dataset, the output file will follow the ShareGPT format ([example](https://github.com/lm-sys/FastChat/blob/main/data/dummy_conversation.json)).
You can use the following command to convert the generated text to the same format.
After running the code, the whole token of the dataset is printed.
```bash
python data_generation/convert_to_train.py 
```