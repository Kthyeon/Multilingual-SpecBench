"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import argparse
from evaluation.eval import run_eval

from fastchat.utils import str_to_torch_dtype

from model.hydra.utils import *
from model.hydra.hydra_model import HydraModel
from model.hydra.kv_cache import initialize_past_key_values
from model.hydra.hydra_choices import *

def hydra_forward(inputs, model, tokenizer, max_new_tokens, hydra_choices=None, temperature=0.0, posterior_threshold=0.09, posterior_alpha=0.3, max_steps=512):
    input_ids = inputs.input_ids
    assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
    # Avoid modifying the input_ids in-place
    input_ids = input_ids.clone()
    accept_length_list = []

    # Cache hydra buffers (the fixed patterns for tree attention)
    if hasattr(model, "hydra_choices") and model.hydra_choices == hydra_choices:
        # Load the cached hydra buffer
        hydra_buffers = model.hydra_buffers
    else:
        # Initialize the hydra buffer
        hydra_buffers = generate_hydra_buffers(
            hydra_choices, device=model.base_model.device
        )
    model.hydra_buffers = hydra_buffers
    model.hydra_choices = hydra_choices

    # Initialize the past key and value states
    if hasattr(model, "past_key_values"):
        past_key_values = model.past_key_values
        past_key_values_data = model.past_key_values_data
        current_length_data = model.current_length_data
        # Reset the past key and value states
        current_length_data.zero_()
    else:
        (
            past_key_values,
            past_key_values_data,
            current_length_data,
        ) = initialize_past_key_values(model.base_model, model.hydra_head_arch)
        model.past_key_values = past_key_values
        model.past_key_values_data = past_key_values_data
        model.current_length_data = current_length_data

    input_len = input_ids.shape[1]
    cur_length = input_len
    reset_hydra_mode(model)
    hidden_states, logits = initialize_hydra(
        input_ids, model, hydra_buffers["hydra_attn_mask"], past_key_values, hydra_buffers["proposal_cross_attn_masks"]
    )
    new_token = 0

    for idx in range(max_steps): 
        to_pass_input_ids = None
        if idx == 0:
            to_pass_input_ids = input_ids
        candidates, tree_candidates = model.hydra_head.proposal(logits, hidden_states, hydra_buffers, past_key_values, to_pass_input_ids)
        hidden_states, logits = tree_decoding(
                model,
                tree_candidates,
                past_key_values,
                hydra_buffers["hydra_position_ids"],
                input_ids,
                hydra_buffers["retrieve_indices"],
            )
        best_candidate, accept_length = evaluate_posterior(
                logits, candidates, temperature, posterior_threshold, posterior_alpha, hydra_buffers["max_accepts"]
            )
        input_ids, logits, hidden_states, new_token = update_inference_inputs(
            input_ids,
            candidates,
            best_candidate,
            accept_length,
            hydra_buffers["retrieve_indices"],
            logits,
            hidden_states,
            new_token,
            past_key_values_data,
            current_length_data,
            model.hydra_head_arch
        )
        accept_length_tree = input_ids.shape[1] - cur_length
        cur_length = accept_length_tree + cur_length
        accept_length_list.append(accept_length_tree)
        if tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
            break
        if new_token > max_new_tokens:
            break
    return input_ids, new_token, idx+1, accept_length_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument("--base-model", type=str, default="lmsys/vicuna-7b-v1.3")
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=512,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="The temperature for hydra sampling.",
    )
    parser.add_argument(
        "--posterior-threshold",
        type=float,
        default=0.09,
        help="The posterior threshold for hydra sampling.",
    )
    parser.add_argument(
        "--posterior-alpha",
        type=float,
        default=0.3,
        help="The posterior alpha for hydra sampling.",
    )
    parser.add_argument(
        "--hydra-choices",
        type=str,
        default="mc_sim_7b_63",
        help="The hydra choices for hydra sampling.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float32", "float64", "float16", "bfloat16"],
        help="Override the default dtype. If not set, it will use float16 on GPU.",
    )

    args = parser.parse_args()

    args.hydra_choices = eval(args.hydra_choices)

    question_file = f"data/{args.bench_name}/question.jsonl"
    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"data/{args.bench_name}/model_answer/{args.model_id}.jsonl"

    print(f"Output to {answer_file}")

    model = HydraModel.from_pretrained(
        args.model_path,
        args.base_model,
        torch_dtype=str_to_torch_dtype(args.dtype),
        low_cpu_mem_usage=True,
        device_map="auto"
    )

    tokenizer = model.get_tokenizer()

    run_eval(
        model=model,
        tokenizer=tokenizer,
        forward_func=hydra_forward,
        model_id=args.model_id,
        question_file=question_file,
        question_begin=args.question_begin,
        question_end=args.question_end,
        answer_file=answer_file,
        max_new_tokens=args.max_new_tokens,
        num_choices=args.num_choices,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        hydra_choices=args.hydra_choices,
        max_steps=args.max_steps,
        temperature=args.temperature,
        posterior_threshold=args.posterior_threshold,
        posterior_alpha=args.posterior_alpha,
    )
