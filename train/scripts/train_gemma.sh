#!/usr/bin/env
# CUDA_VISIBLE_DEVICES=0,1,2,3 \
#    torchrun --nproc_per_node=4 ./train/train_gemma.py \
#     --model_name_or_path google/gemma-7b \
#     --load_model_weight True \
#     --data_path /home/taehyeon/AngMu/c4_400m.json \
#     --bf16 True \
#     --output_dir model/gemma_c4 \
#     --num_train_epochs 2 \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size 0 \
#     --gradient_accumulation_steps 32 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 1200 \
#     --save_total_limit 10 \
#     --learning_rate 3e-5 \
#     --weight_decay 0.01 \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --fsdp "full_shard auto_wrap" \
#     --fsdp_transformer_layer_cls_to_wrap 'GemmaDecoderLayer' \
#     --tf32 True \
#     --model_max_length 2048 \
#     --report_to wandb \
#     --logging_steps 5 \
#     --gradient_checkpointing True \
#     --lazy_preprocess True

# CUDA_VISIBLE_DEVICES=0,1,2,3 \
#    torchrun --nproc_per_node=4 ./train/sft_gemma.py \
#     --model_name_or_path /home/taehyeon/AngMu/model/gemma_c4/final \
#     --load_model_weight True \
#     --data_path /home/taehyeon/AngMu/gemma_sharegpt_integrated.json  \
#     --bf16 True \
#     --output_dir model/gemma_c4_sft \
#     --num_train_epochs 2 \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size 0 \
#     --gradient_accumulation_steps 16 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 190 \
#     --save_total_limit 10 \
#     --learning_rate 3e-5 \
#     --weight_decay 0.01 \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --fsdp "full_shard auto_wrap" \
#     --fsdp_transformer_layer_cls_to_wrap 'GemmaDecoderLayer' \
#     --tf32 True \
#     --model_max_length 2048 \
#     --report_to wandb \
#     --logging_steps 5 \
#     --gradient_checkpointing True \
#     --lazy_preprocess True

CUDA_VISIBLE_DEVICES=0,1,2,3 \
   torchrun --nproc_per_node=4 ./train/sft_gemma.py \
    --model_name_or_path /home/taehyeon/AngMu/model/gemma_c4_sft/final \
    --load_model_weight True \
    --data_path /home/taehyeon/AngMu/gemma_wmt16_deen_integrated.json  \
    --bf16 True \
    --output_dir model/gemma_c4_deen \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 0 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1900 \
    --save_total_limit 10 \
    --learning_rate 3e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'GemmaDecoderLayer' \
    --tf32 True \
    --model_max_length 2048 \
    --report_to wandb \
    --logging_steps 5 \
    --gradient_checkpointing True \
    --lazy_preprocess True

# CUDA_VISIBLE_DEVICES=1,2,5,6 \
#    torchrun --nproc_per_node=4 fastchat/train/train_mem.py \
#     --model_name_or_path double7/vicuna-68m \
#     --load_model_weight True \
#     --data_path ../AngMu/dataset/train/wmt_selfdistill_vicuna7b/wmt16_deen_32k_reformat_shuffle.json \
#     --bf16 True \
#     --output_dir model/vicuna_68m_wmt16_deen_32k_pretrained \
#     --num_train_epochs 3 \
#     --per_device_train_batch_size 2 \
#     --per_device_eval_batch_size 2 \
#     --gradient_accumulation_steps 16 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 1200 \
#     --save_total_limit 10 \
#     --learning_rate 2e-5 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --fsdp "full_shard auto_wrap" \
#     --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
#     --tf32 True \
#     --model_max_length 2048 \
#     --report_to wandb \
#     --run_name vicuna68m_wmt16_deen_32k \
#     --logging_steps 5 \
#     --gradient_checkpointing True \
#     --lazy_preprocess True

# CUDA_VISIBLE_DEVICES=1,2,5,6 \
#    torchrun --nproc_per_node=4 fastchat/train/train_mem.py \
#     --model_name_or_path double7/vicuna-68m \
#     --load_model_weight True \
#     --data_path ../AngMu/dataset/train/wmt_selfdistill_vicuna7b/wmt16_deen_160k_reformat_shuffle.json \
#     --bf16 True \
#     --output_dir model/vicuna_68m_wmt16_deen_160k_pretrained \
#     --num_train_epochs 3 \
#     --per_device_train_batch_size 2 \
#     --per_device_eval_batch_size 2 \
#     --gradient_accumulation_steps 16 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 1200 \
#     --save_total_limit 10 \
#     --learning_rate 2e-5 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --fsdp "full_shard auto_wrap" \
#     --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
#     --tf32 True \
#     --model_max_length 2048 \
#     --report_to wandb \
#     --run_name vicuna68m_wmt16_deen_160k \
#     --logging_steps 5 \
#     --gradient_checkpointing True \
#     --lazy_preprocess True

# CUDA_VISIBLE_DEVICES=1,2,5,6 \
#    torchrun --nproc_per_node=4 fastchat/train/train_mem.py \
#     --model_name_or_path double7/vicuna-68m \
#     --load_model_weight True \
#     --data_path ../AngMu/dataset/train/wmt_selfdistill_vicuna7b/wmt16_deen_340k_reformat_shuffle.json \
#     --bf16 True \
#     --output_dir model/vicuna_68m_wmt16_deen_340k_pretrained \
#     --num_train_epochs 3 \
#     --per_device_train_batch_size 2 \
#     --per_device_eval_batch_size 2 \
#     --gradient_accumulation_steps 16 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 15000 \
#     --save_total_limit 10 \
#     --learning_rate 2e-5 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --fsdp "full_shard auto_wrap" \
#     --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
#     --tf32 True \
#     --model_max_length 2048 \
#     --report_to wandb \
#     --run_name vicuna68m_wmt16_deen_340k \
#     --logging_steps 5 \
#     --gradient_checkpointing True \
#     --lazy_preprocess True

# CUDA_VISIBLE_DEVICES=1,2,5,6 \
#    torchrun --nproc_per_node=4 fastchat/train/train_mem.py \
#     --model_name_or_path double7/vicuna-68m \
#     --load_model_weight True \
#     --data_path ../AngMu/dataset/train/wmt_selfdistill_vicuna7b/wmt16_deen_1m_reformat_shuffle.json \
#     --bf16 True \
#     --output_dir model/vicuna_68m_wmt16_deen_1m_pretrained \
#     --num_train_epochs 3 \
#     --per_device_train_batch_size 2 \
#     --per_device_eval_batch_size 2 \
#     --gradient_accumulation_steps 16 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 30000 \
#     --save_total_limit 10 \
#     --learning_rate 2e-5 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --fsdp "full_shard auto_wrap" \
#     --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
#     --tf32 True \
#     --model_max_length 2048 \
#     --report_to wandb \
#     --run_name vicuna68m_wmt16_deen_1m \
#     --logging_steps 5 \
#     --gradient_checkpointing True \
#     --lazy_preprocess True

# # CUDA_VISIBLE_DEVICES=0 \
# #    torchrun --nproc_per_node=4 train/train_mem.py \
# #     --model_name_or_path double7/vicuna-68m \
# #     --load_model_weight True \
# #     --data_path ../AngMu/dataset/train/wmt_selfdistill_vicuna7b/wmt16_deen_3.2k_reformat_shuffle.json \
# #     --bf16 True \
# #     --output_dir model/vicuna_68m_wmt16_deen_3.2k_pretrained_test \
# #     --num_train_epochs 3 \
# #     --per_device_train_batch_size 2 \
# #     --per_device_eval_batch_size 2 \
# #     --gradient_accumulation_steps 16 \
# #     --evaluation_strategy "no" \
# #     --save_strategy "steps" \
# #     --save_steps 1200 \
# #     --save_total_limit 10 \
# #     --learning_rate 2e-5 \
# #     --weight_decay 0. \
# #     --warmup_ratio 0.03 \
# #     --lr_scheduler_type "cosine" \
# #     --logging_steps 1 \
# #     --fsdp "full_shard auto_wrap" \
# #     --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
# #     --tf32 True \
# #     --model_max_length 2048 \
# #     --report_to wandb \
# #     --run_name vicuna68m_wmt16_deen_3.2k_test \
# #     --logging_steps 5 \
# #     --gradient_checkpointing True \
# #     --lazy_preprocess True