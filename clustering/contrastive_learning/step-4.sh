cd ./FlagEmbedding

mkdir -p ./test_encoder_only_base_bge-base-en-v1.5_v1

torchrun --nproc_per_node 2 \
	-m FlagEmbedding.finetune.embedder.encoder_only.base \
	--model_name_or_path BAAI/bge-base-en-v1.5 \
    --cache_dir ./cache/model \
    --train_data ./data/score_data_ouput.jsonl \
    --cache_path ./cache/data \
    --train_group_size 8 \
    --query_max_len 512 \
    --passage_max_len 512 \
    --pad_to_multiple_of 8 \
    --query_instruction_for_retrieval 'Represent this sentence for searching relevant passages: ' \
    --query_instruction_format '{}{}' \
    --knowledge_distillation False \
	--output_dir ./test_encoder_only_base_bge-base-en-v1.5_v1 \
    --overwrite_output_dir \
    --learning_rate 1e-5 \
    --fp16 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --dataloader_drop_last True \
    --warmup_ratio 0.1 \
    --gradient_checkpointing \
    --deepspeed ./FlagEmbedding/examples/finetune/ds_stage0.json \
    --logging_steps 1 \
    --save_steps 1000 \
    --negatives_cross_device \
    --temperature 0.02 \
    --sentence_pooling_method cls \
    --normalize_embeddings True \
    --kd_loss_type kl_div 2>&1 | tee ./test_encoder_only_base_bge-base-en-v1.5_v1/train.log