cd ./FlagEmbedding/scripts
echo $(which python)

python hn_mine.py \
        --input_file ./contrastive_learning/data/new_output.jsonl \
        --output_file ./contrastive_learning/data/neg_data_ouput.jsonl \
        --range_for_sampling 2-200 \
        --negative_number 15 \
        --use_gpu_for_searching \
        --embedder_name_or_path BAAI/bge-base-en-v1.5 \
        2>&1 | tee ./contrastive_learning/log/neg_data.log