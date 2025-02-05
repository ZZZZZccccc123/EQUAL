cd ./FlagEmbedding/scripts
echo $(which python)

python add_reranker_score.py \
        --input_file ./contrastive_learning/data/neg_data_ouput.jsonl \
        --output_file ./contrastive_learning/data/score_data_ouput.jsonl \
        --reranker_name_or_path BAAI/bge-reranker-v2-m3 \
        --devices cuda:0 cuda:1 \
        --cache_dir ./cache/model \
        --reranker_query_max_length 512 \
        --reranker_max_length 1024 \
        2>&1 | tee ./contrastive_learning/log/score_data.log