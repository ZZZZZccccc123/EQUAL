import os
import sys
import faiss
import numpy as np
from functools import partial
import pandas as pd
import logging
logging.basicConfig(
    format='%(asctime)s %(filename)s:%(lineno)s [%(levelname)s] %(message)s', level=logging.INFO)
import base64
import contextlib
from functools import partial
from typing import List, Union
import json
import numpy as np
import torch
from datasets import load_dataset, Dataset
from pathlib import Path 

from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm

# llama-chat model's instruction format
B_INST, E_INST = "[INST]", "[/INST]"

import multiprocessing


def embedding():
    # # Sentences we want sentence embeddings for

    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-base-en-v1.5')
    model = AutoModel.from_pretrained('BAAI/bge-base-en-v1.5')
    folder_path = Path('./data/AutoMathText/') 
    train_files = ['./data/AutoMathText/' + f.name  for f in folder_path.iterdir() if f.is_file()]
    gpu_ids_str = os.environ.get('CUDA_VISIBLE_DEVICES')
    gpu = gpu_ids_str.split(',') if gpu_ids_str else []
    device = torch.device("cpu")
    print(device)
    model.to(device)
    if isinstance(train_files, str):
        train_files = [train_files]
    processed_datasets = []
    for i in train_files:
        print(i)
        with open(i, 'r', encoding='utf-8') as file: 
            data = [json.loads(line.strip()) for line in file]
        processed_datasets = processed_datasets + data
    print(len(processed_datasets))
    print(processed_datasets[0])
    sentences = []
    for i in tqdm(range(len(processed_datasets))):
        messages = "<|user|>\n" + processed_datasets[i]["prompt"] + "\n\n<|assistant|>\n" + processed_datasets[i]["output"]
        sentences.append(messages)
    print(sentences[0])
    print("embedding...")
    result = torch.tensor([])
    result.to(device)
    print(result.device)
    length = int(len(sentences)/100)
    if len(sentences)%100 != 0:
        length = length + 1
    for i in tqdm(range(length)):
        encoded_single_input = tokenizer(sentences[i*100:min((i+1)*100,len(sentences))], padding=True, truncation=True, return_tensors='pt', max_length=512)
        encoded_single_input.to(device)
        with torch.no_grad():
            model_output = model(**encoded_single_input)
            # Perform pooling. In this case, cls pooling.
            sentence_embeddings = model_output[0][:, 0]
        # normalize embeddings
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        if i==0:
            result = sentence_embeddings
        else:
            result = torch.cat((result, sentence_embeddings), dim=0)

    print("saving...")
    torch.save(result, './data/AutoMathText.pt')
    return result





import io
def encode_vector(a):
    buffer = io.BytesIO()
    np.save(buffer, a, allow_pickle=False)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")











def kmeans_faiss(embeddings, ncentroids, niter, seed=42, min_points_per_centroid=1):
    logging.info(
        f"start faiss kmeans ! embeddings shape: {embeddings.shape} , embeddings dtype: {embeddings.dtype} , ncentroids: {ncentroids} , niter: {niter}")
    ncentroids = ncentroids
    niter = niter
    verbose = True
    d = embeddings.shape[1]
    kmeans = faiss.Kmeans(
        d,
        ncentroids,
        niter=niter,
        verbose=verbose,
        gpu=True,
        seed=seed,
        min_points_per_centroid=min_points_per_centroid,
        max_points_per_centroid=embeddings.shape[0],
    )
    kmeans.train(embeddings)
    if verbose:
        logging.info(f"kmeans.obj: {kmeans.obj}")
        logging.info(f"kmeans.iteration_stats: {kmeans.iteration_stats}")
    return kmeans


def semdedup_do_clustering():
    embeddings = torch.load("./data/AutoMathText.pt")
    embeddings = embeddings.cpu().numpy().astype(np.float32) # np.float32 is need for faiss
    logging.info("start kmeans !")
    kmeans_n_centroids = 1000
    kmeans_n_iter = 500
    kmeans_result = kmeans_faiss(embeddings, kmeans_n_centroids, kmeans_n_iter)
    clusters = kmeans_result.centroids
    logging.info("kmeans done ! ")
    logging.info(f"cluster shape: {clusters.shape}, dtype: {clusters.dtype}")
    cluster_jsons = []
    for i in range(clusters.shape[0]):
        cluster_json = {"id": i, "embs": encode_vector(clusters[i])}
        cluster_jsons.append(cluster_json)
    with open("./AutoMathText.jsonl", 'w') as f:
        for item in cluster_jsons:
            json_line = json.dumps(item) + '\n'
            f.write(json_line)
    line_count = clusters.shape[0]
    return line_count


if __name__ == "__main__":
    embedding()
    semdedup_do_clustering()