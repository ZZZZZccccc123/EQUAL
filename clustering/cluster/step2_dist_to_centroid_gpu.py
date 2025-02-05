import os
import sys
import faiss
import torch
import numpy as np
import logging
logging.basicConfig(
    format='%(asctime)s %(filename)s:%(lineno)s [%(levelname)s] %(message)s', level=logging.INFO)
import base64
#from config import *
import json

def faiss_index_to_gpu(cpu_index):
    """
    Convert a Faiss CPU index to a GPU index.
    """
    ngpus = torch.cuda.device_count()
    logging.info("You have %d CUDA device(s) available:" % ngpus)

    gpu_index = faiss.index_cpu_to_all_gpus(cpu_index, ngpu=ngpus)
    return gpu_index

import io
def encode_vector(a):
    buffer = io.BytesIO()
    np.save(buffer, a, allow_pickle=False)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def decode_vector(s):
    buffer = io.BytesIO(base64.b64decode(s))
    return np.load(buffer, allow_pickle=False)


def write_allocate_result(ids, embeddings, encoded_embeddings, c_embeddings, output_dir, batch_size):
    line_count = 0
    index = faiss.IndexFlatL2(c_embeddings.shape[1])
    index.add(c_embeddings)
    # index = faiss_index_to_gpu(index)
    cluster_members = {}
    logging.info(f"searching to allocate cluster members...")
    for i_batch in range(0, embeddings.shape[0], batch_size):
        print(i_batch)
        length = min(batch_size, embeddings.shape[0]-i_batch)
        print(length)
        logging.info(
            f"searching batch : from {i_batch} to {i_batch+length}...")
        D, I = index.search(embeddings[i_batch:i_batch+length], 1)
        for i in range(length):
            if int(I[i][0]) not in cluster_members:
                cluster_members[int(I[i][0])] = []
            cluster_members[int(I[i][0])].append(
                {"id": ids[i_batch+i], "embs": encoded_embeddings[i_batch+i],
                 "c_id": int(I[i][0]), "l2_distance": float(D[i][0])})
    logging.info(f"searching all done!")
    for c_id, members in cluster_members.items():
        result_file_path = os.path.join(output_dir, f"dist-to-centroid-{c_id}.jsonl")
        with open(result_file_path, 'w') as f:
            for item in members:
                json_line = json.dumps(item) + '\n'
                f.write(json_line)
        line_count += len(members)
    logging.info(f"writing distances all done!")
    return line_count

def read_embeddings_decode(emb_file):
    logging.info(f"reading embeddings from {emb_file} ...")
    ids = []
    embeddings = []
    encoded_embeddings = []
    line_dicts = []
    with open(emb_file, 'r') as file:
        for line in file:
            data = json.loads(line)
            line_dicts.append(data)
    for line in line_dicts:
        ids.append(line['id'])
        embeddings.append(decode_vector(line['embs']))
        encoded_embeddings.append(line['embs'])
    logging.info(f"read {len(ids)} embeddings from {emb_file} done !")
    return embeddings

def semdedup_member_allocate(emb_file_paths, centroid_file_path, output_dir, batch_size):
    embeddings = torch.load(emb_file_paths)
    print(embeddings)
    embeddings = embeddings.cpu().numpy().astype(np.float32)
    print(embeddings)
    ids = [i for i in range(1459288)]
    print(embeddings.shape)
    encoded_embeddings = []
    for line in embeddings:
        encoded_embeddings.append(encode_vector(line))
    c_embeddings= read_embeddings_decode(centroid_file_path)
    c_embeddings = np.array(c_embeddings).astype(np.float32)
    line_count = write_allocate_result(ids, embeddings, encoded_embeddings, c_embeddings, output_dir, batch_size)
    return line_count

if __name__ == "__main__":
    centroid_file_path = "./AutoMathText.jsonl"
    output_dir = "./data/cluster/AutoMathText/"
    emb_file_paths = "/data/AutoMathText.pt"
    result_line_count = semdedup_member_allocate(emb_file_paths, centroid_file_path, output_dir, 10000)

