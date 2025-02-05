import json
import numpy as np
import pandas as pd
from tqdm import tqdm
all_cov = []
all_mean = []
from pathlib import Path 
#step 1 将每个数据的content也放到cluster中
folder_path = Path('./data/AutoMathText') 
train_files = ['./data/AutoMathText/' + f.name  for f in folder_path.iterdir() if f.is_file()] 
all_datasets = []
for files in train_files:
    print(files)
    datasets = []
    with open(files, 'r') as file:
        for line in file:
            data = json.loads(line)
            datasets.append(data)
    all_datasets = all_datasets + datasets
print(len(all_datasets))
print(all_datasets[0])
for i in tqdm(range(1000), desc='first'):
    cluster_datasets = []
    with open("./data/cluster/AutoMathText/dist-to-centroid-" + str(i) + ".jsonl", 'r') as file:
        for line in file:
            data = json.loads(line)
            data["text"] = all_datasets[int(data["id"])]["content"]
            #data["meta"] = all_datasets[int(data["id"])]["meta"]
            data["QA_pair"] = all_datasets[int(data["id"])]["decontaminated_qa_pairs"]
            data["doc_id"] = all_datasets[int(data["id"])]["id"]
            cluster_datasets.append(data)
    with open("./data/cluster/AutoMathText/dist-to-centroid-" + str(i) + ".jsonl", 'w') as file:
        for item in cluster_datasets:
            json.dump(item, file)
            file.write("\n")
#step 2 calculate distance between the clusters
import base64
import io
for i in tqdm(range(1000), desc='first'):
    all_datasets = []
    with open('./data/cluster/AutoMathText/dist-to-centroid-' + str(i) + '.jsonl', 'r') as file:
        for line in file:
            data = json.loads(line)
            all_datasets.append(data)
    result = []
    for embed in all_datasets:
        buffer = io.BytesIO(base64.b64decode(embed['embs']))
        buffer = np.load(buffer, allow_pickle=False)
        result.append(buffer)
    result = np.array(result)
    cov = np.cov(result, rowvar = False)
    all_cov.append(cov)
    all_mean.append(np.mean(result, axis=0))
import json

import torch
all_mean_tensor = torch.tensor(np.array(all_mean)).cuda()
all_cov_tensor = torch.tensor(np.array(all_cov)).cuda()
result = {}
for i in tqdm(range(1000), desc='second'):
    col = []
    for j in range(1000):
        difference = all_mean_tensor[i] - all_mean_tensor[j]
        d = torch.sum(difference**2)
        cluster_distance = d + torch.trace(all_cov_tensor[i] + all_cov_tensor[j] - 2 * torch.sqrt(torch.sqrt(all_cov_tensor[i])*all_cov_tensor[j]*torch.sqrt(all_cov_tensor[i])))
        col.append(cluster_distance)
    result[i] = col
for key, value in tqdm(result.items()):
        result[key] = [float(item.cpu()) for item in value]
df = pd.DataFrame(result)
df.to_csv('./data/AutoMathText-matrix.csv', index=False)
print("finished")