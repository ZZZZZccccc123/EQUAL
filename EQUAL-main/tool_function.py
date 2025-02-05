import os
import re
from s3 import *

def load_train_files(all_file_paths):
    train_file_path = []
    filepaths = []
    for file_path in all_file_paths:
        if os.path.splitext(file_path)[1] != '.jsonl' and os.path.splitext(file_path)[1] != '.txt':
            filepaths = get_folder_files(file_path)
            filepaths = load_train_files(filepaths)
        else:
            filepaths = [file_path]
        train_file_path = train_file_path + filepaths
    result = []
    for path in train_file_path:
        if os.path.splitext(path)[1] == '.jsonl':
            result.append(path)
    return result

def get_folder_files(file_path):
    file_paths = []
    if file_path.startswith("s3:"):
        return get_s3_job_embedding_file_list(file_path)
    for root, _, files in os.walk(file_path):
        for file in files:
            file_paths.append(os.path.join(root, file))
            print(os.path.join(root, file))
    return file_paths

def get_s3_job_embedding_file_list(root_dir):
    s3_key = root_dir
    s3_client = get_s3_client(s3_key)
    all_path = list_s3_objects(s3_client, s3_key, recursive=False)
    all_files = [file for file in all_path if os.path.splitext(file)[1] == '.jsonl']
    print(len(all_files))
    return all_files

def save_list(spath, my_list):
    with open(spath, "w") as f:
        for item in my_list:
            f.write(str(item) + "\n")



def read_list(spath):
    result_list = []
    with open(spath, "r") as f:
        content = f.read().strip().split("\n")
        for item in content:
            input_ids = item.split("[")[1].split("]")[0]
            labels = item.split("[")[2].split("]")[0]
            input_ids=re.sub(r','," ",input_ids)
            labels=re.sub(r','," ",labels)
            c1 = input_ids.split()
            c2 = labels.split()
            for i in range(len(c1)):
                c1[i]= int(c1[i])
            for i in range(len(c2)):
                c2[i]= int(c2[i])
            result_list.append({"input_ids":c1, "labels":c2})
    return result_list
