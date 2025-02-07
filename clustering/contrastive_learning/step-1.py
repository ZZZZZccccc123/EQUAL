import json
from tqdm import tqdm
import os
import random

random.seed(42)

def load_jsonl(in_file):
    datas = []
    with open(in_file, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="load"):
            try:
                datas.append(json.loads(line))
            except Exception as e:
                print(e)
                continue
    return datas


def save_jsonl(data: list, path: str, mode='w', verbose=True) -> None:
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
    file_name = path
    with open(file_name, mode, encoding='utf-8') as f:
        if verbose:
            for line in tqdm(data, desc='save'):
                f.write(json.dumps(line, ensure_ascii=False) + '\n')
        else:
            for line in data:
                f.write(json.dumps(line, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    input_dir = "folder/path/input"
    output_path = "./data/output.jsonl"
    p = 0.05

    prepared_data = []
    for f in os.listdir(input_dir):
        if not f.endswith(".jsonl"):
            continue

        raw_data = load_jsonl(os.path.join(input_dir, f))
        
        for index, item in enumerate(raw_data):
            if not isinstance(item, dict):
                continue

            query = item.get("text", "")
            qa_pairs = item.get("decontaminated_qa_pairs", [])
            pos = []

            for pair in qa_pairs:
                prompt = pair.get("prompt", "").strip()
                output = pair.get("output", "").strip()
                combined = f"<|user|>\n{prompt}\n\n<|assistant|>\n{output}"
                pos.append(combined)

            new_data = {
                "query": query,
                "pos": pos
            }
            if len(pos) > 0:
                prepared_data.append(new_data)
    
    selected_num = int(len(prepared_data) * p)
    prepared_data = random.sample(prepared_data, selected_num)
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_jsonl(prepared_data, output_path)
    print(f"saved as {output_path}，total {len(prepared_data)} data samples")
    print(f"data format:")
    print(json.dumps(prepared_data[0], ensure_ascii=False, indent=4))