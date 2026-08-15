import json, random

with open("data/dataset/python_tutor_dataset_reviewed.jsonl", "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

random.seed(42)
random.shuffle(data)
split = int(len(data) * 0.9)

train = data[:split]
val = data[split:]

with open("data/dataset/train.jsonl", "w", encoding="utf-8") as f:
    for item in train:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

with open("data/dataset/val.jsonl", "w", encoding="utf-8") as f:
    for item in val:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Train: {len(train)}, Val: {len(val)}")