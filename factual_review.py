import json
import re
from pathlib import Path
from collections import defaultdict
import torch
# Patch missing attributes in torch for torchao compatibility
for i in range(1, 8):
    if not hasattr(torch, f"int{i}"):
        setattr(torch, f"int{i}", torch.int8)
import torch.utils._pytree
if not hasattr(torch.utils._pytree, "register_constant"):
    torch.utils._pytree.register_constant = lambda cls: cls

from transformers import AutoTokenizer, AutoModelForCausalLM

INPUT_FILE = Path("data/dataset/python_tutor_dataset_final.jsonl")
OUTPUT_FILE = Path("data/dataset/python_tutor_dataset_reviewed.jsonl")
REJECTED_FILE = Path("data/dataset/python_tutor_dataset_rejected.jsonl")
MANUAL_REVIEW_FILE = Path("data/dataset/python_tutor_dataset_manual_review.jsonl")

MODEL_NAME = "Qwen/Qwen3-1.7B"
DATA_FOLDER = Path("data/cleaned")
CHUNK_SIZE = 3000

TEST_LIMIT = None  # Set to None for full 3,912

FACT_CHECK_PROMPT = """Task: Review Python Q&A training data. Output ONLY one word.

Example 1:
SOURCE: x = 10
print(x)
QUESTION: What will this code output?
ANSWER: It will output 10 because x is assigned 10.
CATEGORY: Output prediction
LABEL: PASS

Example 2:
SOURCE: Lists are mutable sequences.
QUESTION: What is a list?
ANSWER: A list is an immutable sequence.
CATEGORY: Theory
LABEL: FACTUAL_ERROR

Example 3:
SOURCE: Python uses dynamic typing.
QUESTION: What is dynamic typing?
ANSWER: Python requires every variable to be declared as an integer before it can store a value.
CATEGORY: Theory
LABEL: FACTUAL_ERROR

Now review this:
SOURCE: {source}
QUESTION: {question}
ANSWER: {answer}
CATEGORY: {category}

Choose ONE label: PASS, FACTUAL_ERROR, MISLEADING, UNSUPPORTED, CATEGORY_MISMATCH

LABEL:"""


def split_text(text, size=CHUNK_SIZE):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            newline = text.rfind("\n", start, end)
            if newline > start:
                end = newline
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end
    return chunks


VALID_LABELS = {"PASS", "FACTUAL_ERROR", "MISLEADING", "UNSUPPORTED", "CATEGORY_MISMATCH"}

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    device_map="auto"
)
model.eval()

print("Loading source chunks...")
source_chunks = {}
txt_files = list(DATA_FOLDER.glob("*.txt"))
for txt_file in txt_files:
    text = txt_file.read_text(encoding="utf-8", errors="replace")
    chunks = split_text(text)
    for idx, chunk in enumerate(chunks, 1):
        source_chunks[(txt_file.name, idx)] = chunk
print(f"Loaded {len(source_chunks)} chunks.")

with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    lines = f.readlines()

if TEST_LIMIT is not None:
    lines = lines[:TEST_LIMIT]
print(f"Testing {len(lines)} examples...\n")

total = len(lines)
passed = []
failed = []
manual_review = []
passed_count = 0
failed_count = 0
manual_count = 0
label_counts = defaultdict(int)

for i, line in enumerate(lines, 1):
    record = json.loads(line)
    question = record['messages'][1]['content']
    answer = record['messages'][2]['content']
    category = record.get('category', '')
    source_file = record.get('source_file', '')
    chunk_id = record.get('chunk_id', 0)
    
    source_chunk = source_chunks.get((source_file, chunk_id), "Source not available")
    
    prompt = FACT_CHECK_PROMPT.format(
        source=source_chunk[:2000],
        question=question,
        answer=answer,
        category=category
    )
    
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=False
    ).to(model.device)
    
    # Only these parameters - no temperature, top_p, top_k
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    raw_result = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip().upper()
    
    found_label = None
    for label in VALID_LABELS:
        if re.search(rf"\b{re.escape(label)}\b", raw_result):
            found_label = label
            break
    
    result = found_label or "UNKNOWN"
    label_counts[result] += 1
    
    if result == "PASS":
        passed.append(record)
        passed_count += 1
        status = "PASS"
    elif result in ["FACTUAL_ERROR", "CATEGORY_MISMATCH"]:
        record['review_result'] = result
        failed.append(record)
        failed_count += 1
        status = result
    else:
        record['review_result'] = result
        manual_review.append(record)
        manual_count += 1
        status = result
    
    if i % 100 == 0 or i == total:
        print(f"  [{i:>4}/{total}] ✅{passed_count} ❌{failed_count} ⚠️{manual_count}")

# Save
for filepath, data in [
    (OUTPUT_FILE, passed),
    (REJECTED_FILE, failed),
    (MANUAL_REVIEW_FILE, manual_review)
]:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        for record in data:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

print()
print("=" * 60)
print(f"FACTUAL REVIEW RESULTS ({len(lines)} examples)")
print("=" * 60)
print(f"  ✅ PASSED:           {passed_count} ({100*passed_count/max(1,total):.0f}%)")
print(f"  ❌ REJECTED:         {failed_count} ({100*failed_count/max(1,total):.0f}%)")
print(f"  ⚠️  MANUAL REVIEW:   {manual_count} ({100*manual_count/max(1,total):.0f}%)")
print()
print("Label distribution:")
for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"  {label:<25} {count:>5}")
print()
print(f"  Reviewed:    {OUTPUT_FILE}")
print(f"  Rejected:    {REJECTED_FILE}")
print(f"  Manual:      {MANUAL_REVIEW_FILE}")
print("=" * 60)

if TEST_LIMIT is not None and TEST_LIMIT < 3912:
    print(f"\nTested {TEST_LIMIT} examples.")
    print("Set TEST_LIMIT = None for full review.")