import json
import re
import ast
from pathlib import Path

import torch
# Patch missing attributes in torch for torchao compatibility
for i in range(1, 8):
    if not hasattr(torch, f"int{i}"):
        setattr(torch, f"int{i}", torch.int8)
import torch.utils._pytree
if not hasattr(torch.utils._pytree, "register_constant"):
    torch.utils._pytree.register_constant = lambda cls: cls

from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "Qwen/Qwen3-1.7B"

INPUT_FOLDER = Path("data/cleaned")
OUTPUT_FOLDER = Path("data/dataset")
OUTPUT_FILE = OUTPUT_FOLDER / "python_tutor_dataset.jsonl"

MAX_TEST_CHUNKS = 99999999
CHUNK_SIZE = 3000
MAX_NEW_TOKENS = 500
BATCH_SIZE = 4
MAX_REGENERATION_ATTEMPTS = 0

SYSTEM_PROMPT = (
    "You are an expert Python programming tutor. "
    "Give accurate, clear, technically grounded explanations. "
    "When including Python code, ALWAYS use proper triple-backtick Markdown fences with the python language tag. "
    "Format code blocks EXACTLY as:\n```python\n# your code here\n```\n"
    "Never use single backticks for code blocks. "
    "Never add meaningless or decorative code. "
    "Be factually precise - avoid oversimplifications that could mislead learners."
)

ALL_CATEGORIES = [
    "Theory",
    "Deep theory",
    "Conceptual",
    "Code explanation",
    "Output prediction",
    "Code writing",
    "Debugging",
    "Why",
    "What",
    "How",
    "When",
    "Comparison",
    "Trade-offs",
    "Scenario-based",
    "Research-level",
    "Edge cases",
    "Output + explanation",
    "Code + theory",
    "Performance",
    "Design",
    "Interview-style"
]

CATEGORIES_REQUIRING_CODE = [
    "Output prediction",
    "Output + explanation",
    "Code explanation",
    "Code writing",
    "Debugging",
    "Code + theory"
]

CATEGORY_COUNTS = {cat: 0 for cat in ALL_CATEGORIES}
CATEGORY_CYCLE = ALL_CATEGORIES.copy()
CYCLE_INDEX = 0

SEEN_QUESTIONS = set()
REJECTED_MALFORMED = 0
REJECTED_DUPLICATE = 0
REJECTED_NO_CODE = 0
REJECTED_SHORT = 0
REJECTED_CATEGORY_LABEL = 0
REJECTED_UNKNOWN_CATEGORY = 0
REJECTED_BAD_CODE_FORMAT = 0


OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    device_map="auto"
)
model.eval()
print("Model loaded successfully.")
print("Device:", model.device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


def get_next_categories(num_categories=3):
    global CYCLE_INDEX
    selected = []
    for _ in range(num_categories):
        category = CATEGORY_CYCLE[CYCLE_INDEX % len(CATEGORY_CYCLE)]
        selected.append(category)
        CYCLE_INDEX += 1
    return selected


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


def make_prompt(text, target_categories):
    cat1, cat2, cat3 = target_categories
    
    user_content = (
        "Read the following Python source material carefully.\n\n"
        "SOURCE:\n"
        + text + "\n\n"
        f"Generate exactly 3 question-answer pairs using these SPECIFIC categories:\n"
        f"1. {cat1}\n"
        f"2. {cat2}\n"
        f"3. {cat3}\n\n"
        "CRITICAL: Write NATURAL questions. Do NOT use category names as questions.\n"
        "Example: Write 'What is a variable in Python?' NOT 'Theory'\n\n"
        "CODE FORMAT RULES:\n"
        "Code is REQUIRED for: Output prediction, Output + explanation, Code explanation, Code writing, Debugging, Code + theory\n"
        "Code is OPTIONAL for all other categories.\n"
        "When including code, ALWAYS use:\n"
        "```python\n"
        "x = 10\n"
        "print(x)\n"
        "```\n\n"
        "RULES:\n"
        "1. Use ONLY information from the source text.\n"
        "2. Do NOT invent facts or use outside knowledge.\n"
        "3. Be factually precise.\n"
        "4. Quality and correctness are highest priority.\n\n"
        "Return EXACTLY this structure:\n\n"
        "CATEGORY: CategoryName\n"
        "QUESTION: Your natural question here\n"
        "ANSWER: Your answer here\n\n"
        "CATEGORY: CategoryName\n"
        "QUESTION: Your natural question here\n"
        "ANSWER: Your answer here\n\n"
        "CATEGORY: CategoryName\n"
        "QUESTION: Your natural question here\n"
        "ANSWER: Your answer here\n"
    )

    messages = [
        {
            "role": "system",
            "content": (
                SYSTEM_PROMPT + " "
                "Generate 3 high-quality Q&A pairs with CATEGORY labels. "
                "Use triple-backtick python code blocks when code is needed."
            )
        },
        {
            "role": "user",
            "content": user_content
        }
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )


def generate_batch(prompts):
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4000
    )
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    input_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.eos_token_id
        )

    responses = []
    for output in outputs:
        generated_tokens = output[input_length:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        responses.append(response)
    return responses


def is_valid_python(text):
    code_pattern = r"```(?:python)?\s*\n(.*?)```"
    code_blocks = re.findall(code_pattern, text, re.DOTALL | re.IGNORECASE)
    if not code_blocks:
        return True
    for code in code_blocks:
        try:
            ast.parse(code)
        except SyntaxError:
            return False
    return True


def has_code_blocks(text):
    code_pattern = r"```(?:python)?\s*\n(.*?)```"
    code_blocks = re.findall(code_pattern, text, re.DOTALL | re.IGNORECASE)
    return len(code_blocks) > 0


def is_category_label(question):
    normalized = question.strip().lower()
    return normalized in {cat.lower() for cat in ALL_CATEGORIES}


def normalize_question(question):
    return re.sub(r"\s+", " ", question.strip().lower())


def validate_category_requirements(category, answer):
    if category in CATEGORIES_REQUIRING_CODE:
        if not has_code_blocks(answer):
            return False
    return True


def parse_response(response, target_categories, source_file, chunk_id):
    global SEEN_QUESTIONS
    global REJECTED_MALFORMED, REJECTED_DUPLICATE, REJECTED_NO_CODE, REJECTED_SHORT
    global REJECTED_CATEGORY_LABEL, REJECTED_UNKNOWN_CATEGORY, REJECTED_BAD_CODE_FORMAT
    
    response = response.strip()
    valid = []
    
    blocks = re.split(r'\n(?=CATEGORY:)', response)
    
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        
        category_match = re.search(r'CATEGORY:\s*(.+)', block)
        if not category_match:
            continue
        generated_category = category_match.group(1).strip()
        
        if generated_category not in ALL_CATEGORIES:
            REJECTED_UNKNOWN_CATEGORY += 1
            continue
        
        question_match = re.search(r'QUESTION:\s*(.+?)(?=\nANSWER:|\Z)', block, re.DOTALL)
        if not question_match:
            REJECTED_MALFORMED += 1
            continue
        question = question_match.group(1).strip()
        
        answer_match = re.search(r'ANSWER:\s*(.+?)(?=\nCATEGORY:|\Z)', block, re.DOTALL)
        if not answer_match:
            REJECTED_MALFORMED += 1
            continue
        answer = answer_match.group(1).strip()
        
        if is_category_label(question):
            REJECTED_CATEGORY_LABEL += 1
            continue
        
        if len(question) < 10:
            REJECTED_SHORT += 1
            continue
        
        if len(answer) < 40:
            REJECTED_SHORT += 1
            continue
        
        if "CATEGORY:" in answer or "QUESTION:" in answer:
            REJECTED_MALFORMED += 1
            continue
        
        if not is_valid_python(answer):
            continue
        
        if not validate_category_requirements(generated_category, answer):
            REJECTED_NO_CODE += 1
            continue
        
        question_key = normalize_question(question)
        if question_key in SEEN_QUESTIONS:
            REJECTED_DUPLICATE += 1
            continue
        SEEN_QUESTIONS.add(question_key)
        
        valid.append({
            "question": question,
            "answer": answer,
            "category": generated_category,
            "source_file": source_file,
            "chunk_id": chunk_id
        })
    
    return valid


def save_examples(examples):
    global CATEGORY_COUNTS
    with open(OUTPUT_FILE, "a", encoding="utf-8") as file:
        for item in examples:
            CATEGORY_COUNTS[item["category"]] += 1
            record = {
                "source_file": item["source_file"],
                "chunk_id": item["chunk_id"],
                "category": item["category"],
                "model": MODEL_NAME,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": item["question"]},
                    {"role": "assistant", "content": item["answer"]}
                ]
            }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def print_stats():
    print()
    print("=" * 70)
    print("DATASET GENERATION STATISTICS")
    print("=" * 70)
    print()
    print("CATEGORY DISTRIBUTION:")
    sorted_cats = sorted(CATEGORY_COUNTS.items(), key=lambda x: x[1], reverse=True)
    for cat, count in sorted_cats:
        bar = "█" * (count // 3) if count > 0 else ""
        print(f"  {cat:<25} {count:>5}  {bar}")
    total = sum(CATEGORY_COUNTS.values())
    categories_used = sum(1 for count in CATEGORY_COUNTS.values() if count > 0)
    print(f"\n  {'TOTAL GENERATED':<25} {total:>5}")
    print(f"  {'CATEGORIES USED':<25} {categories_used:>5} / 21")
    print()
    print("QUALITY CONTROL:")
    total_rejected = (REJECTED_DUPLICATE + REJECTED_MALFORMED + REJECTED_NO_CODE + 
                      REJECTED_SHORT + REJECTED_CATEGORY_LABEL + REJECTED_UNKNOWN_CATEGORY + 
                      REJECTED_BAD_CODE_FORMAT)
    print(f"  Accepted examples:          {total}")
    print(f"  Total rejected:             {total_rejected}")
    print(f"  No code rejected:           {REJECTED_NO_CODE}")
    print(f"  Duplicates rejected:        {REJECTED_DUPLICATE}")
    print(f"  Malformed rejected:         {REJECTED_MALFORMED}")
    print(f"  Too short rejected:         {REJECTED_SHORT}")
    print("=" * 70)


def main():
    global CYCLE_INDEX
    
    txt_files = sorted(INPUT_FOLDER.glob("*.txt"))
    if not txt_files:
        print("ERROR: No .txt files found in data/cleaned")
        return

    print()
    print(f"Found {len(txt_files)} cleaned files.")
    print(f"Chunk size: {CHUNK_SIZE} | Max tokens: {MAX_NEW_TOKENS} | Batch: {BATCH_SIZE}")
    print()

    OUTPUT_FILE.write_text("", encoding="utf-8")
    processed_chunks = 0
    total_examples = 0

    for txt_file in txt_files:
        print("=" * 70)
        print("FILE:", txt_file.name)
        text = txt_file.read_text(encoding="utf-8", errors="replace")
        chunks = split_text(text)
        print(f"Characters: {len(text)} | Chunks: {len(chunks)}")

        chunk_batches = [chunks[i:i + BATCH_SIZE] for i in range(0, len(chunks), BATCH_SIZE)]
        
        for batch_num, batch_chunks in enumerate(chunk_batches, 1):
            remaining = MAX_TEST_CHUNKS - processed_chunks
            if remaining <= 0:
                break
            batch_chunks = batch_chunks[:remaining]
            
            batch_prompts = []
            batch_categories = []
            batch_info = []
            
            for chunk in batch_chunks:
                chunk_id = processed_chunks + len(batch_prompts) + 1
                target_cats = get_next_categories(3)
                batch_categories.append(target_cats)
                batch_info.append((txt_file.name, chunk_id, target_cats))
                batch_prompts.append(make_prompt(chunk, target_cats))
            
            print(f"\n  Batch {batch_num}: Chunks {processed_chunks + 1}-{processed_chunks + len(batch_chunks)}")
            
            responses = generate_batch(batch_prompts)
            
            for response, (src_file, chk_id, tgt_cats) in zip(responses, batch_info):
                examples = parse_response(response, tgt_cats, src_file, chk_id)
                if examples:
                    save_examples(examples)
                    total_examples += len(examples)
            
            processed_chunks += len(batch_chunks)
            print(f"  Total examples: {total_examples} | Progress: {processed_chunks} chunks")
            
            if processed_chunks >= MAX_TEST_CHUNKS:
                break
        
        if processed_chunks >= MAX_TEST_CHUNKS:
            break

    print()
    print("=" * 70)
    print("RUN COMPLETE")
    print(f"Processed chunks: {processed_chunks}")
    print(f"Total valid examples: {total_examples}")
    print(f"Output: {OUTPUT_FILE}")
    print("=" * 70)
    print_stats()


if __name__ == "__main__":
    main()