import json
import re
import ast
from pathlib import Path
from collections import defaultdict

INPUT_FILE = Path("data/dataset/python_tutor_dataset.jsonl")
OUTPUT_FILE = Path("data/dataset/python_tutor_dataset_final.jsonl")

ALL_CATEGORIES = [
    "Theory", "Deep theory", "Conceptual", "Code explanation",
    "Output prediction", "Code writing", "Debugging", "Why",
    "What", "How", "When", "Comparison", "Trade-offs",
    "Scenario-based", "Research-level", "Edge cases",
    "Output + explanation", "Code + theory", "Performance",
    "Design", "Interview-style"
]

CATEGORIES_REQUIRING_CODE = [
    "Output prediction", "Output + explanation", "Code explanation",
    "Code writing", "Debugging", "Code + theory"
]

# FIXED: Triple backtick regex
CODE_BLOCK_PATTERN = r"```(?:python)?\s*\n(.*?)```"

QUESTION_STARTERS = (
    "what ", "why ", "how ", "when ", "where ",
    "which ", "can ", "write ", "explain ",
    "compare ", "find ", "debug ", "implement ",
    "create ", "define ", "describe ", "list ",
    "is ", "are ", "does ", "do ", "should "
)

# Stats counters
removed_invalid_json = 0
removed_bad_structure = 0
removed_invalid_category = 0
removed_duplicate = 0
removed_near_duplicate = 0
removed_no_code = 0
removed_invalid_python = 0
removed_short_answer = 0
removed_invalid_question = 0
removed_category_label = 0
total_input = 0
total_output = 0

seen_questions = set()
question_word_sets = defaultdict(list)


def normalize(text):
    return re.sub(r'\s+', ' ', text.strip().lower())


def has_code_blocks(text):
    return bool(re.findall(CODE_BLOCK_PATTERN, text, re.DOTALL | re.IGNORECASE))


def is_valid_python(text):
    code_blocks = re.findall(CODE_BLOCK_PATTERN, text, re.DOTALL | re.IGNORECASE)
    if not code_blocks:
        return True
    for code in code_blocks:
        try:
            ast.parse(code.strip())
        except SyntaxError:
            return False
    return True


def is_category_label(text):
    return normalize(text) in {cat.lower() for cat in ALL_CATEGORIES}


def is_valid_question(question):
    q = normalize(question)
    
    if len(q) < 15:
        return False
    
    if q.endswith("?"):
        return True
    
    if q.startswith(QUESTION_STARTERS):
        return True
    
    return False


def get_question_words(question):
    words = set(normalize(question).split())
    stopwords = {'what', 'is', 'the', 'a', 'an', 'in', 'of', 'to', 'how', 
                 'why', 'when', 'does', 'do', 'can', 'you', 'are', 'it', 
                 'and', 'or', 'for', 'with', 'this', 'that', 'be', 'on', 'at',
                 'its', 'by', 'from', 'as', 'if', 'not', 'but', 'has', 'have'}
    return words - stopwords


print(f"Reading: {INPUT_FILE}")
print("Filtering...\n")

with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    lines = f.readlines()

total_input = len(lines)
filtered = []

for line_num, line in enumerate(lines, 1):
    # FILTER 1: Valid JSON
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        removed_invalid_json += 1
        continue
    
    category = record.get('category', '')
    messages = record.get('messages', [])
    
    # FILTER 2: Validate message structure
    if len(messages) != 3:
        removed_bad_structure += 1
        continue
    
    if (messages[0].get('role') != 'system' or 
        messages[1].get('role') != 'user' or 
        messages[2].get('role') != 'assistant'):
        removed_bad_structure += 1
        continue
    
    # FILTER 3: Valid category
    if category not in ALL_CATEGORIES:
        removed_invalid_category += 1
        continue
    
    question = messages[1].get('content', '')
    answer = messages[2].get('content', '')
    
    # FILTER 4: Question not empty
    if not question.strip():
        removed_invalid_question += 1
        continue
    
    # FILTER 5: Answer not empty
    if not answer.strip():
        removed_short_answer += 1
        continue
    
    # FILTER 6: Question not a category label
    if is_category_label(question):
        removed_category_label += 1
        continue
    
    # FILTER 7: Question validation
    if not is_valid_question(question):
        removed_invalid_question += 1
        continue
    
    # FILTER 8: Answer minimum length
    if len(answer) < 50:
        removed_short_answer += 1
        continue
    
    # FILTER 9: Code-required categories must have code
    if category in CATEGORIES_REQUIRING_CODE:
        if not has_code_blocks(answer):
            removed_no_code += 1
            continue
    
    # FILTER 10: Validate Python syntax
    if has_code_blocks(answer):
        if not is_valid_python(answer):
            removed_invalid_python += 1
            continue
    
    # FILTER 11: Exact duplicates
    q_normalized = normalize(question)
    if q_normalized in seen_questions:
        removed_duplicate += 1
        continue
    seen_questions.add(q_normalized)
    
    # FILTER 12: Near-duplicate within same category (>80% word overlap)
    current_words = get_question_words(question)
    is_near_duplicate = False
    
    existing_sets = question_word_sets[category]
    for existing_words in existing_sets:
        if current_words and existing_words:
            overlap = len(current_words & existing_words)
            max_len = max(len(current_words), len(existing_words))
            if max_len > 0 and overlap / max_len > 0.8:
                is_near_duplicate = True
                break
    
    if is_near_duplicate:
        removed_near_duplicate += 1
        continue
    
    question_word_sets[category].append(current_words)
    
    # KEEP
    filtered.append(record)

total_output = len(filtered)

# Save
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    for record in filtered:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')

# Stats
print("=" * 60)
print("QUALITY FILTER RESULTS")
print("=" * 60)
print(f"  Input examples:              {total_input}")
print(f"  Output examples:             {total_output}")
print(f"  Removed:                     {total_input - total_output}")
print()
print("REMOVAL BREAKDOWN:")
print(f"  Invalid JSON:                {removed_invalid_json}")
print(f"  Bad message structure:       {removed_bad_structure}")
print(f"  Invalid category:            {removed_invalid_category}")
print(f"  Category label as question:  {removed_category_label}")
print(f"  Invalid question format:     {removed_invalid_question}")
print(f"  Answer too short:            {removed_short_answer}")
print(f"  No code (code-required):     {removed_no_code}")
print(f"  Invalid Python in code:      {removed_invalid_python}")
print(f"  Exact duplicates:            {removed_duplicate}")
print(f"  Near duplicates (>80%):      {removed_near_duplicate}")
print()

# Final distribution
cat_counts = defaultdict(int)
for record in filtered:
    cat_counts[record['category']] += 1

print("FINAL CATEGORY DISTRIBUTION:")
for cat in sorted(cat_counts.keys(), key=lambda x: cat_counts[x], reverse=True):
    count = cat_counts[cat]
    bar = '█' * (count // 5)
    print(f"  {cat:<25} {count:>5}  {bar}")

print(f"\n  {'TOTAL':<25} {total_output:>5}")
print(f"  {'CATEGORIES':<25} {len(cat_counts):>5} / 21")
print("=" * 60)
print(f"\nSaved to: {OUTPUT_FILE}")