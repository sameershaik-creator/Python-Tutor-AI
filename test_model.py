# save as real_test.py
import torch
# Patch missing attributes in torch for torchao compatibility
for i in range(1, 8):
    if not hasattr(torch, f"int{i}"):
        setattr(torch, f"int{i}", torch.int8)
import torch.utils._pytree
if not hasattr(torch.utils._pytree, "register_constant"):
    torch.utils._pytree.register_constant = lambda cls: cls

from unsloth import FastLanguageModel
import warnings
warnings.filterwarnings("ignore")

print("🔬 TESTING: Did your model learn from YOUR data?")
print("=" * 60)

model, tokenizer = FastLanguageModel.from_pretrained(
    "python-tutor-lora-final",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

# Questions ONLY from your dataset
tests = [
    # From Muppai Rojullo Python (Telugu book)
    ("Muppai rojullo Python ante enti?", "Muppai Rojullo"),
    ("List ante enti mowa?", "mowa"),
    ("Functions gurinchi cheppu bro", "def"),
    
    # From Learning Python book
    ("What is dynamic typing?", "runtime"),
    ("Explain Python garbage collection", "memory"),
    
    # From Python One-Liners
    ("Write a one-liner to reverse a string", "[::-1]"),
]

for question, keyword in tests:
    messages = [
        {"role": "system", "content": "You are a Python tutor. Answer in the style of your training data."},
        {"role": "user", "content": question}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    answer = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    found = keyword.lower() in answer.lower()
    print(f"Q: {question}")
    print(f"A: {answer[:200]}...")
    print(f"Keyword '{keyword}' found: {'✅ YES' if found else '❌ NO'}")
    print("-" * 40)