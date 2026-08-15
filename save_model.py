import torch
# Patch missing attributes in torch for torchao compatibility
for i in range(1, 8):
    if not hasattr(torch, f"int{i}"):
        setattr(torch, f"int{i}", torch.int8)
import torch.utils._pytree
if not hasattr(torch.utils._pytree, "register_constant"):
    torch.utils._pytree.register_constant = lambda cls: cls

from unsloth import FastLanguageModel
import torch

# Load the checkpoint
model, tokenizer = FastLanguageModel.from_pretrained(
    "python-tutor-lora-v2/checkpoint-400",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

# Save without the pickle error
model.save_pretrained("python-tutor-lora-final")
tokenizer.save_pretrained("python-tutor-lora-final")
print("Model saved to python-tutor-lora-final!")