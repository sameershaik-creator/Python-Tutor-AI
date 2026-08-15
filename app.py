import time
import warnings
from pathlib import Path

import torch
import gradio as gr

for i in range(1, 8):
    if not hasattr(torch, f"int{i}"):
        setattr(torch, f"int{i}", torch.int8)

if not hasattr(torch, "float4_e2m1fn_x2"):
    setattr(torch, "float4_e2m1fn_x2", getattr(torch, "float8_e4m3fn", torch.float16))

import torch.utils._pytree

if not hasattr(torch.utils._pytree, "register_constant"):
    torch.utils._pytree.register_constant = lambda cls: cls

warnings.filterwarnings("ignore")

from unsloth import FastLanguageModel
from python_executor import execute_python_code

MODEL_PATH = "python-tutor-lora-final"
MAX_INPUT_TOKENS = 2500
NORMAL_MAX_NEW_TOKENS = 160
CODE_MAX_NEW_TOKENS = 256

SYSTEM_PROMPT = r"""
You are Python Tutor AI, a friendly and patient Python programming tutor.

Your personality:
- Talk like a supportive senior friend helping a beginner.
- Be friendly, natural and encouraging.
- You may use "bro", "rey", etc. when the user uses them.
- Do not overuse slang.
- Never sound robotic or formal unless the user asks for formal language.

LANGUAGE:
- If the user uses Telugu + English, respond naturally in Telugu + English (Telglish).
- Keep Python keywords and programming terminology in English.
- Do NOT translate Python syntax.
- Example:
  "Ikkada `if` condition check chestundi. Condition `True` ayithe..."
- If the user writes only English, normally answer in English.
- If the user explicitly asks for Telugu/Telglish, follow that request.
Explain the execution result in natural Telglish.

Telglish means:
- Use Telugu conversational grammar written mainly in English/Latin script.
- Keep Python and programming terminology in English.
- Example: "Ee code lo list create chestunnam."
- Example: "Index 3 ki value ledu kabatti IndexError vastundi."
- Example: "for loop prati element ni one by one process chestundi."
- Do NOT generate formal Telugu script.
- Do NOT translate Python keywords.
- Do NOT generate unrelated Telugu sentences.
- Be friendly and beginner-oriented.
- Explain WHY the output/error happened.

TEACHING STYLE:
- Explain concepts simply for beginners.
- Explain WHY something works, not just WHAT it does.
- Give a small example when useful.
- If the user asks for code, provide working code.
- If the user asks to explain code, explain it clearly step-by-step.
- If the user asks to fix code, identify the error, explain why it happens, and provide corrected code.
- If the user asks a follow-up such as "example ivvu", use the previous conversation context.
- Do not ask the user to repeat information that is already available in the conversation.

ANTI-REPETITION RULES:
- NEVER repeat the user's question.
- NEVER copy the user's message into your answer.
- NEVER repeat the same sentence multiple times.
- Start directly with the answer.
- Never output <think> reasoning.

CODE RULES:
- Python code MUST use fenced Markdown code blocks.
- Use exactly 4 spaces for indentation.
- Generated Python must be syntactically valid.
- When correcting code, show the corrected version.

ANSWER QUALITY:
- Do not give generic filler.
- Answer the actual question.
- For a beginner question, prefer a short explanation + example.
"""

print("🐍 Loading Python Tutor AI...")

model, tokenizer = FastLanguageModel.from_pretrained(
    MODEL_PATH,
    max_seq_length=4096,
    dtype=None,
    load_in_4bit=True,
)

FastLanguageModel.for_inference(model)

print("🐍 Model loaded successfully.")

def build_prompt_messages(history, current_message):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    if history:
        for item in history:
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            content = item.get("content", "")
            if role not in ("user", "assistant"):
                continue
            if content is None:
                continue
            content = str(content).strip()
            if not content:
                continue
            messages.append({"role": role, "content": content})
    
    messages.append({"role": "user", "content": str(current_message).strip()})
    return messages

def count_tokens(messages):
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    return len(tokenizer.encode(text, add_special_tokens=False))

def trim_history(history, current_message):
    history = list(history or [])
    
    while True:
        messages = build_prompt_messages(history, current_message)
        token_count = count_tokens(messages)
        
        if token_count <= MAX_INPUT_TOKENS:
            return history, token_count
        
        if len(history) < 2:
            return [], token_count
        
        history.pop(0)
        if history and history[0].get("role") == "assistant":
            history.pop(0)

def clean_response(response, user_message):
    if response is None:
        return "Bro, response generate avvaledu. Please try again."
    
    response = str(response).strip()
    
    if not response:
        return "Bro, response empty ga vachindi. Please try again."
    
    if "<think>" in response:
        if "</think>" in response:
            response = response.split("</think>", 1)[1].strip()
        else:
            return "Bro, response generation incomplete ga ayyindi. Please ask the question again."
    
    response = response.replace("<|im_end|>", "").strip()
    
    normalized_user = " ".join(str(user_message).strip().lower().split())
    normalized_response = " ".join(response.lower().split())
    
    if (normalized_user and len(normalized_user) > 10 and normalized_response.count(normalized_user) >= 2):
        return "Bro, response lo repetition vachindi. Please ask the question once again."
    
    return response

def chat(message, history, is_code_mode=False):
    start_time = time.time()
    
    if isinstance(message, dict):
        message_text = message.get("text") or message.get("content") or ""
    else:
        message_text = str(message)
    
    message_text = str(message_text).strip()
    
    if not message_text:
        return "Bro, question empty ga undi. Ask me something about Python."
    
    clean_history = []
    
    for item in history or []:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content", "")
        if role not in ("user", "assistant"):
            continue
        if content is None:
            continue
        content = str(content).strip()
        if not content:
            continue
        clean_history.append({"role": role, "content": content})
    
    if clean_history:
        last = clean_history[-1]
        if last.get("role") == "user" and last.get("content", "").strip() == message_text:
            clean_history.pop()
    
    clean_history, token_count = trim_history(clean_history, message_text)
    
    final_messages = build_prompt_messages(clean_history, message_text)
    
    text = tokenizer.apply_chat_template(
        final_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_INPUT_TOKENS).to(model.device)
    
    max_new_tokens = CODE_MAX_NEW_TOKENS if is_code_mode else NORMAL_MAX_NEW_TOKENS
    
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.10,
            no_repeat_ngram_size=4,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    generated_tokens = outputs[0, inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    response = clean_response(response, message_text)
    
    total_time = time.time() - start_time
    print(f"[Profiling] Total response time: {total_time:.2f}s")
    
    return response

def run_and_explain(code_str, stdin_str, history):
    history = list(history or [])
    
    if not code_str or not code_str.strip():
        return history, "❌ No Python code provided."
    
    code_str = code_str.strip()
    
    try:
        result = execute_python_code(code_str, stdin_str or "")
    except Exception as exc:
        return history, f"❌ Executor error:\n{type(exc).__name__}: {exc}"
    
    if result.get("success"):
        output = result.get("output", "")
        if not output:
            output = "(Program completed without output.)"
        output = str(output)
        if len(output) > 5000:
            output = output[:5000] + "\n...[output truncated]"
        
        code_output = "✅ SUCCESS\n\n" + output
        
        explanation_prompt = f"User ran this code:\n```python\n{code_str}\n```\n\nOutput:\n{code_output}\n\nExplain in Telglish."
    else:
        error_msg = result.get("error_message", "Unknown error")
        error_type = result.get("error_type", "Error")
        code_output = f"❌ {error_type}\n\n{error_msg}"
        
        explanation_prompt = f"User ran this code:\n```python\n{code_str}\n```\n\nError:\n{code_output}\n\nExplain the error in Telglish and give fixed code."
    
    history.append({"role": "user", "content": explanation_prompt})
    response = chat(explanation_prompt, history[:-1], is_code_mode=True)
    history.append({"role": "assistant", "content": response})
    
    return history, code_output

with gr.Blocks(title="Python Tutor AI") as demo:
    gr.Markdown("# 🐍 Python Tutor AI")
    
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(height=500)
            msg = gr.Textbox(show_label=False, placeholder="Ask me a Python question...")
            
        with gr.Column(scale=1):
            code_input = gr.Code(language="python", label="Code Editor")
            stdin_input = gr.Textbox(label="Stdin Input (optional)")
            run_btn = gr.Button("▶️ Run & Explain", variant="primary")
            output_box = gr.Textbox(label="Program Output", lines=5, interactive=False)
    
    def chat_wrapper(user_msg, chat_hist):
        chat_hist = list(chat_hist or [])
        
        # Add user message immediately
        chat_hist.append({"role": "user", "content": user_msg})
        
        # Get bot response
        response = chat(user_msg, chat_hist[:-1])
        
        # Add bot response
        chat_hist.append({"role": "assistant", "content": response})
        
        return "", chat_hist
    
    def run_wrapper(code, stdin, hist):
        new_hist, output = run_and_explain(code, stdin, hist)
        return new_hist, output
    
    msg.submit(chat_wrapper, [msg, chatbot], [msg, chatbot])
    run_btn.click(run_wrapper, [code_input, stdin_input, chatbot], [chatbot, output_box])

if __name__ == "__main__":
    try:
        demo.launch(share=True, server_name="0.0.0.0", server_port=7860)
    except OSError:
        demo.launch(share=True, server_name="0.0.0.0")