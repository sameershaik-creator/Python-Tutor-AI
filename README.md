# 🐍 Python Tutor AI

An interactive, bilingual AI programming tutor designed to explain Python concepts, assist with code debugging, and run code with natural explanations in **Telglish** (Telugu + English) and **English**.

Powered by a fine-tuned LLM using **Unsloth** and integrated with an interactive **Gradio** web interface and custom execution environment.

---

## ✨ Features

- **🎓 Interactive AI Tutor**: Patient, beginner-friendly programming assistant tailored for learning Python.
- **🗣️ Telglish & English Support**: Naturally responds in conversational Telglish (Telugu script in Latin letters) or English based on user input.
- **▶️ Code Execution & Step-by-Step Explanation**: Write code in the editor, run it, and receive an instant breakdown of standard outputs or error tracebacks.
- **⚡ Efficient Inference**: Fine-tuned using LoRA 4-bit quantization with Unsloth for fast execution and low memory consumption.

---

## 🛠️ Project Architecture

```
Python-Tutor-AI/
├── app.py                     # Main Gradio Web Application UI & Chat logic
├── python_executor.py         # Subprocess-isolated Python code execution runner
├── python-tutor-lora-final/   # Fine-tuned LoRA adapter weights & tokenizer
├── fine_tuning.py             # Unsloth fine-tuning pipeline script
├── generate_dataset.py        # Dataset generation & processing script
├── factual_review.py          # Quality and factual review filtering
├── quality_filter.py           # Dataset curation script
├── requirements.txt           # Environment dependencies
└── README.md                  # Project documentation
```

> **Note on Code Execution**: Python code execution is designed for local educational testing and prototyping. For public production deployment, sandboxed container isolation (e.g., Docker/gVisor) is recommended.

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.10+
- NVIDIA GPU with CUDA support (recommended)

### 2. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/YOUR_USERNAME/Python-Tutor-AI.git
cd Python-Tutor-AI
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Running the Application

Launch the Gradio app locally:

```bash
python app.py
```

Open the local URL displayed in the terminal (default: `http://localhost:7860`).

---

## 🔬 Model & Fine-tuning

The model was fine-tuned using [Unsloth](https://github.com/unslothai/unsloth) on custom curated Telglish and English Python programming QA pairs.

Scripts included:
- `generate_dataset.py`: Parses and prepares training text & dataset pairs.
- `fine_tuning.py`: Trains the LoRA adapters using QLoRA.
- `save_model.py`: Merges or saves the LoRA adapter checkpoints.

---

## 📜 License

This project is released under the MIT License.
