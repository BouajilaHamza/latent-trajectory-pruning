# HF Integration Guide: InternLM Models & ML-Intern Agent

This guide outlines how to leverage the Hugging Face ecosystem—specifically **InternLM/Intern-S1 reasoning models** and the **`ml-intern` autonomous ML engineering agent**—on both **Modal** (for scalable GPU inference) and **local CPU** environments to make your Latent Trajectory Pruning (LTP) research more grounded and professional.

---

## 1. Using InternLM / Intern-S1 Models on Hugging Face

The `internlm` organization hosts state-of-the-art open-source reasoning models (such as `internlm2_5-7b-chat`, `internlm2_5-1_8b-chat`, and the scientific reasoning model `intern-s1-mini`). These models are highly suited as student or teacher models in trajectory pruning because of their advanced step-by-step reasoning capabilities.

### Option A: Running on Modal (Recommended for Speed & Scale)
Modal allows you to host InternLM models using `vLLM` on cost-effective GPUs (like NVIDIA L4 or A10G).

#### 1. Configure the Modal App (`src/modal_annotator.py` or `src/ltp_pipeline.py`)
InternLM models require passing `trust_remote_code=True` to the vLLM engine. Update your class setup as follows:

```python
# Change the MODEL_NAME to the desired InternLM variant
MODEL_NAME = "internlm/internlm2_5-7b-chat"  # Or "internlm/intern-s1-mini"

@app.cls(
    image=vllm_image, 
    gpu="L4", 
    secrets=[modal.Secret.from_name("huggingface-secret")],
    container_idle_timeout=60,
)
class TeacherAnnotator:
    @modal.enter()
    def setup(self):
        from vllm import LLM
        # InternLM models require trust_remote_code=True
        self.llm = LLM(
            model=MODEL_NAME, 
            tensor_parallel_size=1, 
            gpu_memory_utilization=0.9,
            max_model_len=8192,
            trust_remote_code=True  
        )
```

#### 2. Chat Template and Sampling
InternLM2.5 uses standard chat formatting. Under `vLLM`, you can apply the tokenizer's chat template directly:

```python
tokenizer = self.llm.get_tokenizer()
prompts = [
    tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "You are a math tutor..."},
            {"role": "user", "content": f"Question: {item['question']}\nStudent trajectory: {item['trajectory']}"}
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    for item in items
]
```

---

### Option B: Running on CPU (Recommended for Light Debugging & Local Tests)
Running models on CPU is slow, so you should use lightweight variants (like the 1.8B parameter model).

#### 1. Using Hugging Face `transformers`
You can load the model on CPU using the standard `transformers` library with `torch.float32`.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_internlm_cpu(model_name: str = "internlm/internlm2_5-1_8b-chat"):
    print(f"Loading {model_name} on CPU...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cpu",
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    return model, tokenizer

# Usage:
# model, tokenizer = load_internlm_cpu()
```

#### 2. Using `llama.cpp` / Ollama (Highly Optimized for CPU)
For faster local CPU inference, running GGUF-quantized versions is recommended.
1. Download a quantized InternLM2.5 model (e.g. `internlm2_5-7b-chat-gguf`).
2. Run it via Ollama:
   ```bash
   ollama run internlm2.5
   ```
3. Query it using the local API endpoint (compatible with OpenAI client):
   ```python
   from openai import OpenAI
   client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
   ```

---

## 2. Using HF `ml-intern` (Autonomous ML Agent)

`huggingface/ml-intern` is an AI agent designed to operate as an autonomous machine learning engineer. You can use it to design, execute, and debug your trajectory pruning experiments.

### How to Install and Run Locally (CPU)
Since `ml-intern` is an orchestrator that calls LLM APIs (like Anthropic Claude or OpenAI GPT-4) to plan actions and write python scripts, it runs natively on CPU.

#### 1. Installation
Clone the repository and install it in your environment using `uv`:

```bash
# Clone the ml-intern repository
git clone https://github.com/huggingface/ml-intern.git
cd ml-intern

# Sync dependencies using uv
uv sync
uv tool install -e .
```

#### 2. Configure API Keys (Including Free Options)
The agent uses `smolagents` and LiteLLM under the hood. You do **not** need paid OpenAI or Anthropic keys. You can use free backends:

##### Option A: Hugging Face Serverless Inference API (Free & Cloud-Based)
You can use a free Hugging Face account and user access token to call models hosted by HF:
1. Create a free account on Hugging Face and generate a User Access Token in your settings.
2. Export the token and run the agent using free serverless endpoints (such as `Qwen/Qwen2.5-Coder-32B-Instruct`):
```bash
export HF_TOKEN="your-free-hf-token-here"
```

##### Option B: Local Model via Ollama (100% Free & Local CPU/GPU)
You can run a model entirely locally on your machine for free:
1. Install Ollama and run a coding/reasoning model:
   ```bash
   ollama run qwen2.5-coder:7b
   ```
2. Configure LiteLLM to route calls to your local server:
   ```bash
   export OLLAMA_API_BASE="http://localhost:11434/v1"
   ```

##### Option C: Google Gemini API Free Tier (Free Cloud API)
Google AI Studio offers a generous free tier (e.g. 15 requests per minute) for models like `gemini-1.5-flash` and `gemini-2.0-flash`:
1. Get a free API key from Google AI Studio.
2. Export the key to your environment:
   ```bash
   export GEMINI_API_KEY="your-free-gemini-key"
   ```

#### 3. Run a Research Task
You can direct `ml-intern` to run experiments in your `latent-trajectory-pruning` directory. For example, you can prompt it to optimize your linear probe:

```bash
ml-intern run "Optimize the logistic regression probe in src/probe.py by testing different regularization strengths (L1 vs L2) and evaluate on traces.pt. Write the results to docs/research/experiment_results.md"
```

The agent will:
1. Inspect `src/probe.py` and `data/traces.pt`.
2. Write a scratch script to run the sweep.
3. Execute the script locally on your CPU.
4. Analyze the output and write the markdown report.

---

### Running `ml-intern` Workloads on Modal
If `ml-intern` needs to run heavy CPU/GPU workloads (e.g. extracting hidden states from a larger model), you can configure the agent to submit Modal jobs:
1. Wrap the heavy computation task (e.g. `extractor.py`) inside a Modal function.
2. Instruct the agent: *"Trigger the trajectory extraction on Modal using a GPU, then download the traces and train the probe locally."*
3. The agent will execute `modal run src/ltp_pipeline.py` or similar CLI commands to orchestrate the remote execution.
