# src/annotator.py
import os
import json
import re
import time
import urllib.request
import urllib.error
from tqdm import tqdm
from src.data import load_math500_subset, format_prompt


def load_env_vars(env_path: str = ".env") -> dict[str, str]:
    """Loads environment variables manually from a .env file to avoid external dependencies."""
    env_vars = {}
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    parts = line.split("=", 1)
                    if len(parts) == 2:
                        env_vars[parts[0].strip()] = parts[1].strip()
    # Also merge with actual os.environ
    for k, v in os.environ.items():
        env_vars[k] = v
    return env_vars


def extract_boxed(s: str) -> str:
    """Extracts the content of the last \\boxed{...} in a string, handling balanced braces."""
    idx = s.rfind("\\boxed{")
    if idx == -1:
        return s.strip()
    i = idx + len("\\boxed{")
    depth = 1
    start = i
    while i < len(s) and depth > 0:
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return s[start:i].strip()
        i += 1
    return s[start:].strip()


def is_correct(trajectory: str, answer: str) -> bool:
    """Checks if the trajectory's final boxed answer matches the expected answer."""
    # Attempt using math-verify if installed, else fallback to manual extraction
    try:
        from math_verify import parse, verify

        return bool(verify(parse(answer), parse(trajectory)))
    except Exception:
        p = extract_boxed(trajectory)
        g = extract_boxed(answer)
        return p == g and p != ""


def call_gemini_api(prompt: str, api_key: str, model: str = "gemini-2.5-flash") -> str:
    """Calls Google's Gemini API via direct REST endpoint (no heavy SDK required)."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 2048},
    }

    req = urllib.request.Request(
        url, data=json.dumps(data).encode("utf-8"), headers=headers, method="POST"
    )

    # Try with exponential backoff on HTTP 429/503 errors
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                return res_data["candidates"][0]["content"]["parts"][0]["text"]
        except urllib.error.HTTPError as e:
            if e.code in (429, 503):
                sleep_time = (2**attempt) * 4
                print(
                    f"Gemini API error ({e.code}). Retrying in {sleep_time}s "
                    f"(attempt {attempt + 1}/5)..."
                )
                time.sleep(sleep_time)
                # Rebuild request object for retry (urllib consumes it)
                req = urllib.request.Request(
                    url,
                    data=json.dumps(data).encode("utf-8"),
                    headers=headers,
                    method="POST",
                )
                continue
            raise RuntimeError(f"Gemini API error: {e.code} - {e.reason}")
    raise RuntimeError("Failed to call Gemini API after 5 retries due to rate limits.")


def call_ollama_local(prompt: str, model: str = "qwen2.5-coder:7b") -> str:
    """Attempts to call a local Ollama server if running."""
    url = "http://localhost:11434/api/generate"
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0},
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            return res_data["response"]
    except Exception as e:
        raise RuntimeError(f"Ollama local call failed: {e}")


# Lazy loaded transformers components
_local_model = None
_local_tokenizer = None


def call_transformers_local(
    prompt: str, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
) -> str:
    """Fallback local model execution via Hugging Face transformers on CPU/GPU."""
    global _local_model, _local_tokenizer
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if _local_model is None or _local_tokenizer is None:
        print(
            f"Loading local fallback model '{model_name}' (this may take a few minutes on CPU)..."
        )
        _local_tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _local_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=torch.float32 if device == "cpu" else torch.float16,
            trust_remote_code=True,
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = _local_tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = _local_model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
            pad_token_id=_local_tokenizer.eos_token_id,
        )
    generated_ids = outputs[0][inputs.input_ids.shape[-1] :]
    return _local_tokenizer.decode(generated_ids, skip_special_tokens=True)


def generate_local_student_trajectories(num_samples: int = 5) -> list[dict]:
    """Generates initial student trajectories locally on CPU if none exist."""
    print("Generating student trajectories locally on CPU...")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    student_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(student_model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        student_model_name,
        device_map=device,
        torch_dtype=torch.float32 if device == "cpu" else torch.float16,
    )

    dataset = load_math500_subset(num_samples=num_samples)
    records = []

    for item in tqdm(dataset, desc="Running Student Model"):
        prompt = format_prompt(tokenizer, item["question"])
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )

        generated_ids = outputs[0][inputs.input_ids.shape[-1] :]
        trajectory = tokenizer.decode(generated_ids, skip_special_tokens=True)

        records.append(
            {
                "question": item["question"],
                "answer": item["answer"],
                "trajectory": trajectory,
            }
        )
    return records


def annotate_dataset(input_file: str, output_file: str, max_samples: int = 5):
    """Loads student trajectories, annotates incorrect ones with the teacher model (Gemini -> Fallback)."""
    env_vars = load_env_vars()
    gemini_key = env_vars.get("GEMINI_API_KEY", "")

    # 1. Load or generate input trajectories
    if os.path.exists(input_file):
        print(f"Loading trajectories from {input_file}...")
        with open(input_file, "r") as f:
            trajectories = json.load(f)
    else:
        print(f"Input file {input_file} not found.")
        trajectories = generate_local_student_trajectories(num_samples=max_samples)
        os.makedirs(os.path.dirname(input_file) or ".", exist_ok=True)
        with open(input_file, "w") as f:
            json.dump(trajectories, f, indent=2)
            print(f"Saved initial student trajectories to {input_file}")

    annotated_records = []

    # 2. Setup prompt and perform annotations
    for item in tqdm(trajectories[:max_samples], desc="Annotating Trajectories"):
        correct = is_correct(item["trajectory"], item["answer"])
        item["is_correct"] = correct

        if correct:
            item["teacher_analysis"] = "Correct trajectory, no annotation needed."
            item["pof_quote"] = ""
            item["annotation_valid"] = True
            annotated_records.append(item)
            continue

        # Build prompt for step-by-step annotation
        prompt = (
            "You are a meticulous math tutor. You analyze a student's incorrect reasoning "
            "and identify the first step that contains a logical or computational error.\n\n"
            f"Question: {item['question']}\n\n"
            f"Ground Truth Solution: {item['answer']}\n\n"
            f"Student's reasoning trace:\n\"\"\"\n{item['trajectory']}\n\"\"\"\n\n"
            "The student's final answer is wrong. Find the FIRST step where the student's "
            "reasoning breaks (logical error, wrong formula, arithmetic mistake, misread problem).\n\n"
            "Output STRICTLY in this XML format and nothing else after the closing tag:\n"
            "<analysis>Brief explanation of the first error and why it is wrong.</analysis>\n"
            "<pof_quote>Copy the exact verbatim substring from the student's reasoning where "
            "the error first appears. Do not paraphrase. Do not summarize. Copy characters exactly.</pof_quote>"
        )

        response_text = ""
        used_fallback = False

        # A. Try Gemini API
        if gemini_key:
            try:
                response_text = call_gemini_api(prompt, gemini_key)
                # Respect free-tier rate limits (~15 RPM = 4s between requests)
                time.sleep(5)
            except Exception as e:
                print(f"\nGemini API call failed: {e}. Trying local fallback...")
                used_fallback = True
        else:
            print("\nGEMINI_API_KEY not configured. Using local fallback...")
            used_fallback = True

        # B. Fallback to Local Setup (Ollama -> Transformers)
        if used_fallback:
            try:
                # Try local Ollama first (faster)
                response_text = call_ollama_local(prompt)
            except Exception as e:
                # Try local Transformers on CPU
                print(
                    f"Ollama fallback unavailable ({e}). Using local transformers model..."
                )
                try:
                    response_text = call_transformers_local(prompt)
                except Exception as ex:
                    print(f"Transformers fallback failed: {ex}")
                    response_text = ""

        # Parse XML tags
        analysis_match = re.search(
            r"<analysis>(.*?)</analysis>", response_text, re.DOTALL
        )
        pof_match = re.search(r"<pof_quote>(.*?)</pof_quote>", response_text, re.DOTALL)

        analysis = (
            analysis_match.group(1).strip() if analysis_match else response_text.strip()
        )
        pof = pof_match.group(1).strip() if pof_match else ""

        # Clean up tags in case of truncation or failure to close tags
        if not analysis_match:
            analysis = re.sub(r"^<analysis>", "", analysis).strip()
            analysis = re.sub(r"</analysis>$", "", analysis).strip()
        if not pof_match:
            # Check for unclosed <pof_quote> tags
            pof_fallback_match = re.search(r"<pof_quote>(.*)", response_text, re.DOTALL)
            if pof_fallback_match:
                pof = pof_fallback_match.group(1).strip()
            pof = re.sub(r"^<pof_quote>", "", pof).strip()
            pof = re.sub(r"</pof_quote>$", "", pof).strip()

        # Ground pof_quote: must be substring of trajectory. If not, mark ungrounded.
        pof_grounded = bool(pof) and pof in item["trajectory"]

        item["teacher_analysis"] = analysis
        item["pof_quote"] = pof if pof_grounded else ""
        item["annotation_valid"] = bool(analysis_match) and pof_grounded

        # Add metadata on backend source
        item["backend"] = "local_fallback" if used_fallback else "gemini"

        annotated_records.append(item)

    # Save the output
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(annotated_records, f, indent=2)

    print(f"\nSuccessfully annotated {len(annotated_records)} items.")
    print(f"Saved annotated dataset to: {output_file}")


if __name__ == "__main__":
    annotate_dataset(
        input_file="data/student_trajectories.json",
        output_file="data/annotated_trajectories.json",
        max_samples=3,
    )
