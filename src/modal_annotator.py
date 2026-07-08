# src/modal_annotator.py
import modal
from modal import Image, App

# Robust image with compilers for vLLM JIT kernels
vllm_image = (
    Image.debian_slim(python_version="3.10")
    .apt_install("git", "g++", "nvidia-cuda-toolkit")
    .pip_install("vllm>=0.6.0", "datasets", "huggingface_hub")
)

app = App("ltp-teacher-annotator")

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"


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

        self.llm = LLM(
            model=MODEL_NAME,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            max_model_len=16384,
        )

    @modal.method()
    def annotate_item(self, item: dict) -> dict:
        from vllm import SamplingParams
        import re

        if item.get("is_correct", False):
            item["teacher_analysis"] = "Correct trajectory, no annotation needed."
            item["pof_quote"] = ""
            return item

        prompt = (
            f"Question: {item['question']}\n"
            f"Ground Truth: {item['ground_truth']}\n"
            f"Student's reasoning:\n{item['trajectory']}\n\n"
            "The student's reasoning is incorrect. Analyze the student's reasoning step-by-step to identify the first logical or mathematical error.\n"
            "You must first reason internally about the error, and then provide your final analysis and the point of failure quote in the following XML format:\n"
            "<analysis>Your final step-by-step analysis here</analysis>\n"
            "<pof_quote>The exact sentence from the student's reasoning where they first failed</pof_quote>"
        )

        outputs = self.llm.generate(
            [prompt], SamplingParams(temperature=0.0, max_tokens=2048)
        )
        teacher_response = outputs[0].outputs[0].text

        # Extract using XML tags, stripping internal thought process
        clean_response = re.sub(
            r"<thought>.*?</thought>", "", teacher_response, flags=re.DOTALL
        )
        analysis_match = re.search(
            r"<analysis>(.*?)</analysis>", clean_response, re.DOTALL
        )
        pof_match = re.search(
            r"<pof_quote>(.*?)</pof_quote>", clean_response, re.DOTALL
        )

        analysis = (
            analysis_match.group(1).strip()
            if analysis_match
            else clean_response.strip()
        )
        pof_quote = pof_match.group(1).strip() if pof_match else ""

        # Cleanup common tokenizer artifacts
        item["teacher_analysis"] = analysis.replace("Ġ", " ").replace("Ċ", "\n")
        item["pof_quote"] = pof_quote.replace("Ġ", " ").replace("Ċ", "\n")
        return item


@app.local_entrypoint()
def main(input_repo: str, output_repo: str):
    from datasets import load_dataset, Dataset

    print(f"Loading dataset from {input_repo}...")
    ds = load_dataset(input_repo, split="train")
    items = [row for row in ds]

    annotator = TeacherAnnotator()

    print(f"Starting parallel annotation of {len(items)} items via Modal L4 GPUs...")
    # This is the key for speed: Modal will auto-scale to multiple GPUs
    annotated_items = list(annotator.annotate_item.map(items, order_preserved=True))

    print(f"Pushing annotated dataset to {output_repo}...")
    Dataset.from_list(annotated_items).push_to_hub(output_repo)
    print("Done!")
