# src/ltp_pipeline.py
import modal
from modal import Image, App, Secret

# Use the exact stable image from your working run
vllm_image = Image.debian_slim(python_version="3.10").pip_install(
    "vllm>=0.7.0",  # Newer version fixes the Qwen RoPE bug
    "datasets",
    "huggingface_hub",
    "math-verify",  # robust MATH-500 answer equivalence
)

app = App("ltp-accelerated-pipeline")

STUDENT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
TEACHER_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"


@app.cls(
    gpu="L4",
    image=vllm_image,
    secrets=[Secret.from_name("huggingface-secret")],
    max_containers=10,
)
class StudentGenerator:
    @modal.enter()
    def setup(self):
        from vllm import LLM

        self.llm = LLM(
            model=STUDENT_MODEL, max_model_len=4096, gpu_memory_utilization=0.4
        )

    @modal.method()
    def generate_batch(self, questions: list[dict], n_samples: int = 4) -> list[dict]:
        from vllm import SamplingParams

        system = "You are a helpful mathematical reasoning assistant."
        instruction = "Please reason step-by-step and then output your final answer enclosed in \\boxed{}."
        prompts = [
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\nQuestion: {q['question']}\n\n{instruction}<|im_end|>\n"
            f"<|im_start|>assistant\n"
            for q in questions
        ]
        # Diverse sampling -> richer failure modes per question
        sp = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=1536, n=n_samples)
        outputs = self.llm.generate(prompts, sp)
        results = []
        for q, out in zip(questions, outputs):
            for i, o in enumerate(out.outputs):
                results.append(
                    {
                        "question": q["question"],
                        "answer": q["answer"],
                        "trajectory": o.text,
                        "sample_idx": i,
                    }
                )
        return results


@app.cls(
    gpu="L4",
    image=vllm_image,
    secrets=[Secret.from_name("huggingface-secret")],
    max_containers=10,
    scaledown_window=60,
)
class TeacherAnnotator:
    @modal.enter()
    def setup(self):
        from vllm import LLM

        self.llm = LLM(
            model=TEACHER_MODEL, gpu_memory_utilization=0.9, max_model_len=16384
        )
        self.tokenizer = self.llm.get_tokenizer()

    @modal.method()
    def annotate_batch(self, items: list[dict]) -> list[dict]:
        from vllm import SamplingParams
        import re

        try:
            from math_verify import parse, verify

            def equivalent(pred: str, gold: str) -> bool:
                try:
                    return bool(verify(parse(gold), parse(pred)))
                except Exception:
                    return False
        except Exception:
            equivalent = None

        def extract_boxed(s: str) -> str:
            # Brace-balanced extraction of last \boxed{...}
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

        def is_correct(traj: str, gold: str) -> bool:
            if equivalent is not None and equivalent(traj, gold):
                return True
            p, g = extract_boxed(traj), extract_boxed(gold)
            return p == g and p != ""

        # 1. Verify correctness up-front; only incorrect items hit the LLM
        to_annotate = []
        for item in items:
            item["is_correct"] = is_correct(item["trajectory"], item["answer"])
            if item["is_correct"]:
                item["teacher_analysis"] = "Correct trajectory, no annotation needed."
                item["pof_quote"] = ""
            else:
                item["teacher_analysis"] = ""
                item["pof_quote"] = ""
                to_annotate.append(item)

        if not to_annotate:
            return items

        # 2. Batched annotation — apply chat template for DeepSeek-R1-Distill
        system_msg = (
            "You are a meticulous math tutor. You analyze a student's incorrect reasoning "
            "and identify the first step that contains a logical or computational error."
        )

        def build_user(it):
            return (
                f"Question: {it['question']}\n\n"
                f"Ground Truth Solution: {it['answer']}\n\n"
                f"Student's reasoning trace:\n\"\"\"\n{it['trajectory']}\n\"\"\"\n\n"
                "The student's final answer is wrong. Find the FIRST step where the student's "
                "reasoning breaks (logical error, wrong formula, arithmetic mistake, misread problem).\n\n"
                "Output STRICTLY in this XML format and nothing else after the closing tag:\n"
                "<analysis>Brief explanation of the first error and why it is wrong.</analysis>\n"
                "<pof_quote>Copy the exact verbatim substring from the student's reasoning where "
                "the error first appears. Do not paraphrase. Do not summarize. Copy characters exactly.</pof_quote>"
            )

        prompts = [
            self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": build_user(it)},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for it in to_annotate
        ]
        sp = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=4096)
        outputs = self.llm.generate(prompts, sp)

        for it, out in zip(to_annotate, outputs):
            resp = out.outputs[0].text
            # Strip DeepSeek-R1 think block (correct tag is <think>, not <thought>)
            clean = re.sub(r"<think>.*?</think>", "", resp, flags=re.DOTALL).strip()
            # Fallback: strip dangling opening <think> if generation was truncated
            if clean.startswith("<think>"):
                clean = clean[len("<think>") :].strip()
            a_m = re.search(r"<analysis>(.*?)</analysis>", clean, re.DOTALL)
            p_m = re.search(r"<pof_quote>(.*?)</pof_quote>", clean, re.DOTALL)

            analysis = (
                (a_m.group(1).strip() if a_m else "")
                .replace("Ġ", " ")
                .replace("Ċ", "\n")
            )
            pof = (
                (p_m.group(1).strip() if p_m else "")
                .replace("Ġ", " ")
                .replace("Ċ", "\n")
            )

            # Ground pof_quote: must be substring of trajectory. If not, mark ungrounded.
            pof_grounded = bool(pof) and pof in it["trajectory"]
            # Mark annotation validity
            it["teacher_analysis"] = analysis
            it["pof_quote"] = pof if pof_grounded else ""
            it["annotation_valid"] = bool(a_m) and pof_grounded

        return items


@app.local_entrypoint()
def main(samples: int = 500, repo: str = "hamzabouajila/ltp-trajectories-full-v3"):
    from src.data import load_math500_subset
    from datasets import Dataset
    import time

    data = load_math500_subset(num_samples=samples)

    print("STEP 1: Parallel Generation (L4 GPUs)...")
    gen = StudentGenerator()
    batches = [data[i : i + 50] for i in range(0, len(data), 50)]
    start = time.time()
    batch_results = list(gen.generate_batch.map(batches))
    results = [item for batch in batch_results for item in batch]

    print("STEP 2: Parallel Annotation (L4 GPUs)...")
    ann = TeacherAnnotator()
    ann_batches = [results[i : i + 50] for i in range(0, len(results), 50)]
    ann_results = list(ann.annotate_batch.map(ann_batches, order_outputs=True))
    final_data = [item for batch in ann_results for item in batch]

    # Quality stats
    n_total = len(final_data)
    n_correct = sum(1 for x in final_data if x.get("is_correct"))
    n_valid_ann = sum(1 for x in final_data if x.get("annotation_valid"))
    n_incorrect = n_total - n_correct
    print(
        f"Stats: total={n_total} correct={n_correct} incorrect={n_incorrect} "
        f"valid_annotations={n_valid_ann}/{n_incorrect}"
    )

    print(f"Pushing to HF: {repo}")
    Dataset.from_list(final_data).push_to_hub(repo)
    print(f"Done in {time.time() - start:.1f}s!")
