# src/modal_annotator.py
import modal
from modal import Image, App

# Define the Modal image with vLLM
vllm_image = Image.debian_slim().pip_install(
    "vllm==0.4.3", 
    "datasets", 
    "huggingface_hub"
)

app = App("ltp-teacher-annotator")

# We will use Qwen2.5-72B-Instruct for annotation
MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct"

@app.cls(image=vllm_image, gpu="A100", secrets=[modal.Secret.from_name("huggingface-secret")])
class TeacherAnnotator:
    @modal.enter()
    def setup(self):
        from vllm import LLM
        # Load model using vLLM for high throughput
        self.llm = LLM(model=MODEL_NAME, tensor_parallel_size=1)
        
    @modal.method()
    def annotate_batch(self, items: list[dict]) -> list[dict]:
        from vllm import SamplingParams
        
        prompts = []
        for item in items:
            if item["is_correct"]:
                prompts.append("") # Skip correct trajectories
                continue
                
            # Construct a prompt asking the model to find the Point of Failure (PoF)
            prompt = (
                f"Question: {item['question']}\n"
                f"Ground Truth: {item['ground_truth']}\n"
                f"Student's reasoning:\n{item['trajectory']}\n\n"
                "The student's reasoning is incorrect. Analyze the student's reasoning step-by-step. "
                "Identify the exact sentence where the first logical or mathematical error occurs. "
                "Output your analysis, and end with: 'Point of Failure: <exact sentence from student>'"
            )
            prompts.append(prompt)
            
        sampling_params = SamplingParams(temperature=0.0, max_tokens=512)
        
        # We filter out empty prompts for vLLM, keeping track of indices
        valid_indices = [i for i, p in enumerate(prompts) if p != ""]
        valid_prompts = [prompts[i] for i in valid_indices]
        
        outputs = []
        if valid_prompts:
            outputs = self.llm.generate(valid_prompts, sampling_params)
            
        # Merge back
        results = []
        out_idx = 0
        for i, item in enumerate(items):
            if i in valid_indices:
                teacher_analysis = outputs[out_idx].outputs[0].text
                out_idx += 1
                
                # Simple extraction of the PoF quote (can be refined)
                pof_quote = ""
                if "Point of Failure:" in teacher_analysis:
                    pof_quote = teacher_analysis.split("Point of Failure:")[-1].strip()
                    
                item["teacher_analysis"] = teacher_analysis
                item["pof_quote"] = pof_quote
            else:
                item["teacher_analysis"] = "Correct trajectory, no annotation needed."
                item["pof_quote"] = ""
            results.append(item)
            
        return results

@app.local_entrypoint()
def main(input_repo: str, output_repo: str):
    from datasets import load_dataset, Dataset
    
    print(f"Loading dataset from {input_repo}...")
    ds = load_dataset(input_repo, split="train")
    items = [row for row in ds]
    
    annotator = TeacherAnnotator()
    
    # Process in batches
    batch_size = 100
    annotated_items = []
    
    print("Starting annotation via Modal...")
    for i in range(0, len(items), batch_size):
        batch = items[i:i+batch_size]
        result_batch = annotator.annotate_batch.remote(batch)
        annotated_items.extend(result_batch)
        print(f"Annotated {len(annotated_items)}/{len(items)} items")
        
    print(f"Pushing annotated dataset to {output_repo}...")
    annotated_ds = Dataset.from_list(annotated_items)
    annotated_ds.push_to_hub(output_repo)
    print("Done!")
