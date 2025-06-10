import os
import json
import re
import torch
from tqdm import tqdm
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoProcessor
import random
import numpy as np

# Set random seed for reproducibility
random.seed(42)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Set cache directory to workspace
cache_dir = "/workspace/model_cache"
os.makedirs(cache_dir, exist_ok=True)
print(f"Using cache directory: {cache_dir}")

# Model configuration
MODEL_ID = "llava-hf/llava-1.5-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=cache_dir)
processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=cache_dir)
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    cache_dir=cache_dir
).to(device)

# Dataset configuration
SPLIT = "val"   # Using validation split
SHAPES_ROOT = "SHAPES_dataset"
OUT_PATH = f"pre_swirl_SHAPES_{SPLIT}_results.jsonl"
NUM_SAMPLES = 200

# Load questions, images, and true yes/no answers
query_file = os.path.join(SHAPES_ROOT, f"{SPLIT}.query_str.txt")
output_file = os.path.join(SHAPES_ROOT, f"{SPLIT}.output")
images_file = os.path.join(SHAPES_ROOT, f"{SPLIT}.input.npy")

# Load the data
print("Loading dataset...")
qs_all = open(query_file).read().splitlines()
trues_all = open(output_file).read().splitlines()
images_all = np.load(images_file)  # Load numpy array of images

# Print dataset info
print(f"Total questions in {SPLIT} split: {len(qs_all)}")
print(f"Total images in {SPLIT} split: {len(images_all)}")
print(f"Total labels in {SPLIT} split: {len(trues_all)}")

# Sanity check
assert len(qs_all) == len(trues_all) == len(images_all), "Dataset files have mismatched lengths!"

# Zip all and shuffle deterministically
zipped = list(zip(qs_all, images_all, trues_all))
random.shuffle(zipped)

# Select the first NUM_SAMPLES
qs, imgs, trues = zip(*zipped[:NUM_SAMPLES])

print(f"Selected {len(qs)} random samples for evaluation")

def build_prompt(question: str) -> str:
    """Build the prompt for LLaVA"""
    return f"<image>\nQuestion: {question}\nAnswer:"

def normalize(answer):
    """Normalize answer to yes/no"""
    a = answer.strip().lower()
    if a in ["yes", "true"]: return "yes"
    if a in ["no", "false"]: return "no"
    return "N/A"

# Create directory to save images
os.makedirs("images", exist_ok=True)

# Save dataset info
with open("dataset_info.txt", "w") as f:
    f.write(f"Dataset: {SPLIT}\n")
    f.write(f"Total samples: {len(qs)}\n\n")
    for i, (q, t) in enumerate(zip(qs, trues)):
        f.write(f"Sample {i}:\n")
        f.write(f"Question: {q}\n")
        f.write(f"True answer: {t}\n\n")

with open(OUT_PATH, "w") as fout:
    for i, (question, image, true_ans) in enumerate(tqdm(zip(qs, imgs, trues), total=len(qs))):
        # Convert numpy array to PIL Image and ensure it's RGB
        image_pil = Image.fromarray((image * 255).astype(np.uint8))
        if image_pil.mode != 'RGB':
            image_pil = image_pil.convert('RGB')
        
        # Save original 30x30 image
        image_pil.save(f"images/image_{i}_original.png")
        
        # Resize image to 224x224 (LLaVA's expected size)
        image_pil = image_pil.resize((224, 224), Image.Resampling.LANCZOS)
        
        # Save the resized image
        image_pil.save(f"images/image_{i}_resized.png")
        
        # Build prompt
        prompt = build_prompt(question)
        
        # Process image and text with LLaVA processor
        inputs = processor(
            text=prompt,
            images=image_pil,
            return_tensors="pt",
            padding=True
        ).to(device)
        
        # Generate response
        outputs = model.generate(
            **inputs,
            max_new_tokens=16,    # assume yes/no is short
            num_beams=1,          # for speed
            do_sample=False
        )
        
        pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract answer
        last_match = re.search(r"(yes|no)\s*$", pred.strip().lower())
        ans = last_match.group(1) if last_match else "N/A"
        
        # Compare to SHAPES ground truth
        ans_norm = normalize(ans)
        true_norm = normalize(true_ans)
        success = (ans_norm == true_norm)
        
        # Write results
        result = {
            "question": question,
            "true_answer": true_ans,
            "model_output": pred,
            "extracted_answer": ans,
            "success": success
        }
        fout.write(json.dumps(result) + "\n")
        
        # Print first few examples for verification
        if i < 5:
            print(f"\nExample {i}:")
            print(f"Question: {question}")
            print(f"True answer: {true_ans}")
            print(f"Model output: {pred}")
            print(f"Success: {success}")

print("Done — results written to", OUT_PATH)

# Compute accuracy
total, correct = 0, 0
with open(OUT_PATH) as f:
    for line in f:
        rec = json.loads(line)
        total += 1
        correct += rec["success"]

print(f"Accuracy: {correct}/{total} = {correct/total*100:.2f}%")