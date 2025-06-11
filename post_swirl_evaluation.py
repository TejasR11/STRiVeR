import os
import json
import re
import torch
from tqdm import tqdm
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoProcessor
import random
import numpy as np
from safetensors.torch import load_file

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
BASE_MODEL_ID = "/root/model_cache/models--llava-hf--llava-1.5-7b-hf/snapshots/b234b804b114d9e37bb655e11cbbb5f5e971b7a9" # The original model from CACHE
FINETUNED_MODEL_PATH = "/root/llava-swirl-shapes-rlhf-final" # Our fine-tuned weights

# Load processor and tokenizer from the fine-tuned path (as it has the complete config)
tokenizer = AutoTokenizer.from_pretrained(FINETUNED_MODEL_PATH, cache_dir=cache_dir)
processor = AutoProcessor.from_pretrained(FINETUNED_MODEL_PATH, cache_dir=cache_dir)

# 1. Load the base model architecture with the correct dtype
model = AutoModelForVision2Seq.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    local_files_only=True # Force use of local files
)

# 2. Resize the token embeddings to match our trained model
model.resize_token_embeddings(32001)

# 3. Load the fine-tuned weights from our saved state_dict
# The state_dict contains only the learned weights, not the model architecture.
state_dict = {}
# This model is sharded, so we need to load all the pieces
for f in os.listdir(FINETUNED_MODEL_PATH):
    if f.endswith(".safetensors"):
        state_dict.update(load_file(os.path.join(FINETUNED_MODEL_PATH, f), device="cpu"))

# Manually fix the keys by ADDING the required prefixes to match the base model structure.
# This is necessary because the saved state_dict has a different structure.
corrected_state_dict = {}
for k, v in state_dict.items():
    if k.startswith("language_model.") or k.startswith("multi_modal_projector."):
        new_key = "model." + k
        corrected_state_dict[new_key] = v
    elif k.startswith("vision_tower."):
        new_key = "model." + k
        corrected_state_dict[new_key] = v
    else:
        # This case should ideally not be hit if all keys are handled
        corrected_state_dict[k] = v

model.load_state_dict(corrected_state_dict, strict=False)

# Dataset configuration
SPLIT = "val"   # Using validation split
SHAPES_ROOT = "/root/SHAPES_dataset" # Use absolute path
OUT_PATH = f"pre_swirl_SHAPES_{SPLIT}_results.jsonl"

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

# Use the full validation set for a stable evaluation
qs, imgs, trues = qs_all, images_all, trues_all
print(f"Selected {len(qs)} samples for evaluation (full dataset)")

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
        last_match = re.search(r"\b(yes|no)\b", pred.lower())
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