import os
import json
import re
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoProcessor
import openai
import random
from torch.utils.data import DataLoader

# Set random seed for reproducibility
random.seed(42)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Set cache directory to workspace
cache_dir = "/root/model_cache"
os.makedirs(cache_dir, exist_ok=True)
print(f"Using cache directory: {cache_dir}")

# Model configuration
MODEL_ID = "llava-hf/llava-1.5-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=cache_dir)
processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=cache_dir)

# The default pad_token for this model is <pad> with id 32001, which is out-of-bounds
# for the vocabulary size of 32000. We'll use the eos_token as the pad_token.
# This will set the pad_token_id to the eos_token_id (2), which is valid.
tokenizer.pad_token = tokenizer.eos_token
processor.tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForVision2Seq.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    cache_dir=cache_dir
)

model.resize_token_embeddings(32001)

# Dataset configuration
SPLIT = "train.large"
SHAPES_ROOT = "/root/SHAPES_dataset"
NUM_SAMPLES = 10000

# Load questions, images, and true yes/no answers
# Handle the special case for the 'train.large' split file naming
if SPLIT == "train.large":
    # The 'ls' command showed the main training files are in a subdirectory
    query_file = os.path.join(SHAPES_ROOT, "train.large.query_str.txt")
    output_file = os.path.join(SHAPES_ROOT, "train.large.output")
    images_file = os.path.join(SHAPES_ROOT, "train.large.input.npy")
else:
    query_file = os.path.join(SHAPES_ROOT, f"{SPLIT}.query_str.txt")
    output_file = os.path.join(SHAPES_ROOT, f"{SPLIT}.output")
    images_file = os.path.join(SHAPES_ROOT, f"{SPLIT}.input.npy")

print("Loading dataset...")
try:
    qs_all = open(query_file).read().splitlines()
    trues_all = open(output_file).read().splitlines()
    images_all = np.load(images_file)  # shape = (N, 2700)
except FileNotFoundError:
    print(f"Error: Dataset files not found for split '{SPLIT}'. Please check the file paths: {query_file}, {output_file}, {images_file}")
    exit()

assert len(qs_all) == len(trues_all) == len(images_all), "Dataset files have mismatched lengths!"

# Zip all and shuffle deterministically
zipped = list(zip(qs_all, images_all, trues_all))
random.shuffle(zipped)
qs, imgs, trues = zip(*zipped[:NUM_SAMPLES])

class ShapesDataset(torch.utils.data.Dataset):
    def __init__(self, qs, imgs, trues):
        self.qs = qs
        self.imgs = imgs
        self.trues = trues
    def __len__(self):
        return len(self.qs)
    def __getitem__(self, idx):
        question = self.qs[idx]
        # Convert exactly like pre_swirl_shapes.py
        image = self.imgs[idx].reshape(30, 30, 3)
        image_pil = Image.fromarray((image * 255).astype(np.uint8))
        if image_pil.mode != 'RGB':
            image_pil = image_pil.convert('RGB')
        # Resize to LLaVA's expected size
        image_pil = image_pil.resize((224, 224), Image.Resampling.LANCZOS)
        true_ans = self.trues[idx]
        return question, image_pil, true_ans

dataset = ShapesDataset(qs, imgs, trues)

# Extract steps and answer from generated text
def extract_steps_and_answer(generated):
    lines = generated.strip().splitlines()
    steps = [l for l in lines if l.lower().startswith("step")]
    m = re.search(r"<answer>(yes|no)</answer>", generated, re.IGNORECASE)
    ans = m.group(1).lower() if m else None
    return steps, ans

# Build prompt for LLaVA
def build_prompt(question, interactions):
    # Use the exact format from working pre_swirl_shapes.py but with reasoning request
    prompt = f"""<image>
Question: {question}

Please analyze the image step by step:
Step 1: [describe what you see]
Step 2: [analyze the spatial relationships]
Step 3: [determine your answer]

Then provide your final answer as <answer>yes</answer> or <answer>no</answer>

Answer:"""
    return prompt

# --- Verifier using OpenAI GPT-4 Vision ---
def verify_step_with_openai(question, image_pil, step_text):
    import io
    import base64
    buf = io.BytesIO()
    image_pil.save(buf, format='PNG')
    img_bytes = buf.getvalue()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("WARNING: OPENAI_API_KEY environment variable is not set. Verifier will always return False.")
        return False
    openai.api_key = openai_api_key
    # Encode image as base64 for OpenAI vision API
    img_b64 = base64.b64encode(img_bytes).decode('utf-8')
    img_data_url = f"data:image/png;base64,{img_b64}"
    messages = [
        {"role": "system", "content": "You are a visual reasoning expert. Given an image, a question, and one reasoning step, reply ONLY CORRECT or INCORRECT."},
        {"role": "user", "content": [
            {"type": "text", "text": f"Question: {question}\nStep: {step_text}"},
            {"type": "image_url", "image_url": {"url": img_data_url}}
        ]}
    ]
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )
        verdict = response.choices[0].message.content.strip().upper()
        return verdict.startswith("CORRECT")
    except Exception as e:
        print(f"Verifier error: {e}")
        return False

# --- Custom RLHF Training Loop ---
BATCH_SIZE = 1
EPOCHS = 3
SAVE_EVERY = 500 # We'll save every 500 steps
CHECKPOINT_PATH = "/root/llava-swirl-shapes-rlhf-checkpoint" # Overwrite this path
ACCUMULATION_STEPS = 4

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

def collate_fn(batch):
    return list(zip(*batch))  # returns (questions, images, true_answers) as lists

train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

print("Starting custom RLHF training loop...")
step = 0
for epoch in range(EPOCHS):
    for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")):
        questions, images, true_answers = batch
        # Build prompts (revert to original logic)
        prompts = [build_prompt(q, []) for q in questions]
        # Use the exact approach from working pre_swirl_shapes.py
        # Process one image at a time to avoid batching issues
        inputs = processor(
            text=prompts[0],  # Process first (and only) item since BATCH_SIZE=1
            images=images[0],
            return_tensors="pt",
            padding=True
        ).to(device)
        
        # Ensure all inputs match the model's dtype
        inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)

        # Generate responses
        try:
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=200,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                )
        except RuntimeError as e:
            print(f"CUDA error during model.generate: {e}")
            continue  # Skip this batch
        responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        # Compute rewards using verifier
        rewards = []
        for q, img, true_ans, resp in zip(questions, images, true_answers, responses):
            steps, pred = extract_steps_and_answer(resp)
            step_results = []
            for step_text in steps:
                result = verify_step_with_openai(q, img, step_text)
                step_results.append(result)
            
            true_ans_bool = (true_ans.lower() == 'true')
            pred_bool = (pred == 'yes')
            final_ok = (pred_bool == true_ans_bool)

            # Reward: +1 per correct step, +5 for correct final answer, -5 for wrong final answer
            reward = sum(step_results) + (5 if final_ok else -5)
            rewards.append(reward)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        # Compute logprobs for generated responses
        # (Simple REINFORCE-style loss)
        logprobs = []
        for i_log in range(len(generated_ids)):
            prompt_len = inputs["input_ids"][i_log].shape[0]
            full_ids = generated_ids[i_log]
            
            labels = full_ids.clone()
            labels[:prompt_len] = -100
            
            input_ids = full_ids.unsqueeze(0)
            labels = labels.unsqueeze(0)
            
            try:
                out = model(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    pixel_values=inputs["pixel_values"][i_log:i_log+1],
                    labels=labels
                )
                logprob = -out.loss
            except RuntimeError as e:
                print(f"CUDA error during forward pass: {e}")
                logprob = torch.tensor(0.0, device=device)
            logprobs.append(logprob)
        logprobs = torch.stack(logprobs)
        # REINFORCE loss: maximize reward * logprob
        loss = -(rewards * logprobs).mean()
        
        # Normalize the loss for accumulation
        loss = loss / ACCUMULATION_STEPS
        
        loss.backward()

        if (i + 1) % ACCUMULATION_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()
            print(f"Epoch {epoch+1}, Step {step+1}, Loss: {loss.item() * ACCUMULATION_STEPS:.4f}, Reward: {rewards.mean().item():.2f}")
            step += 1
            if step % SAVE_EVERY == 0:
                model.save_pretrained(CHECKPOINT_PATH) # Use the constant path
                tokenizer.save_pretrained(CHECKPOINT_PATH) # Also save the tokenizer
                print(f"Model checkpoint saved to {CHECKPOINT_PATH} at step {step}")

print("Training complete.")
model.save_pretrained("/root/llava-swirl-shapes-rlhf-final")
tokenizer.save_pretrained("/root/llava-swirl-shapes-rlhf-final") # Also save tokenizer
print("Final model saved.")