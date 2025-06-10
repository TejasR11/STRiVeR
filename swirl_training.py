# -*- coding: utf-8 -*-
# SWiRL_Training.py - Adapted for local VS Code environment
# Original Colab notebook code and magics removed

import os, re, json, torch, numpy as np
from PIL import Image
import openai
from transformers import (
    Qwen2_5OmniProcessor,
    Qwen2_5OmniForConditionalGeneration,
    AutoProcessor,
    AutoModelForCausalLM
)
from qwen_omni_utils import process_mm_info
from peft import LoraConfig, get_peft_model
from trl import PPOConfig, PPOTrainer
from huggingface_hub import login
import torch.optim as optim
from torch.amp import autocast, GradScaler
import torch.nn.utils

# Set Hugging Face cache directory to workspace
os.environ['TRANSFORMERS_CACHE'] = '/workspace/hf_cache'
os.environ['HF_HOME'] = '/workspace/hf_cache'
os.environ['HF_DATASETS_CACHE'] = '/workspace/hf_cache'

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
openai.api_key = os.getenv("OPENAI_API_KEY")

# Set the base directory to /workspace
BASE_DIR = "/workspace"
SHAPES_ROOT = os.path.join(BASE_DIR, "SHAPES_dataset")
SPLIT = "val"
IMAGE_DIR = os.path.join(SHAPES_ROOT, SPLIT, "images")

# Model configuration
MODEL_ID = "Qwen/Qwen2.5-Omni-3B"
VIS_VER_ID = "gpt-4-vision-preview"

print("Loading processor...")
try:
    processor = Qwen2_5OmniProcessor.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        cache_dir="/workspace/hf_cache"
    )
    print("Processor loaded successfully!")
except Exception as e:
    print(f"Error loading processor: {str(e)}")
    raise

from transformers import GenerationMixin

class QwenNoAudio(Qwen2_5OmniForConditionalGeneration):
    def generate(self, *args, **kwargs):
        # Remove audio-related parameters
        kwargs.pop("return_video", None)
        kwargs.pop("return_audio", None)
        kwargs["use_audio_in_video"] = False
        
        # Add generation parameters
        kwargs["do_sample"] = True
        kwargs["temperature"] = 0.7
        kwargs["max_new_tokens"] = 100
        kwargs["pad_token_id"] = self.config.pad_token_id
        kwargs["eos_token_id"] = self.config.eos_token_id
        
        return super().generate(*args, **kwargs)

from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# Add error handling for model loading
try:
    print("Loading policy model...")
    policy = QwenNoAudio.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        cache_dir="/workspace/hf_cache"
    )
    print("Policy model loaded successfully!")

    print("Loading reference model...")
    ref_qwen = QwenNoAudio.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        cache_dir="/workspace/hf_cache"
    )
    ref_qwen.eval()
    print("Reference model loaded successfully!")

except Exception as e:
    print(f"Error loading models: {str(e)}")
    print("Please ensure you have enough disk space and GPU memory.")
    raise

lora_cfg = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj","k_proj","v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
policy = get_peft_model(policy, lora_cfg)

# Load and process the dataset
def load_shapes_dataset(split="val"):
    query_file = os.path.join(SHAPES_ROOT, f"{split}.query_str.txt")
    output_file = os.path.join(SHAPES_ROOT, f"{split}.output")
    input_file = os.path.join(SHAPES_ROOT, f"{split}.input.npy")
    
    # Load queries and answers
    queries = open(query_file).read().splitlines()
    answers = open(output_file).read().splitlines()
    
    # Load images from numpy array
    images = np.load(input_file)
    
    # Convert numpy array to list of image paths
    image_paths = []
    for i in range(len(images)):
        img_path = os.path.join(IMAGE_DIR, f"{i:05d}.png")
        # Save numpy array as image if it doesn't exist
        if not os.path.exists(img_path):
            os.makedirs(IMAGE_DIR, exist_ok=True)
            Image.fromarray(images[i]).save(img_path)
        image_paths.append(f"{i:05d}.png")
    
    return [
        {"question": q, "image": img, "true_answer": ans}
        for q, img, ans in zip(queries, image_paths, answers)
    ]

# Load the dataset
examples = load_shapes_dataset(SPLIT)
train_subset = examples[:20]  # Using a small subset for testing

def build_prompt(question, image_path, interactions=[]):
    system = {
        "role":"system","content":[{"type":"text","text":
            "You are a visual reasoning assistant for DeepMind SHAPES. "
            "You will be shown a 30×30 RGB image and a yes/no spatial question. "
            "Think step-by-step, up to 5 numbered steps, then wrap exactly one "
            "<answer>yes</answer> or <answer>no</answer> and output nothing else.\n"
            "<end_of_turn>\n"
        }]
    }
    conv = [system]
    for model_turn, user_turn in interactions:
        conv += [
            {"role":"model","content":[{"type":"text","text":model_turn}]},
            {"role":"user", "content":[{"type":"text","text":user_turn}]}
        ]
    conv += [
        {"role":"user","content":[{"type":"image","image":image_path}]},
        {"role":"user","content":[{"type":"text","text":f"Question: {question}"}]}
    ]
    return conv

# 12) GPT-4 Vision Verifier Function
def verify_step_with_openai(question, image_path, step_text):
    # Read image bytes
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    # Call the OpenAI multmodal ChatCompletion API
    resp = openai.ChatCompletion.create(
        model="gpt-4-vision-preview",
        messages=[
            {"role":"system","content":
             "You are a visual reasoning expert. Given an image, a question, "
             "and one reasoning step, reply ONLY CORRECT or INCORRECT."},
            {"role":"user","content":
             f"Question: {question}\nStep: {step_text}"}
        ],
        files=[{"name":"image.png","data":img_bytes}],
        temperature=0.0,
    )
    verdict = resp.choices[0].message.content.strip().upper()
    return verdict.startswith("CORRECT")

def data_collator(batch):
    # batch is a list of examples: each has keys question, image, true_answer
    texts = []
    imgs = []
    
    for ex in batch:
        # build the conversation for each example
        conv = build_prompt(
            ex["question"],
            os.path.join(IMAGE_DIR, ex["image"]),
            interactions=[]
        )
        # Convert conversation to text format
        text = processor.apply_chat_template(
            conv, 
            add_generation_prompt=True, 
            tokenize=False
        )
        # If text is a list, extract the first element
        if isinstance(text, list):
            text = text[0]
        if not isinstance(text, str):
            text = str(text)
        texts.append(text)
        # load the PIL image
        img = Image.open(os.path.join(IMAGE_DIR, ex["image"]))
        imgs.append(img)
    # Debug print
    print("[DEBUG] First 2 texts:", texts[:2])
    print("[DEBUG] Types:", [type(t) for t in texts[:2]])
    # Process images and texts
    inputs = processor(
        text=texts,
        images=imgs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    return inputs

def extract_steps_and_answer(generated):
    lines = generated.strip().splitlines()
    steps = [l for l in lines if l.lower().startswith("step")]
    m = re.search(r"<answer>(yes|no)</answer>", generated, re.IGNORECASE)
    ans = m.group(1).lower() if m else None
    return steps, ans

# 14) PPO Reward Function
def reward_fn(samples, responses):
    rewards = []
    for sample, resp in zip(samples, responses):
        steps, pred = extract_steps_and_answer(resp)
        if not steps or not pred:  # Handle empty responses
            rewards.append(-1)
            continue
            
        correct_steps = sum(
            verify_step_with_openai(
                sample["question"],
                os.path.join(IMAGE_DIR, sample["image"]),
                step
            ) for step in steps
        )
        final_ok = (pred == sample["true_answer"].lower())
        rewards.append(correct_steps + (1 if final_ok else -1))
    return rewards

def collate_fn(batch):
    texts, images = [], []
    for ex in batch:
        conv = build_prompt(
            ex["question"],
            os.path.join(IMAGE_DIR, ex["image"]),
            interactions=[]
        )
        txt = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        texts.append(txt)
        images.append(Image.open(os.path.join(IMAGE_DIR, ex["image"])))
    return texts, images

# Set up optimizer for LoRA parameters only
optimizer = optim.AdamW(policy.parameters(), lr=1e-5)

num_epochs = 1  # You can increase this for more training
print("Starting custom single-example training loop...")

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    total_reward = 0
    optimizer.zero_grad()  # Zero gradients at start of epoch
    
    for i, sample in enumerate(train_subset):
        # Prepare prompt and image
        conv = build_prompt(
            sample["question"],
            os.path.join(IMAGE_DIR, sample["image"]),
            interactions=[]
        )
        text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        if isinstance(text, list):
            text = text[0]
        img = Image.open(os.path.join(IMAGE_DIR, sample["image"]))
        
        # Process inputs
        inputs = processor(
            text=[text],
            images=[img],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate response with mixed precision
        with torch.no_grad(), autocast(device_type='cuda'):
            # Generate with specific parameters
            output_ids = policy.generate(
                **inputs,
                do_sample=True,
                temperature=0.7,
                max_new_tokens=100,
                pad_token_id=policy.config.pad_token_id,
                eos_token_id=policy.config.eos_token_id
            )
            
            # Handle the output robustly
            if isinstance(output_ids, torch.Tensor):
                if output_ids.dim() > 1:
                    output_ids = output_ids[0]
                output_ids_np = output_ids.cpu().numpy()
            else:
                # If it's a list or already a numpy array
                output_ids_np = np.array(output_ids)
                if len(output_ids_np.shape) > 1:
                    output_ids_np = output_ids_np[0]
        
        # Decode the response
        response = processor.tokenizer.decode(output_ids_np, skip_special_tokens=True)
        
        # Compute reward
        reward = reward_fn([sample], [response])[0]
        total_reward += reward
        
        # Compute loss with mixed precision
        with autocast(device_type='cuda'):
            loss = -torch.tensor(reward, dtype=torch.float32, device=device)
            loss = loss / gradient_accumulation_steps  # Normalize loss
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        
        # Update weights if we've accumulated enough gradients
        if (i + 1) % gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        # Print progress
        if (i + 1) % 5 == 0:  # Print every 5 samples
            print(f"Sample {i+1}/{len(train_subset)}: reward={reward:.2f}, loss={loss.item():.2f}")
            print(f"Response: {response[:100]}...")  # Print first 100 chars of response
            
        # Clear CUDA cache periodically
        if (i + 1) % 10 == 0:
            torch.cuda.empty_cache()
    
    print(f"Epoch {epoch+1} total reward: {total_reward:.2f}")

# Save the model
output_dir = "swirl-shapes-ppo-custom"
policy.save_pretrained(output_dir)
print(f"Model saved to {output_dir}")

