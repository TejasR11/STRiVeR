import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from torch.optim import AdamW
import openai
import os
from typing import List, Dict, Any, Tuple
import json
from tqdm import tqdm
import numpy as np
import base64
from PIL import Image
from io import BytesIO

# Set the cache directory to our workspace
os.environ['TRANSFORMERS_CACHE'] = '/workspace/models/cache'
os.environ['HF_HOME'] = '/workspace/models/cache'

# Set OpenAI API key from file if not set in environment
if not os.getenv('OPENAI_API_KEY'):
    try:
        with open('/workspace/openai_api_key.txt') as f:
            os.environ['OPENAI_API_KEY'] = f.read().strip()
    except Exception as e:
        raise ValueError('OPENAI_API_KEY environment variable not set and openai_api_key.txt not found!')

def build_prompt(question: str, interactions: List[Tuple[str, str]]) -> str:
    """Build the prompt for step-by-step reasoning about shapes"""
    system = (
        "Please help me answer the following yes/no question about shapes. You should answer step by step, "
        "carefully analyzing the image and breaking down your reasoning into clear steps. "
        "You will be penalized if you combine steps together.\n"
        "I will allow you to make up to 5 sequential reasoning steps before answering the question.\n"
        "Please do not repeat steps you have already taken, as this is a waste of time.\n"
        "Once you have enough information, generate your final answer enclosed by <answer>YES</answer> or <answer>NO</answer> tags.\n"
        f"The question is: {question}\n"
        "<end_of_turn>\n"
    )
    trace = ""
    for model_turn, user_turn in interactions:
        trace += f"<start_of_turn>model\n{model_turn}\n<end_of_turn>\n"
        trace += f"<start_of_turn>user\n{user_turn}\n<end_of_turn>\n"
    trace += "<start_of_turn>model\n"
    return system + trace

def verify_step_with_openai(question: str, current_step: str, previous_steps: List[str], image_path: str) -> Tuple[bool, str]:
    """Verify a single reasoning step using OpenAI's visual model"""
    steps_text = "\n".join([f"Step {i+1}: {step}" for i, step in enumerate(previous_steps + [current_step])])
    
    verification_prompt = f"""
Please verify if this reasoning step about shapes is correct:

Question: {question}

Previous steps:
{steps_text}

Is this step logical and correct? Consider:
1. Does it follow from previous steps?
2. Is it relevant to answering the question?
3. Is it factually correct about the shapes in the image?

Respond with only "CORRECT" or "INCORRECT" followed by a brief explanation.
"""
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "system",
                    "content": "You are evaluating a single reasoning step in a solution to a yes/no question about shapes."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": verification_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_path}"}
                        }
                    ]
                }
            ],
            max_tokens=100
        )
        result = response.choices[0].message.content
        return "CORRECT" in result.upper(), result
    except Exception as e:
        print(f"Verification error: {e}")
        return False, str(e)

def verify_final_answer_with_openai(question: str, trajectory: List[str], image_path: str, correct_answer: str) -> Tuple[bool, str]:
    """Verify the final answer using OpenAI's visual model"""
    trajectory_text = "\n".join([f"Step {i+1}: {step}" for i, step in enumerate(trajectory)])
    
    verification_prompt = f"""
Please verify if this solution about shapes is correct:

Question: {question}

Step-by-step solution:
{trajectory_text}

The correct final answer should be: {correct_answer}

Respond with only "CORRECT" or "INCORRECT" based on whether the reasoning and final answer are both correct.
"""
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "system",
                    "content": "You are evaluating a complete solution to a yes/no question about shapes. Verify if the reasoning is logical and the final answer is correct."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": verification_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_path}"}
                        }
                    ]
                }
            ],
            max_tokens=50
        )
        result = response.choices[0].message.content
        return "CORRECT" in result.upper(), result
    except Exception as e:
        print(f"Verification error: {e}")
        return False, str(e)

def image_to_base64(img_arr):
    """Convert a numpy array image to base64 string"""
    img = Image.fromarray((img_arr * 255).astype(np.uint8))  # adjust if needed
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def label_to_yesno(label):
    """Convert 'true'/'false' to 'YES'/'NO'"""
    return "YES" if label.lower() == "true" else "NO"

def main():
    # Load the shapes dataset
    print("Loading SHAPES dataset...")
    images = np.load("SHAPES_dataset/train.tiny.input.npy")
    with open("SHAPES_dataset/train.tiny.query_str.txt") as f:
        questions = [line.strip() for line in f]
    with open("SHAPES_dataset/train.tiny.output") as f:
        labels = [line.strip() for line in f]
    
    # Sanity check
    assert len(images) == len(questions) == len(labels), "Dataset files have mismatched lengths!"
    
    # Start with a small sanity test
    print("Starting with a small sanity test (5 examples)...")
    images = images[:5]
    questions = questions[:5]
    labels = labels[:5]
    
    # Initialize model and processor for Qwen2.5-Omni-3B
    print("Loading Qwen2.5-Omni-3B model...")
    model_name = "Qwen/Qwen2.5-Omni-3B"
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(model_name, torch_dtype="auto", device_map="auto", cache_dir="/workspace/models/cache")
    processor = Qwen2_5OmniProcessor.from_pretrained(model_name, cache_dir="/workspace/models/cache")
    
    # Set up OpenAI API key
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set!")
    
    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=2e-4)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="shapes_swirl_output",
        num_train_epochs=1,  # Start with 1 epoch for testing
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        warmup_steps=10,  # Reduced for testing
        logging_steps=1,  # Log more frequently for testing
        save_steps=5,  # Save more frequently for testing
        save_strategy="steps",
        load_best_model_at_end=False,
        learning_rate=2e-4,
        fp16=True,
        dataloader_pin_memory=False,
        report_to="none",
        gradient_checkpointing=True,
        dataloader_num_workers=0,
        save_total_limit=1,
    )
    
    # Training loop
    print("Starting training loop...")
    model.train()
    total_steps = 0
    correct_steps = 0
    correct_answers = 0
    
    for epoch in range(training_args.num_train_epochs):
        print(f"\nEpoch {epoch + 1}/{training_args.num_train_epochs}")
        
        for i in tqdm(range(len(images))):
            question = questions[i]
            image = images[i]
            correct_answer = label_to_yesno(labels[i])
            image_b64 = image_to_base64(image)
            
            print(f"\nProcessing example: {question}")
            
            # Initialize empty trajectory
            trajectory = []
            step_scores = []
            
            # Generate step-by-step reasoning
            for step_num in range(5):  # Maximum 5 reasoning steps
                print(f"\nGenerating step {step_num + 1}...")
                
                # Build prompt with current trajectory
                prompt = build_prompt(question, [(step, "") for step in trajectory])
                
                # Generate next step
                inputs = processor(prompt, return_tensors="pt", padding=True)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                outputs = model.generate(
                    **inputs,
                    max_length=512,
                    num_return_sequences=1,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True
                )
                response = processor.decode(outputs[0].cpu().numpy().tolist(), skip_special_tokens=True)
                print(f"Generated response: {response}")
                
                # Verify this step
                is_correct, feedback = verify_step_with_openai(question, response, trajectory, image_b64)
                print(f"Step verification: {'CORRECT' if is_correct else 'INCORRECT'}")
                print(f"Feedback: {feedback}")
                step_scores.append(1.0 if is_correct else 0.0)
                
                # Update model based on step verification
                loss = model(**inputs, labels=inputs["input_ids"]).loss
                if not is_correct:
                    loss = -loss  # Negative reinforcement for incorrect steps
                
                # If this is the final answer step, apply a higher weight
                if "<answer>" in response:
                    FINAL_ANSWER_MULTIPLIER = 3.0  # Higher weight for final answer
                    loss = loss * FINAL_ANSWER_MULTIPLIER
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                # Track statistics
                total_steps += 1
                if is_correct:
                    correct_steps += 1
                
                # Check if we got a final answer
                if "<answer>" in response:
                    trajectory.append(response)
                    # Verify final answer
                    is_correct, feedback = verify_final_answer_with_openai(question, trajectory, image_b64, correct_answer)
                    print(f"Final answer verification: {'CORRECT' if is_correct else 'INCORRECT'}")
                    print(f"Feedback: {feedback}")
                    
                    # Apply additional reward/punishment for final answer correctness
                    if is_correct:
                        correct_answers += 1
                        # Positive reinforcement for correct final answer
                        final_answer_loss = model(**inputs, labels=inputs["input_ids"]).loss * 2.0
                        final_answer_loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                    else:
                        # Stronger negative reinforcement for incorrect final answer
                        final_answer_loss = -model(**inputs, labels=inputs["input_ids"]).loss * 4.0
                        final_answer_loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                    break
                
                trajectory.append(response)
            
            # Log progress
            if total_steps % training_args.logging_steps == 0:
                print(f"\nStep {total_steps}")
                print(f"Step accuracy: {correct_steps/total_steps:.2%}")
                print(f"Answer accuracy: {correct_answers/(total_steps//5):.2%}")
            
            # Save checkpoint
            if total_steps % training_args.save_steps == 0:
                print(f"\nSaving checkpoint at step {total_steps}")
                model.save_pretrained(f"{training_args.output_dir}/checkpoint-{total_steps}")

if __name__ == "__main__":
    main()
