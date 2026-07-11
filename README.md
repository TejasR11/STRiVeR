# ARiSe: Adaptive Reinforcement for Spatial Reasoning

ARiSe is a research prototype for improving multimodal spatial reasoning on
SHAPES-style visual question answering tasks. The project explores whether a
vision-language model can be guided toward better step-by-step reasoning by
using verifier feedback as a reinforcement signal.

The core idea is simple: ask a model to reason through a visual yes/no question,
score the generated reasoning and final answer with a verifier, and use that
feedback to reinforce responses that are both visually grounded and correct.

## Motivation

Many vision-language models can answer simple visual questions, but they often
struggle when the answer depends on spatial relationships such as containment,
relative position, object color, or object shape. SHAPES-style datasets are
useful because they isolate these reasoning skills in a controlled setting.

This repository experiments with a small reinforcement loop for that setting:

1. Generate a step-by-step answer for a visual question.
2. Check intermediate reasoning steps with a verifier model.
3. Compare the final yes/no answer against the dataset label.
4. Reward trajectories that contain useful reasoning and a correct final answer.
5. Evaluate the resulting model on held-out SHAPES examples.

## Repository Layout

```text
LLaVA_Swirl.py              Verifier-guided training loop for LLaVA on SHAPES
train_swirl.py              Experimental training loop for Qwen2.5-Omni
pre_swirl_shapes.py         Baseline evaluation before reinforcement training
post_swirl_evaluation.py    Evaluation for a fine-tuned checkpoint
prepare_shapes_dataset.py   Dataset preparation utilities
evaluate_base_model.py      Base-model evaluation utilities
download_model.py           Model download helper
requirements.txt            Python dependencies
```

## Method

The training scripts build prompts that ask the model to analyze an image in
multiple reasoning steps before giving a final answer in a normalized yes/no
format. A verifier model checks whether each generated step is relevant and
visually consistent, while the dataset label supplies the final-answer signal.

The reward combines two pieces of feedback:

- step-level reward for reasoning steps judged correct by the verifier
- answer-level reward or penalty based on whether the final yes/no answer
  matches the SHAPES label

The implementation is intentionally lightweight and experimental. It is meant to
make the training and evaluation loop easy to inspect rather than to serve as a
packaged training framework.

## Setup

Create a Python environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Some scripts use OpenAI vision models as verifiers. Set an API key before
running verifier-guided training:

```bash
export OPENAI_API_KEY=your_api_key_here
```

## Required Artifacts

Large artifacts are not committed to this repository. To reproduce the
experiments, provide the following locally:

- SHAPES dataset files for the desired split
- base vision-language model weights, such as LLaVA 1.5 or Qwen2.5-Omni
- fine-tuned checkpoints, if running post-training evaluation

The SHAPES files are expected to include:

```text
<split>.query_str.txt
<split>.output
<split>.input.npy
```

Update the path constants near the top of the relevant script for your local
dataset, model cache, and checkpoint locations.

## Running Experiments

Run a baseline evaluation before reinforcement training:

```bash
python pre_swirl_shapes.py
```

Run verifier-guided training:

```bash
python LLaVA_Swirl.py
```

or, for the Qwen2.5-Omni experiment:

```bash
python train_swirl.py
```

Evaluate a trained checkpoint:

```bash
python post_swirl_evaluation.py
```

Outputs are written as JSONL records containing the question, ground-truth
answer, generated answer, and success flag. Some scripts also save a small
number of example images for sanity checking preprocessing.

## Limitations

- The repo is an experiment snapshot, not a polished training library.
- Model weights, datasets, and checkpoints are excluded because of size.
- Several scripts were run in a specific compute environment and may need path
  edits before reuse.
- The verifier-guided reward depends on an external vision model, so results can
  vary with model version and prompting.
- The current implementation focuses on SHAPES-style yes/no spatial questions
  rather than broad visual reasoning benchmarks.

## Status

This project is best viewed as an exploratory research prototype for
verifier-guided multimodal reasoning. The code captures the training and
evaluation loop used for the experiment, while leaving larger artifacts and
environment-specific files outside the repository.
