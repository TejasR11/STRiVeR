# Fine-tuned LLaVA Model Evaluation on SHAPES Dataset

This repository contains the code for evaluating a fine-tuned LLaVA-1.5 7B model on the SHAPES dataset.

## Setup and Installation

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd <your-repo-name>
```

### 2. Install Dependencies

It is recommended to use a virtual environment.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Download Models and Dataset

The evaluation script requires access to the base LLaVA model, the fine-tuned model weights, and the SHAPES dataset. Due to their size, they are not included in this repository.

**Important:** The script uses absolute paths to load the models and data. You must place the files in the exact locations specified below, or update the paths in `pre_swirl_shapes.py`.

- **Base Model (LLaVA-1.5 7B)**:
  - You need the `llava-hf/llava-1.5-7b-hf` model from Hugging Face.
  - The script expects it to be at: `/root/model_cache/models--llava-hf--llava-1.5-7b-hf/snapshots/b234b804b114d9e37bb655e11cbbb5f5e971b7a9`
  - You can download it using `git lfs`:
    ```bash
    # Make sure you have git-lfs installed (https://git-lfs.com)
    git lfs install
    git clone https://huggingface.co/llava-hf/llava-1.5-7b-hf
    # You will need to manually create the directory structure and move the files if the snapshot hash is different.
    ```

- **Fine-tuned Model Weights**:
  - The fine-tuned weights should be placed at: `/root/llava-swirl-shapes-rlhf-final`
  - You will need to obtain these weights separately.

- **SHAPES Dataset**:
  - The SHAPES dataset files should be located at: `/root/SHAPES_dataset`
  - The directory should contain the following files for the `val` split:
    - `val.query_str.txt`
    - `val.output`
    - `val.input.npy`

## Running the Evaluation

Once the setup is complete, you can run the evaluation script:

```bash
python pre_swirl_shapes.py
```

The script will:
1. Load the base model and fine-tuned weights.
2. Load the SHAPES validation dataset.
3. Preprocess the images and formulate prompts.
4. Run inference on each sample.
5. Extract and normalize the answers.
6. Calculate and print the final accuracy.
7. Save the detailed results to `pre_swirl_SHAPES_val_results.jsonl`.

The script will also save the first 5 original and resized images to the `images/` directory for verification purposes. 
