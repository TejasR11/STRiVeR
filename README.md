# SWiRL Training (Local Setup)

## Setup Instructions

1. **Clone the SHAPES dataset**
   - The dataset is automatically downloaded and prepared by running `prepare_shapes_dataset.py`.

2. **Install dependencies**
   - It is recommended to use a Python virtual environment:
     ```bash
     cd /workspace
     python3 -m venv venv
     source venv/bin/activate
     pip install -r requirements.txt
     ```

3. **Prepare the dataset**
   - Run the following script to extract images and questions:
     ```bash
     python prepare_shapes_dataset.py
     ```

4. **Set your OpenAI API key**
   - Export your API key as an environment variable:
     ```bash
     export OPENAI_API_KEY=your-key-here
     ```
   - Or create a `.env` file and use `python-dotenv` (optional).

5. **Run training**
   - Start the training script:
     ```bash
     python swirl_training.py
     ```

## Notes
- All data and outputs are stored in the `/workspace/SHAPES_dataset` directory.
- Make sure you have enough disk space (at least 10GB recommended).
- For GPU training, ensure CUDA is available and PyTorch is installed with CUDA support. 