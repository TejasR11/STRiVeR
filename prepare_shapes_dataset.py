import os
import numpy as np
from PIL import Image

BASE = "/workspace/SHAPES_dataset"
splits = ["train.tiny", "train.small", "train.med", "train.large", "val", "test"]

for split in splits:
    # paths to raw data
    inp_npy = os.path.join(BASE, f"{split}.input.npy")
    qtxt = os.path.join(BASE, f"{split}.query_str.txt")

    # load the flat images and questions
    images_flat = np.load(inp_npy)                  # shape = (N, 2700)
    with open(qtxt) as f:
        questions = [l.strip() for l in f]

    # make output dirs
    out_img_dir = os.path.join(BASE, split, "images")
    os.makedirs(out_img_dir, exist_ok=True)
    out_q_file = os.path.join(BASE, split, "questions.txt")

    # save each image as a 30×30 PNG
    for i, flat in enumerate(images_flat):
        arr = flat.reshape(30, 30, 3).astype(np.uint8)
        Image.fromarray(arr).save(f"{out_img_dir}/{i:05d}.png")

    # dump the questions
    with open(out_q_file, "w") as f:
        f.write("\n".join(questions))

    print(f"{split}: wrote {len(images_flat)} PNGs → {out_img_dir}")
    print(f"{split}: wrote {len(questions)} questions → {out_q_file}") 