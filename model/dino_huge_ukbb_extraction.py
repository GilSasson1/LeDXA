"""
Extract UKBB DXA embeddings using the DINOv3 ViT-7B (huge+) model.
Weights are loaded directly to CUDA to avoid the CPU RAM bottleneck of a 27GB checkpoint.

Output: MultiIndex DataFrame (eid, visit_index) saved as .pkl, same format as
        dino_ukbb_extraction.py so it can be dropped straight into the Cox pipeline.
"""
import os
import sys

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

# Make sure the parent package (dexa_fm) is importable when run as a script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model.dinov3_baseline import DINOv3

# --- CONFIGURATION ---
MODEL_NAME  = 'vit_huge_plus_patch16_dinov3'
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE  = 128         # ~600M params — reduce if OOM
NUM_WORKERS = 4
HEIGHT, WIDTH = 256, 256  # Matches model's native input size

HDF5_PATH   = '/data/ukbb_data/ukbb_dexa_dataset_v3.h5'
OUTPUT_FILE = '/data/hpp_labdata/Analyses/gilsa/embeddings/ukbb_comparison/dino_huge_plus_fusion.pkl'
# Change to dino_huge_plus_qkvb_fusion.pkl if using the qkvb variant

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class UKBB_HDF5_Dataset(Dataset):
    def __init__(self, h5_path, transform=None):
        self.h5_path = h5_path
        self.transform = transform
        self.hf = None  # lazy-opened per worker

        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

        with h5py.File(h5_path, 'r') as hf:
            self.keys = list(hf.keys())

        print(f"Found {len(self.keys)} subjects in HDF5.")

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        if self.hf is None:
            self.hf = h5py.File(self.h5_path, 'r')

        key = self.keys[idx]
        try:
            data = self.hf[key]['fullbody'][:]               # uint8 (3, H, W)
            img_tensor = torch.from_numpy(data).float() / 255.0
            if self.transform:
                img_tensor = self.transform(img_tensor)
            parts = key.split('_')
            patient_id = parts[0]
            visit_id   = "_".join(parts[1:])
            return img_tensor, patient_id, visit_id
        except Exception as e:
            print(f"Error loading {key}: {e}")
            return torch.zeros((3, HEIGHT, WIDTH)), "error", "error"


def extract_features():
    transform = transforms.Compose([
        transforms.Resize((HEIGHT, WIDTH),
                          interpolation=transforms.InterpolationMode.BICUBIC,
                          antialias=True),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    dataset    = UKBB_HDF5_Dataset(HDF5_PATH, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"))

    print(f"Loading {MODEL_NAME}...")
    model = DINOv3(model_name=MODEL_NAME, freeze_backbone=True, load_to_gpu=False)
    model = model.to(DEVICE)
    model.eval()

    all_embeddings: list[np.ndarray] = []
    all_indices: list[tuple] = []

    print("Starting extraction...")
    with torch.no_grad():
        for images, pids, vids in tqdm(dataloader):
            images = images.to(DEVICE, non_blocking=True)
            embeddings = model(images)          # (B, D)
            batch_emb  = embeddings.cpu().numpy()

            for i, (pid, vid) in enumerate(zip(pids, vids)):
                if pid != "error":
                    all_embeddings.append(batch_emb[i])
                    all_indices.append((pid, vid))

    print(f"Extraction complete. {len(all_embeddings)} embeddings.")

    if not all_embeddings:
        print("No embeddings extracted. Check dataset path and errors above.")
        return

    emb_matrix = np.vstack(all_embeddings)
    print(f"Embedding matrix shape: {emb_matrix.shape}")

    df = pd.DataFrame(emb_matrix,
                      columns=[f"dino_huge_plus_{i}" for i in range(emb_matrix.shape[1])])
    df.index = pd.MultiIndex.from_tuples(all_indices, names=['eid', 'visit_index'])

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    print(f"Saving to {OUTPUT_FILE}...")
    df.to_pickle(OUTPUT_FILE)
    print("Done!")


if __name__ == "__main__":
    extract_features()
