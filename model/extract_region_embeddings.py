"""
extract_region_embeddings.py

GPU extraction of MEAN-POOLED regional-scan embeddings, per subject.

DXA HDF5 groups store, besides the whole-body `bone`/`tissue` dual-energy views,
a `crops` subgroup holding the dedicated regional acquisitions. In HPP these are
exactly the femur-left, femur-right and lumbar-spine scans (94% of subjects have
3 crops). We push every crop through the SAME frozen encoder used for the
whole-body embeddings, then mean-pool the per-crop vectors into one regional
descriptor per subject. Mean-pooling is order-invariant, so the fact that crops
are stored unlabelled (`crops/0,1,2`) does not matter.

Output (one per model): {EMBEDDINGS_DIR}/{model}_regionpool.pkl
  DataFrame(n_subjects, embed_dim), MultiIndex(RegistrationCode, research_stage).
Subjects with 0 crops get an all-NaN row (imputed downstream, per-split).

Run on GPU (SLURM segal.q / conda gilenv); see sbatch_region_embeddings.sh.

Usage:
  python extract_region_embeddings.py --models lejepa dino
  python extract_region_embeddings.py --models dino --ukbb   # (Phase 2; crops incl. knee)
"""

import argparse
import os

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import common.utils as U

# ── UKBB DEFAULTS (Phase 2) ───────────────────────────────────────────────────
UKBB_HDF5_PATH  = "/data/ukbb_data/ukbb_dexa_dataset_v3.h5"
UKBB_OUTPUT_DIR = "/data/hpp_labdata/Analyses/gilsa/embeddings/ukbb_comparison"


# ── DATASET ───────────────────────────────────────────────────────────────────
class CropDataset(Dataset):
    """One item per (subject-key, crop-index) across the whole file.

    Returns (img_tensor, flat_index) where flat_index points back into
    `pairs` so the caller can group per subject after the forward pass.
    """

    def __init__(self, hdf5_path, pairs, transform):
        self.hdf5_path = hdf5_path
        self.pairs = pairs            # list of (key, crop_idx)
        self.transform = transform
        self._h5 = None

    def _open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.hdf5_path, "r", libver="latest", swmr=True)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        self._open()
        key, ci = self.pairs[idx]
        arr = self._h5[key]["crops"][str(ci)][:]
        img = torch.from_numpy(arr).float().unsqueeze(0).repeat(3, 1, 1) / 255.0
        if self.transform is not None:
            img = self.transform(img)
        return img, idx


def _collate(batch):
    imgs = torch.stack([b[0] for b in batch])
    idxs = [b[1] for b in batch]
    return imgs, idxs


# ── METADATA PASS ─────────────────────────────────────────────────────────────
def scan_metadata(hdf5_path, ukbb=False):
    """One header pass: returns (ordered_keys, key->(reg,stage), pairs).

    `pairs` is the flat list of (key, crop_idx) for every crop in the file.
    Keys with no `crops` group contribute no pairs (→ NaN row downstream).
    """
    keys, key_meta, pairs = [], {}, []
    with h5py.File(hdf5_path, "r", libver="latest", swmr=True) as f:
        for key in f.keys():
            g = f[key]
            if ukbb:
                parts = key.split("_")
                reg, stage = parts[0], (parts[1] if len(parts) > 1 else "")
            else:
                reg = g.attrs.get("RegistrationCode", key)
                stage = g.attrs.get("research_stage", "")
                if isinstance(reg, bytes):   reg = reg.decode("utf-8")
                if isinstance(stage, bytes): stage = stage.decode("utf-8")
            keys.append(key)
            key_meta[key] = (reg, stage)
            if "crops" in g:
                n = len(g["crops"])
                pairs.extend((key, i) for i in range(n))
    return keys, key_meta, pairs


# ── ENCODER ───────────────────────────────────────────────────────────────────
def _make_model(model_name, lejepa_ckpt):
    if model_name == "lejepa":
        backbone, embed_dim, _, val_tf, _ = U.make_lejepa(lejepa_ckpt)
    elif model_name == "dino7b":
        backbone, embed_dim, _, val_tf, _ = U.make_dino7b()
    else:
        backbone, embed_dim, _, val_tf, _ = U.make_dino()
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False
    return backbone, embed_dim, val_tf


def _encode_one(model_name, backbone, imgs):
    """Single-view forward → (B, embed_dim) features."""
    out = backbone(imgs)
    feat = out[0] if isinstance(out, (tuple, list)) else out   # LeJEPA returns (feat, _)
    if feat.dim() == 3:        # token sequence → CLS
        feat = feat[:, 0]
    return feat


# Inference batch sizes (frozen encoder, no grad → far larger than the train default).
_INFER_BATCH = {"lejepa": 256, "dino": 64, "dino7b": U.DINO_7B_BATCH_SIZE}


# ── EXTRACTION ────────────────────────────────────────────────────────────────
def extract_and_save(model_name, lejepa_ckpt, hdf5_path, output_dir,
                     index_names, ukbb=False, batch_size=None):
    print(f"\n{'='*60}\nRegional-pool extraction: {model_name}\n{'='*60}")
    keys, key_meta, pairs = scan_metadata(hdf5_path, ukbb=ukbb)
    n_with = len(set(k for k, _ in pairs))
    print(f"  {len(keys)} subjects | {len(pairs)} crops | "
          f"{n_with} subjects with ≥1 crop | {len(keys) - n_with} with 0 crops")

    backbone, embed_dim, val_tf = _make_model(model_name, lejepa_ckpt)
    batch_size = batch_size or _INFER_BATCH.get(model_name, U.BATCH_SIZE)
    print(f"  inference batch size: {batch_size}")

    dataset = CropDataset(hdf5_path, pairs, val_tf)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=False, collate_fn=_collate)

    # accumulate sum + count per subject key
    sums   = {k: np.zeros(embed_dim, dtype=np.float64) for k in keys}
    counts = {k: 0 for k in keys}
    with torch.no_grad():
        for imgs, idxs in tqdm(loader, desc=f"{model_name} crops"):
            imgs = imgs.to(U.DEVICE)
            with U.autocast():
                feats = _encode_one(model_name, backbone, imgs)
            feats = feats.cpu().float().numpy()
            for j, flat_idx in enumerate(idxs):
                k = pairs[flat_idx][0]
                sums[k] += feats[j]
                counts[k] += 1

    # build matrix in `keys` order; 0-crop subjects → NaN
    mat = np.full((len(keys), embed_dim), np.nan, dtype=np.float32)
    for i, k in enumerate(keys):
        if counts[k] > 0:
            mat[i] = (sums[k] / counts[k]).astype(np.float32)

    index = pd.MultiIndex.from_tuples([key_meta[k] for k in keys], names=index_names)
    df = pd.DataFrame(mat, index=index, columns=[f"r{i}" for i in range(embed_dim)])
    df = df[~df.index.duplicated(keep="first")]

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{model_name}_regionpool.pkl")
    df.to_pickle(out_path)
    n_nan = int(df.isna().all(axis=1).sum())
    print(f"  Saved {df.shape[0]} × {embed_dim}  ({n_nan} all-NaN / 0-crop) → {out_path}")


# ── ENTRY POINT ───────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Mean-pooled regional-scan embeddings")
    ap.add_argument("--models", nargs="+", default=["lejepa", "dino"],
                    choices=["lejepa", "dino", "dino7b"])
    ap.add_argument("--lejepa-checkpoint", default=U.LEJEPA_CHECKPOINT)
    ap.add_argument("--hdf5-path", default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--batch-size", type=int, default=None,
                    help="Inference batch size (default: 256 lejepa / 64 dino).")
    ap.add_argument("--ukbb", action="store_true",
                    help="UKBB mode (Phase 2): crops include knee; index by eid/visit.")
    args = ap.parse_args()

    if args.ukbb:
        hdf5_path   = args.hdf5_path  or UKBB_HDF5_PATH
        output_dir  = args.output_dir or UKBB_OUTPUT_DIR
        index_names = ["eid", "visit_index"]
    else:
        hdf5_path   = args.hdf5_path  or U.HDF5_PATH
        output_dir  = args.output_dir or U.EMBEDDINGS_DIR
        index_names = ["RegistrationCode", "research_stage"]

    print(f"HDF5: {hdf5_path}\nOut:  {output_dir}\nDevice: {U.DEVICE} | UKBB: {args.ukbb}")
    for model_name in args.models:
        extract_and_save(model_name, args.lejepa_checkpoint, hdf5_path, output_dir,
                         index_names, ukbb=args.ukbb, batch_size=args.batch_size)
    print("\nDone.")


if __name__ == "__main__":
    main()
