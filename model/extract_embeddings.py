"""
extract_embeddings.py

GPU extraction of frozen bone and tissue embeddings for all HDF5 subjects.

Outputs (one pair per model):
  {EMBEDDINGS_DIR}/{model}_bone.pkl    — DataFrame(n_subjects, embed_dim)
  {EMBEDDINGS_DIR}/{model}_tissue.pkl  — DataFrame(n_subjects, embed_dim)
Both are indexed by MultiIndex(RegistrationCode, research_stage).

Usage:
  python extract_embeddings.py --models lejepa dino
  python -m model.extract_embeddings --models lejepa --lejepa-checkpoint checkpoint.pth

  # UKBB mode (reads bone/tissue views and indexes by eid/visit_index):
  python -m model.extract_embeddings --models lejepa dino --ukbb
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
from config import EMBEDDINGS_DIR, UKBB_DXA_H5

# ── UKBB DEFAULTS ─────────────────────────────────────────────────────────────
UKBB_HDF5_PATH = str(UKBB_DXA_H5)
UKBB_OUTPUT_DIR = str(EMBEDDINGS_DIR / "ukbb")


# ── DATASET (no labels needed) ────────────────────────────────────────────────
class AllHDF5Dataset(Dataset):
    """Returns (bone_tensor, tissue_tensor, id_tuple) for every key in the file.

    10K mode (default): reads 'bone' and 'tissue' sub-datasets; identity from HDF5 attrs.
    UKBB mode (ukbb=True): reads the same views; identity is parsed from the key string.
    """

    def __init__(self, hdf5_path, keys, transform, ukbb=False):
        self.hdf5_path = hdf5_path
        self.keys = keys
        self.transform = transform
        self.ukbb = ukbb
        self._h5 = None

    def _open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.hdf5_path, "r", libver="latest", swmr=True)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        self._open()
        key   = self.keys[idx]
        group = self._h5[key]

        # Both 10K and UKBB v3 have 'bone' and 'tissue' sub-datasets
        bone_np   = group["bone"][:]
        tissue_np = group["tissue"][:]
        bone   = torch.from_numpy(bone_np).float().unsqueeze(0).repeat(3, 1, 1) / 255.0
        tissue = torch.from_numpy(tissue_np).float().unsqueeze(0).repeat(3, 1, 1) / 255.0
        if self.transform is not None:
            bone   = self.transform(bone)
            tissue = self.transform(tissue)

        if self.ukbb:
            # UKBB v3 key format: "eid_visit" e.g. "1000082_2"
            parts = key.split("_")
            reg   = parts[0]            # eid
            stage = parts[1] if len(parts) > 1 else ""
            date  = ""
        else:
            # Read identity from HDF5 attributes
            reg   = group.attrs.get("RegistrationCode", key)
            stage = group.attrs.get("research_stage", "")
            date  = group.attrs.get("Date", "")
            if isinstance(reg,   bytes): reg   = reg.decode("utf-8")
            if isinstance(stage, bytes): stage = stage.decode("utf-8")
            if isinstance(date,  bytes): date  = date.decode("utf-8")

        return bone, tissue, reg, stage, date


def _collate(batch):
    bones   = torch.stack([b[0] for b in batch])
    tissues = torch.stack([b[1] for b in batch])
    regs    = [b[2] for b in batch]
    stages  = [b[3] for b in batch]
    dates   = [b[4] for b in batch]
    return bones, tissues, regs, stages, dates


# ── EXTRACTION ────────────────────────────────────────────────────────────────
def extract_and_save(model_name: str, lejepa_ckpt: str, all_keys: list[str],
                     hdf5_path: str = None, output_dir: str = None,
                     index_names: list[str] = None, ukbb: bool = False):
    hdf5_path   = hdf5_path   or U.HDF5_PATH
    output_dir  = output_dir  or U.EMBEDDINGS_DIR
    index_names = index_names or ["RegistrationCode", "research_stage", "Date"]

    print(f"\n{'='*60}\nExtracting: {model_name}\n{'='*60}")

    if model_name == "lejepa":
        backbone, _, _, val_tf, encode_fn = U.make_lejepa(lejepa_ckpt)
    elif model_name == "dino7b":
        backbone, _, _, val_tf, encode_fn = U.make_dino7b()
    else:
        backbone, _, _, val_tf, encode_fn = U.make_dino()

    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    batch_size = U.DINO_7B_BATCH_SIZE if model_name == "dino7b" else U.BATCH_SIZE
    dataset = AllHDF5Dataset(hdf5_path, all_keys, val_tf, ukbb=ukbb)
    loader  = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=False, collate_fn=_collate,
    )

    bone_feats, tissue_feats, ids = [], [], []
    with torch.no_grad():
        for bone, tissue, regs, stages, dates in tqdm(loader, desc=f"{model_name} forward pass"):
            bone, tissue = bone.to(U.DEVICE), tissue.to(U.DEVICE)
            with U.autocast():
                feat_b, feat_t = encode_fn(bone, tissue)

            # Guard: take CLS token if encoder returns full token sequence
            if feat_b.dim() == 3: feat_b = feat_b[:, 0]
            if feat_t.dim() == 3: feat_t = feat_t[:, 0]

            bone_feats.append(feat_b.cpu().float().numpy())
            tissue_feats.append(feat_t.cpu().float().numpy())
            # Build tuples matching the requested index levels
            use_date = len(index_names) == 3
            for r, s, d in zip(regs, stages, dates):
                ids.append((r, s, d) if use_date else (r, s))

    index     = pd.MultiIndex.from_tuples(ids, names=index_names)
    bone_mat  = np.vstack(bone_feats)
    tissue_mat = np.vstack(tissue_feats)
    dim = bone_mat.shape[1]

    fusion_mat = np.concatenate([bone_mat, tissue_mat], axis=1)

    os.makedirs(output_dir, exist_ok=True)
    bone_path   = os.path.join(output_dir, f"{model_name}_bone.pkl")
    tissue_path = os.path.join(output_dir, f"{model_name}_tissue.pkl")
    fusion_path = os.path.join(output_dir, f"{model_name}_fusion.pkl")

    pd.DataFrame(bone_mat,   index=index, columns=[f"b{i}" for i in range(dim)]).to_pickle(bone_path)
    pd.DataFrame(tissue_mat, index=index, columns=[f"t{i}" for i in range(dim)]).to_pickle(tissue_path)
    pd.DataFrame(fusion_mat, index=index, columns=[f"b{i}" for i in range(dim)] + [f"t{i}" for i in range(dim)]).to_pickle(fusion_path)
    print(f"  Saved {len(ids)} × {dim}-dim embeddings (+ {dim*2}-dim fusion):")
    print(f"    {bone_path}")
    print(f"    {tissue_path}")
    print(f"    {fusion_path}")


# ── ENTRY POINT ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Extract bone/tissue embeddings for LP")
    parser.add_argument("--models", nargs="+", default=["lejepa", "dino"],
                        choices=["lejepa", "dino", "dino7b"])
    parser.add_argument("--lejepa-checkpoint", default=U.LEJEPA_CHECKPOINT)
    parser.add_argument("--hdf5-path", default=None,
                        help="Override HDF5 dataset path")
    parser.add_argument("--output-dir", default=None,
                        help="Override output directory")
    parser.add_argument("--index-names", nargs="+", default=None,
                        help="Index level names, 2 or 3 values. 2 values drops Date. "
                             "E.g. --index-names eid visit_index")
    parser.add_argument("--ukbb", action="store_true",
                        help="UKBB mode: reads bone/tissue views, indexes by eid/visit_index, "
                             "saves to UKBB embeddings dir")
    args = parser.parse_args()

    # UKBB mode sets sensible defaults; explicit flags still override
    if args.ukbb:
        hdf5_path  = args.hdf5_path  or UKBB_HDF5_PATH
        output_dir = args.output_dir or UKBB_OUTPUT_DIR
        index_names = args.index_names or ["eid", "visit_index"]
    else:
        hdf5_path  = args.hdf5_path  or U.HDF5_PATH
        output_dir = args.output_dir or U.EMBEDDINGS_DIR
        index_names = args.index_names or ["RegistrationCode", "research_stage", "Date"]

    print(f"Scanning HDF5: {hdf5_path}")
    with h5py.File(hdf5_path, "r") as f:
        all_keys = list(f.keys())
    print(f"Total keys: {len(all_keys)} | Device: {U.DEVICE} | UKBB mode: {args.ukbb}")

    for model_name in args.models:
        extract_and_save(model_name, args.lejepa_checkpoint, all_keys,
                         hdf5_path=hdf5_path, output_dir=output_dir,
                         index_names=index_names, ukbb=args.ukbb)

    print("\nDone.")


if __name__ == "__main__":
    main()
