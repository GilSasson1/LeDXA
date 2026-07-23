"""Shared datasets, model factories, metrics, and fitting utilities."""

import os
import re
from contextlib import nullcontext

from statsmodels.stats.multitest import multipletests

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr, ttest_rel, wilcoxon as _wilcoxon_test
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from model.dinov3_baseline import DINOv3
from model.model import LeJEPA_Encoder
from config import (
    CHECKPOINTS_DIR,
    EMBEDDINGS_DIR as CONFIG_EMBEDDINGS_DIR,
    HPP_DOWNSTREAM_TARGETS_CSV,
    HPP_DXA_H5,
    LEJEPA_CHECKPOINT as CONFIG_LEJEPA_CHECKPOINT,
)

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
HDF5_PATH = str(HPP_DXA_H5)
TARGETS_CSV = str(HPP_DOWNSTREAM_TARGETS_CSV)
LEJEPA_CHECKPOINT = str(CONFIG_LEJEPA_CHECKPOINT)

CHECKPOINT_DIR = str(CHECKPOINTS_DIR / "comparison")
EMBEDDINGS_DIR = str(CONFIG_EMBEDDINGS_DIR / "comparison")

BATCH_SIZE = 32
DINO_7B_BATCH_SIZE = 8  # 7B model needs ~14GB weights + activations; fits on L40S (48GB)
NUM_EPOCHS_FT = 50
WARMUP_EPOCHS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AUX_LOSS_WEIGHT = 0.5

TARGET_COLUMNS = "all"
MIN_SAMPLES_PER_TARGET = 200
MIN_POSITIVE_CASES     = 100   # minimum positive-class count for classification targets
CLASSIFICATION_TARGETS = {"gender"}
_SEED_POOL        = [42, 73, 99, 123, 2024, 7, 17, 31, 137, 256,
                     13, 21, 55, 89, 144, 233, 377, 610, 987, 1597]
DEFAULT_NUM_SEEDS = 5

def make_seeds(n: int) -> list[int]:
    """Return the first n seeds from the fixed pool (max 20)."""
    if n > len(_SEED_POOL):
        raise ValueError(f"--num-seeds max is {len(_SEED_POOL)}, got {n}")
    return _SEED_POOL[:n]

# Ridge LP: 5 log-spaced alphas, 2-fold CV within the training set
RIDGE_ALPHAS = np.logspace(-1, 3, 5)  # [0.1, 1, 10, 100, 1000]
RIDGE_CV_FOLDS = 2

# LeJEPA native config
LEJEPA_MODEL_NAME = "vit_small_patch16_384"
LEJEPA_IMG_SIZE = (384, 128)
LEJEPA_PROJ_OUT_DIM = 64

# DINOv3 native config
DINO_MODEL_NAME = "vit_huge_plus_patch16_dinov3.lvd1689m"
DINO_7B_MODEL_NAME = "vit_7b_patch16_dinov3.lvd1689m"
DINO_IMG_SIZE = (256, 256)
_DINO_MEAN = (0.485, 0.456, 0.406)
_DINO_STD = (0.229, 0.224, 0.225)

_LEJEPA_MEAN = [0.1960878074169159] * 3
_LEJEPA_STD = [0.2843901515007019] * 3


# ── UTILITIES ─────────────────────────────────────────────────────────────────
def sanitize(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]+", "_", str(name)).strip("_")


def autocast():
    if DEVICE == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def safe_pearson(y_true, y_pred) -> float:
    try:
        r, _ = pearsonr(y_true, y_pred)
        return 0.0 if np.isnan(r) else float(r)
    except Exception:
        return 0.0


def auc_from_logits(y_true, logits) -> float:
    probs = 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float32)))
    if np.unique(y_true).size < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, probs))
    except Exception:
        return float("nan")


def metric(y_true, y_pred, is_cls: bool) -> float:
    if is_cls:
        return auc_from_logits(y_true, y_pred)
    return safe_pearson(np.asarray(y_true), np.asarray(y_pred))


def checkpoint_path(model_name, mode, fusion, target_col, seed):
    return os.path.join(
        CHECKPOINT_DIR,
        f"{model_name}_{mode}_{fusion}_{sanitize(target_col)}_s{seed}.pth",
    )


def embeddings_prefix(model_name, mode, fusion, target_col, seed):
    return os.path.join(
        EMBEDDINGS_DIR,
        f"{model_name}_{mode}_{fusion}_{sanitize(target_col)}_s{seed}",
    )


# ── DATASET ───────────────────────────────────────────────────────────────────
class ComparisonHDF5Dataset(Dataset):
    """Loads bone + tissue from HDF5, applies transform, returns label."""

    def __init__(self, hdf5_path, keys, targets_df, target_col, transform):
        self.hdf5_path = hdf5_path
        self.keys = keys
        self.targets_df = targets_df
        self.target_col = target_col
        self.transform = transform
        self._h5 = None

    def _open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.hdf5_path, "r", libver="latest", swmr=True)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        self._open()
        key = self.keys[idx]

        bone_np = self._h5[key]["bone"][:]
        tissue_np = self._h5[key]["tissue"][:]
        bone = torch.from_numpy(bone_np).float().unsqueeze(0).repeat(3, 1, 1) / 255.0
        tissue = torch.from_numpy(tissue_np).float().unsqueeze(0).repeat(3, 1, 1) / 255.0

        if self.transform is not None:
            bone = self.transform(bone)
            tissue = self.transform(tissue)

        parts = key.split("_")
        s_id = "_".join(parts[:2])
        v_id = "_".join(parts[2:])
        val = self.targets_df.loc[(s_id, v_id), self.target_col]
        if isinstance(val, pd.Series):
            val = val.iloc[0]
        return (bone, tissue), torch.tensor(float(val), dtype=torch.float32)


# ── BACKBONE FACTORIES ────────────────────────────────────────────────────────
def make_lejepa(ckpt_path: str):
    """Fresh LeJEPA encoder from checkpoint. Returns (backbone, embed_dim, train_tf, val_tf, encode_fn)."""
    encoder = LeJEPA_Encoder(
        LEJEPA_MODEL_NAME,
        img_size=LEJEPA_IMG_SIZE,
        proj_out_dim=LEJEPA_PROJ_OUT_DIM,
        pretrained=False,
    ).to(DEVICE)

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    if isinstance(ckpt, dict) and "encoder" in ckpt:
        state = ckpt["encoder"]
        source = "encoder"
    else:
        state = ckpt
        source = "checkpoint"
    missing, unexpected = encoder.load_state_dict(state, strict=False)
    print(
        f"[LeJEPA] Loaded {source}. "
        f"missing={len(missing)}, unexpected={len(unexpected)}"
    )

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(
            LEJEPA_IMG_SIZE, scale=(0.5, 1.0), ratio=(0.25, 0.45),
            interpolation=transforms.InterpolationMode.BICUBIC, antialias=True,
        ),
        transforms.RandomRotation(degrees=10),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.GaussianBlur(5, 0.1)], p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4)], p=0.8),
        transforms.Normalize(mean=_LEJEPA_MEAN, std=_LEJEPA_STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(LEJEPA_IMG_SIZE, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.Normalize(mean=_LEJEPA_MEAN, std=_LEJEPA_STD),
    ])

    def encode(bone, tissue):
        feat_b, _ = encoder(bone)
        feat_t, _ = encoder(tissue)
        return feat_b, feat_t

    return encoder, encoder.embed_dim, train_tf, val_tf, encode


def make_dino():
    """Fresh DINOv3 from pretrained weights. Returns (backbone, embed_dim, train_tf, val_tf, encode_fn)."""
    backbone = DINOv3(model_name=DINO_MODEL_NAME, freeze_backbone=False).to(DEVICE)
    embed_dim = backbone.backbone.num_features

    train_tf = transforms.Compose([
        
        transforms.RandomResizedCrop(
            DINO_IMG_SIZE, scale=(0.5, 1.0), ratio=(0.75, 1.33),
            interpolation=transforms.InterpolationMode.BICUBIC, antialias=True,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4)], p=0.8),
        transforms.RandomApply([transforms.GaussianBlur(5, 0.1)], p=0.5),
        transforms.Normalize(mean=_DINO_MEAN, std=_DINO_STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(DINO_IMG_SIZE, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=_DINO_MEAN, std=_DINO_STD),
    ])

    def encode(bone, tissue):
        return backbone(bone), backbone(tissue)

    return backbone, embed_dim, train_tf, val_tf, encode


def make_dino7b():
    """DINOv3 7B ViT. Same protocol as make_dino() but with the largest model."""
    # load_to_gpu=True: loads weights directly to CUDA, bypassing CPU RAM
    backbone = DINOv3(model_name=DINO_7B_MODEL_NAME, freeze_backbone=False, load_to_gpu=True).to(DEVICE)
    embed_dim = backbone.backbone.num_features

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(
            224, scale=(0.5, 1.0), ratio=(0.75, 1.33),
            interpolation=transforms.InterpolationMode.BICUBIC, antialias=True,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4)], p=0.8),
        transforms.RandomApply([transforms.GaussianBlur(5, 0.1)], p=0.5),
        transforms.Normalize(mean=_DINO_MEAN, std=_DINO_STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(DINO_IMG_SIZE, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=_DINO_MEAN, std=_DINO_STD),
    ])

    def encode(bone, tissue):
        return backbone(bone), backbone(tissue)

    return backbone, embed_dim, train_tf, val_tf, encode


# ── HEADS (gradient-based LP / FT) ────────────────────────────────────────────
def build_heads(embed_dim: int, fusion: str):
    if fusion == "late":
        head_b = nn.Linear(embed_dim, 1).to(DEVICE)
        head_t = nn.Linear(embed_dim, 1).to(DEVICE)
        return {"bone": head_b, "tissue": head_t}, list(head_b.parameters()) + list(head_t.parameters())
    head = nn.Linear(embed_dim * 2, 1).to(DEVICE)
    return {"concat": head}, list(head.parameters())


def forward_heads(heads, feat_b, feat_t, fusion):
    if fusion == "late":
        pb = heads["bone"](feat_b).squeeze(-1)
        pt = heads["tissue"](feat_t).squeeze(-1)
        return pb, pt, 0.5 * (pb + pt)
    pf = heads["concat"](torch.cat([feat_b, feat_t], dim=1)).squeeze(-1)
    return pf, pf, pf


def compute_loss(criterion, pred_b, pred_t, pred_f, label, fusion):
    loss_f = criterion(pred_f, label)
    if fusion == "late":
        return loss_f + AUX_LOSS_WEIGHT * 0.5 * (criterion(pred_b, label) + criterion(pred_t, label))
    return loss_f


# ── VALIDATION ────────────────────────────────────────────────────────────────
def validate(backbone, encode_fn, heads, loader, criterion, fusion, is_cls):
    backbone.eval()
    for h in heads.values():
        h.eval()

    losses, preds_b, preds_t, preds_f, truths = [], [], [], [], []
    with torch.no_grad():
        for (bone, tissue), label in loader:
            bone, tissue = bone.to(DEVICE), tissue.to(DEVICE)
            label = label.to(DEVICE).float()
            with autocast():
                feat_b, feat_t = encode_fn(bone, tissue)
                pred_b, pred_t, pred_f = forward_heads(heads, feat_b, feat_t, fusion)
                loss = compute_loss(criterion, pred_b, pred_t, pred_f, label, fusion)
            losses.append(loss.item())
            preds_b.extend(pred_b.detach().cpu().float().numpy())
            preds_t.extend(pred_t.detach().cpu().float().numpy())
            preds_f.extend(pred_f.detach().cpu().float().numpy())
            truths.extend(label.cpu().numpy())

    out = {"fusion": metric(truths, preds_f, is_cls)}
    if fusion == "late":
        out["bone"] = metric(truths, preds_b, is_cls)
        out["tissue"] = metric(truths, preds_t, is_cls)
    return float(np.mean(losses)), out


# ── EMBEDDING EXTRACTION ──────────────────────────────────────────────────────
def extract_embeddings(encode_fn, backbone, dataset, prefix, batch_size=None):
    """Extract and save bone/tissue embeddings from best model checkpoint."""
    batch_size = batch_size or BATCH_SIZE
    backbone.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    bone_embs, tissue_embs, ids = [], [], []

    with torch.no_grad():
        for i, ((bone, tissue), _) in enumerate(tqdm(loader, desc="Extracting embeddings")):
            bone, tissue = bone.to(DEVICE), tissue.to(DEVICE)
            with autocast():
                feat_b, feat_t = encode_fn(bone, tissue)
            bone_embs.append(feat_b.cpu().float().numpy())
            tissue_embs.append(feat_t.cpu().float().numpy())

            start_idx = i * BATCH_SIZE
            batch_keys = dataset.keys[start_idx: start_idx + len(bone)]
            for k in batch_keys:
                parts = k.split("_")
                ids.append(("_".join(parts[:2]), "_".join(parts[2:])))

    index = pd.MultiIndex.from_tuples(ids, names=["RegistrationCode", "research_stage"])
    bone_mat = np.vstack(bone_embs)
    tissue_mat = np.vstack(tissue_embs)
    fusion_mat = np.concatenate([bone_mat, tissue_mat], axis=1)

    os.makedirs(os.path.dirname(prefix), exist_ok=True)
    pd.DataFrame(bone_mat, index=index,
                 columns=[f"bone_{i}" for i in range(bone_mat.shape[1])]).to_pickle(f"{prefix}_bone.pkl")
    pd.DataFrame(tissue_mat, index=index,
                 columns=[f"tissue_{i}" for i in range(tissue_mat.shape[1])]).to_pickle(f"{prefix}_tissue.pkl")
    pd.DataFrame(fusion_mat, index=index,
                 columns=[f"fusion_{i}" for i in range(fusion_mat.shape[1])]).to_pickle(f"{prefix}_fusion.pkl")
    print(f"  Saved embeddings: {prefix}_[bone|tissue|fusion].pkl")


# ── SINGLE SEED RUN (FT only) ─────────────────────────────────────────────────
def run_one_config(
    model_name: str,
    lejepa_ckpt: str,
    train_keys, val_keys, all_labeled_keys,
    target_df, target_col: str,
    fusion: str, is_cls: bool,
    seed: int,
) -> dict[str, float]:
    """Fine-tune one (model × fusion × target × seed). Returns per-view val metrics."""
    run_name = f"{model_name}_ft_{fusion}_{sanitize(target_col)}_s{seed}"
    print(f"\n{'='*60}\n{run_name}\n{'='*60}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    if model_name == "lejepa":
        backbone, embed_dim, train_tf, val_tf, encode_fn = make_lejepa(lejepa_ckpt)
    elif model_name == "dino7b":
        backbone, embed_dim, train_tf, val_tf, encode_fn = make_dino7b()
    else:
        backbone, embed_dim, train_tf, val_tf, encode_fn = make_dino()

    batch_size = DINO_7B_BATCH_SIZE if model_name == "dino7b" else BATCH_SIZE
    ds_train = ComparisonHDF5Dataset(HDF5_PATH, train_keys, target_df, target_col, train_tf)
    ds_val = ComparisonHDF5Dataset(HDF5_PATH, val_keys, target_df, target_col, val_tf)

    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    loader_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    heads, head_params = build_heads(embed_dim, fusion)
    num_epochs = NUM_EPOCHS_FT

    for p in backbone.parameters():
        p.requires_grad = True
    optimizer = optim.AdamW(
            [{"params": backbone.parameters(), "lr": 1e-5},
             {"params": head_params, "lr": 1e-4}],
            weight_decay=0.01,
        )
    total_steps = num_epochs * len(loader_train)
    warmup_steps = WARMUP_EPOCHS * len(loader_train)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[1e-5, 1e-4],
        total_steps=total_steps,
        pct_start=warmup_steps / total_steps,
        anneal_strategy="cos",
    )

    criterion = nn.BCEWithLogitsLoss() if is_cls else nn.MSELoss()
    metric_name = "auc" if is_cls else "pearson"
    best_val_metric = -float("inf")
    ckpt_path = ckpt_path = checkpoint_path(model_name, "ft", fusion, target_col, seed)
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    for epoch in range(num_epochs):
        backbone.train()
        for h in heads.values():
            h.train()

        train_losses = []
        for (bone, tissue), label in loader_train:
            bone, tissue = bone.to(DEVICE), tissue.to(DEVICE)
            label = label.to(DEVICE).float()
            optimizer.zero_grad()

            with autocast():
                feat_b, feat_t = encode_fn(bone, tissue)

            with autocast():
                pred_b, pred_t, pred_f = forward_heads(heads, feat_b, feat_t, fusion)
                loss = compute_loss(criterion, pred_b, pred_t, pred_f, label, fusion)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(backbone.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            train_losses.append(loss.item())

        val_loss, val_metrics = validate(backbone, encode_fn, heads, loader_val, criterion, fusion, is_cls)
        val_m = val_metrics["fusion"]
        train_loss = float(np.mean(train_losses))
        lr_now = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]["lr"]

        view_msg = f"| val_{metric_name}_fusion={val_metrics['fusion']:.4f}"
        if "bone" in val_metrics:
            view_msg += f" | bone={val_metrics['bone']:.4f} | tissue={val_metrics['tissue']:.4f}"
        print(f"  Ep {epoch + 1:03d} | loss={train_loss:.4f} | val_loss={val_loss:.4f} {view_msg} | lr={lr_now:.2e}")

        if val_m > best_val_metric:
            best_val_metric = val_m
            torch.save({
                "epoch": epoch + 1,
                "backbone": backbone.state_dict(),
                "heads": {k: h.state_dict() for k, h in heads.items()},
                f"best_val_{metric_name}": best_val_metric,
                "config": {"model": model_name, "mode": "ft", "fusion": fusion,
                           "target": target_col, "seed": seed},
            }, ckpt_path)

    print(f"  => Best val {metric_name}: {best_val_metric:.4f} (saved: {ckpt_path})")

    best_ckpt = torch.load(ckpt_path, map_location=DEVICE)
    backbone.load_state_dict(best_ckpt["backbone"])
    for h_name, h in heads.items():
        h.load_state_dict(best_ckpt["heads"][h_name])
    backbone.eval()

    ds_full = ComparisonHDF5Dataset(HDF5_PATH, all_labeled_keys, target_df, target_col, val_tf)
    extract_embeddings(encode_fn, backbone, ds_full,
                       embeddings_prefix(model_name, "ft", fusion, target_col, seed),
                       batch_size=batch_size)

    _, best_val_metrics = validate(backbone, encode_fn, heads, loader_val, criterion, fusion, is_cls)
    return best_val_metrics


# ── PER-TARGET ORCHESTRATION ──────────────────────────────────────────────────
def run_one_target(
    all_keys, target_df_full, target_col: str,
    lejepa_ckpt: str,
    models, fusions, seeds,
):
    is_cls = target_col.lower() in CLASSIFICATION_TARGETS
    target_df = target_df_full[[target_col]].dropna().copy()

    if len(target_df) < MIN_SAMPLES_PER_TARGET:
        print(f"[Skip] {target_col}: only {len(target_df)} labeled samples")
        return [], []

    if is_cls:
        uniq = np.sort(target_df[target_col].unique())
        if len(uniq) != 2:
            print(f"[Skip] {target_col}: not binary ({len(uniq)} classes)")
            return [], []
        target_df[target_col] = target_df[target_col].map({uniq[0]: 0.0, uniq[1]: 1.0}).astype(float)
        n_positive = int((target_df[target_col] == 1.0).sum())
        if n_positive < MIN_POSITIVE_CASES:
            print(f"[Skip] {target_col}: only {n_positive} positive cases (< {MIN_POSITIVE_CASES})")
            return [], []
    else:
        std = target_df[target_col].std()
        if std == 0 or pd.isna(std):
            print(f"[Skip] {target_col}: zero variance")
            return [], []
        target_df[target_col] = (target_df[target_col] - target_df[target_col].mean()) / std
    target_df.sort_index(inplace=True)

    subject_to_keys: dict[str, list[str]] = {}
    subject_to_idx: dict[str, list[tuple]] = {}
    for k in all_keys:
        parts = k.split("_")
        s_id = "_".join(parts[:2])
        idx = (s_id, "_".join(parts[2:]))
        if idx in target_df.index:
            subject_to_keys.setdefault(s_id, []).append(k)
            subject_to_idx.setdefault(s_id, []).append(idx)

    valid_subjects = sorted(subject_to_keys)
    all_labeled_keys = [k for s in valid_subjects for k in subject_to_keys[s]]

    if len(valid_subjects) < 2:
        print(f"[Skip] {target_col}: insufficient subjects ({len(valid_subjects)})")
        return [], []

    if is_cls:
        all_labeled_idx = [idx for s in valid_subjects for idx in subject_to_idx[s]]
        n_pos_intersected = int((target_df.loc[all_labeled_idx, target_col] == 1.0).sum())
        if n_pos_intersected < MIN_POSITIVE_CASES:
            print(f"[Skip] {target_col}: only {n_pos_intersected} positive cases in intersected "
                  f"cohort (< {MIN_POSITIVE_CASES})")
            return [], []

    metric_name = "auc" if is_cls else "pearson"
    results: dict[str, dict[str, list[float]]] = {}
    raw_rows = []

    for seed in seeds:
        train_subjects, val_subjects = train_test_split(
            valid_subjects, test_size=0.2, random_state=seed,
        )
        train_keys = [k for s in sorted(train_subjects) for k in subject_to_keys[s]]
        val_keys = [k for s in sorted(val_subjects) for k in subject_to_keys[s]]

        # Leakage guard
        if {"_".join(k.split("_")[:2]) for k in train_keys} & {"_".join(k.split("_")[:2]) for k in val_keys}:
            raise RuntimeError(f"Subject leakage in {target_col} seed={seed}")

        print(f"\n>>> {target_col} | seed={seed} | train={len(train_subjects)} "
              f"| val={len(val_subjects)} | train_keys={len(train_keys)} | val_keys={len(val_keys)}")

        for model_name in models:
            for fusion in fusions:
                key = f"{model_name}_ft_{fusion}"
                try:
                    metrics_by_view = run_one_config(
                        model_name, lejepa_ckpt,
                        train_keys, val_keys, all_labeled_keys,
                        target_df, target_col,
                        fusion, is_cls, seed,
                    )
                    cfg_results = results.setdefault(key, {})
                    for view_name, view_metric in metrics_by_view.items():
                        cfg_results.setdefault(view_name, []).append(float(view_metric))
                        raw_rows.append({
                            "target": target_col, "metric": metric_name,
                            "model": model_name, "mode": "ft", "fusion": fusion,
                            "view": view_name, "seed": seed, "score": float(view_metric),
                        })
                except Exception as e:
                    import traceback
                    print(f"[Error] {key} / {target_col} / s{seed}: {e}")
                    traceback.print_exc()

    # Summary table
    print(f"\n{'═' * 60}")
    print(f"  {target_col} — {len(seeds)}-seed summary (val {metric_name})")
    print(f"{'═' * 60}")
    summary_rows = []
    for key, views_dict in sorted(results.items()):
        model_name, mode, fusion = key.split("_", 2)
        for view_name, scores in sorted(views_dict.items()):
            arr = np.array(scores, dtype=np.float32)
            std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
            se = float(std / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
            print(f"  {key:<30} | {view_name:<6} {arr.mean():.4f} ± {arr.std():.4f}  (n={len(arr)})")
            summary_rows.append({
                "target": target_col, "metric": metric_name,
                "model": model_name, "mode": mode, "fusion": fusion, "view": view_name,
                "n": int(len(arr)), "mean": float(arr.mean()), "std": std, "se": se,
            })
    print()
    return summary_rows, raw_rows


# ── WILCOXON + BH-FDR (disease classification) ────────────────────────────────
def wilcoxon_bh_disease(raw_df: pd.DataFrame,
                        comparisons: list[tuple[str, str]]) -> pd.DataFrame:
    """Paired two-tailed Wilcoxon + BH-FDR for disease AUC comparisons.

    comparisons: list of (model_a, model_b). p-value tests H1: model_a != model_b
    (two-sided); pair with the sign of mean_a - mean_b to call direction.
    Only rows where metric == 'auc' are used.
    Returns one row per (target, comparison) with mean_a, mean_b, p_raw, p_adj.
    """
    cls_df = raw_df[raw_df["metric"] == "auc"].copy()
    rows = []
    for target, tdf in cls_df.groupby("target"):
        piv = tdf.pivot_table(index="seed", columns="model", values="score", aggfunc="mean")
        for model_a, model_b in comparisons:
            if model_a not in piv.columns or model_b not in piv.columns:
                continue
            paired = piv[[model_a, model_b]].dropna()
            if len(paired) < 2:
                p_raw = float("nan")
            else:
                diff = paired[model_a].values - paired[model_b].values
                if np.all(diff == 0):
                    p_raw = 1.0
                else:
                    try:
                        _, p_raw = _wilcoxon_test(diff, alternative="two-sided")
                    except Exception:
                        p_raw = float("nan")
            rows.append({
                "target": target,
                "comparison": f"{model_a}_vs_{model_b}",
                "model_a": model_a, "model_b": model_b,
                "mean_a": float(piv[model_a].mean()) if model_a in piv.columns else float("nan"),
                "mean_b": float(piv[model_b].mean()) if model_b in piv.columns else float("nan"),
                "n_pairs": len(paired),
                "p_raw": float(p_raw),
            })
    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows)
    adj_parts = []
    for _comp, cdf in result.groupby("comparison"):
        pvals = cdf["p_raw"].values.astype(float)
        valid = np.isfinite(pvals)
        q = np.full(len(pvals), float("nan"))
        if valid.sum() > 0:
            _, q_vals, _, _ = multipletests(pvals[valid], method="fdr_bh")
            q[valid] = q_vals
        adj_parts.append(pd.Series(q, index=cdf.index))
    result["p_adj"] = pd.concat(adj_parts).sort_index()
    return result


# ── SHARED RESULTS SAVING ─────────────────────────────────────────────────────
def _ttest_rows(raw_df: pd.DataFrame, ref_model: str) -> list[dict]:
    """Paired t-test: lejepa vs ref_model, one row per (target, metric, mode, fusion, view)."""
    rows = []
    for gvals, gdf in raw_df.groupby(["target", "metric", "mode", "fusion", "view"]):
        piv = gdf.pivot_table(index="seed", columns="model", values="score", aggfunc="mean")
        if not {ref_model, "lejepa"}.issubset(set(piv.columns)):
            continue
        paired = piv[[ref_model, "lejepa"]].dropna()
        if len(paired) < 2:
            continue
        try:
            _, p_value = ttest_rel(paired[ref_model].values, paired["lejepa"].values)
        except Exception:
            p_value = 1.0
        target, metric_name, mode, fusion, view = gvals
        rows.append({
            "target": target, "metric": metric_name, "mode": mode, "fusion": fusion, "view": view,
            "ref_model": ref_model,
            "n": int(len(paired)),
            "ref_mean": float(np.mean(paired[ref_model].values)),
            "lejepa_mean": float(np.mean(paired["lejepa"].values)),
            "p_value": float(p_value),
        })
    return rows


def save_results(all_summary_rows, all_raw_rows, results_csv, results_raw_csv, results_ttest_csv,
                 use_wilcoxon: bool = False):
    if all_summary_rows:
        os.makedirs(os.path.dirname(results_csv), exist_ok=True)
        pd.DataFrame(all_summary_rows).to_csv(results_csv, index=False)
        print(f"Saved summary: {results_csv}")
    else:
        print("No summary rows to save.")

    if results_raw_csv and all_raw_rows:
        os.makedirs(os.path.dirname(results_raw_csv), exist_ok=True)
        pd.DataFrame(all_raw_rows).to_csv(results_raw_csv, index=False)
        print(f"Saved raw scores: {results_raw_csv}")

    if results_ttest_csv and all_raw_rows:
        raw_df = pd.DataFrame(all_raw_rows)
        if use_wilcoxon:
            comparisons = [("lejepa", "dino"), ("lejepa", "tabular"), ("lejepa", "covariates"),
                           ("lejepa", "ensemble")]
            stats_df = wilcoxon_bh_disease(raw_df, comparisons)
            if not stats_df.empty:
                os.makedirs(os.path.dirname(results_ttest_csv), exist_ok=True)
                stats_df.to_csv(results_ttest_csv, index=False)
                print(f"Saved Wilcoxon + FDR: {results_ttest_csv}")
        else:
            all_ttest_rows = []
            for ref in ("dino", "tabular", "covariates"):
                all_ttest_rows.extend(_ttest_rows(raw_df, ref))
            if all_ttest_rows:
                ttest_df = pd.DataFrame(all_ttest_rows)
                fdr_col = []
                for (ref, met), grp in ttest_df.groupby(["ref_model", "metric"]):
                    pvals = grp["p_value"].values
                    _, qvals, _, _ = multipletests(pvals, method="fdr_bh")
                    fdr_col.append(pd.Series(qvals, index=grp.index))
                ttest_df["q_value_fdr_bh"] = pd.concat(fdr_col).sort_index()
                ttest_df["significant_fdr05"] = ttest_df["q_value_fdr_bh"] < 0.05
                os.makedirs(os.path.dirname(results_ttest_csv), exist_ok=True)
                ttest_df.to_csv(results_ttest_csv, index=False)
                print(f"Saved t-tests + FDR: {results_ttest_csv}")
