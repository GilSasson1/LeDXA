import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import os
import h5py
from transformers import get_cosine_schedule_with_warmup
from model.datasets import LeJEPAHDF5Dataset
from model.model import LeJEPA_Encoder, SIGReg
from model.augmentations import train_transforms, val_transforms
from config import (
    CHECKPOINTS_DIR,
    EMBEDDINGS_DIR,
    HPP_DXA_H5,
    HPP_TARGETS_CSV,
)


config = {
    "img_size": (384, 128),
    "batch_size": 256,
    "lr": 5e-4,
    "weight_decay": 5e-2,
    "epochs": 400,
    "lambda": 0.05,
    "sigreg_slices": 2048,
    "warmup_epochs": 30,
    "global_views": 2,
    "local_views": 8,
    "model_name": 'vit_small_patch16_384',
    "drop_path": 0.1,
}


HDF5_PATH = str(HPP_DXA_H5)
TARGETS_CSV = str(HPP_TARGETS_CSV)
CHECKPOINTS = str(CHECKPOINTS_DIR / "hpp")
EMBEDDINGS_PREFIX = str(EMBEDDINGS_DIR / "ledxa_hpp")


RESUME_FROM = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


run_name = "ledxa_late_fusion"


def lejepa_collate_fn(batch):
    views_lists, targets = zip(*batch)
    collated_targets = torch.stack(targets)
    num_views = len(views_lists[0])
    collated_views = [torch.stack([v[i] for v in views_lists]) for i in range(num_views)]
    return collated_views, collated_targets


def extract_and_save_embeddings(encoder, dataset, batch_size, filename_prefix):
    encoder.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=lejepa_collate_fn)
    embs_bone, embs_tissue = [], []
    ids = []
    
    print("Extracting embeddings for entire dataset...")
    with torch.no_grad():
        for i, (views_data, _) in enumerate(loader):
            if len(views_data) < 2:
                raise ValueError("Extraction requires at least 2 global views (bone and tissue).")

            bone_view = views_data[0].to(DEVICE)
            tissue_view = views_data[1].to(DEVICE)

            f_b, _ = encoder(bone_view)
            embs_bone.extend(f_b.cpu().numpy())
            
            f_t, _ = encoder(tissue_view)
            embs_tissue.extend(f_t.cpu().numpy())

            bs = bone_view.size(0)
            start_idx = i * batch_size
            batch_keys = dataset.keys[start_idx : start_idx + bs]
            for k in batch_keys:
                parts = k.split('_')
                ids.append(("_".join(parts[:2]), "_".join(parts[2:])))

    index = pd.MultiIndex.from_tuples(ids, names=['RegistrationCode', 'research_stage'])
    
    df_bone = pd.DataFrame(np.vstack(embs_bone), index=index, columns=[f"lej_bone_{i}" for i in range(embs_bone[0].shape[0])])
    df_bone.to_pickle(f"{filename_prefix}_bone.pkl")
    print(f"Saved {filename_prefix}_bone.pkl")

    if len(embs_tissue) == 0:
        raise ValueError("No tissue embeddings were produced. Dual-modality output is required.")
    df_tissue = pd.DataFrame(np.vstack(embs_tissue), index=index, columns=[f"lej_tissue_{i}" for i in range(embs_tissue[0].shape[0])])
    df_tissue.to_pickle(f"{filename_prefix}_tissue.pkl")
    print(f"Saved {filename_prefix}_tissue.pkl")


def main():
    print(f"Loading targets from {TARGETS_CSV}...")
    targets_orig = pd.read_csv(TARGETS_CSV)

    # STRICT: Set MultiIndex as requested
    targets_orig.set_index(['RegistrationCode', 'research_stage'], inplace=True)
    targets_orig.sort_index(inplace=True)

    # Normalize Targets
    age_mean = targets_orig['age'].mean()
    age_std = targets_orig['age'].std()
    targets_orig['age'] = (targets_orig['age'] - age_mean) / age_std
    print(f"Target Normalization: Mean={age_mean:.2f}, Std={age_std:.2f}")

    # SCAN HDF5 & GROUP BY SUBJECT (REGISTRATION CODE)
    print(f"Scanning HDF5 keys from {HDF5_PATH}...")

    with h5py.File(HDF5_PATH, 'r') as f:
        all_raw_keys = list(f.keys())

    # Map: { "10K_12345": ["10K_12345_baseline", "10K_12345_visit1"] }
    subject_map = {}

    for key in all_raw_keys:
        parts = key.split('_')
        # Reconstruct Subject ID (RegistrationCode): "10K_12345"
        s_id = "_".join(parts[:2])

        if s_id not in subject_map:
            subject_map[s_id] = []
        subject_map[s_id].append(key)

    unique_subjects = list(subject_map.keys())
    print(f"Total HDF5 images: {len(all_raw_keys)}")
    print(f"Total Unique Patients: {len(unique_subjects)}")

    # IDENTIFY LABELED VS UNLABELED PATIENTS

    # Check which patients exist in your targets CSV (Level 0 check)
    valid_target_ids = set(targets_orig.index.get_level_values(0))

    labeled_subjs = []
    unlabeled_subjs = []

    for s in unique_subjects:
        if s in valid_target_ids:
            labeled_subjs.append(s)
        else:
            unlabeled_subjs.append(s)

    print(f" - Patients with Labels: {len(labeled_subjs)}")
    print(f" - Patients w/o Labels: {len(unlabeled_subjs)} (Adding to Train)")

    # Split patients at subject level to avoid leakage across visits.
    train_subs, val_subs = train_test_split(labeled_subjs, test_size=0.2, random_state=73)

    # Combine: Train = All Unlabeled + 80% Labeled
    final_train_subs = unlabeled_subjs + train_subs
    final_val_subs = val_subs

    overlap_subjects = set(final_train_subs).intersection(set(final_val_subs))
    if overlap_subjects:
        raise ValueError(f"Subject leakage detected in split. Overlap count={len(overlap_subjects)}")

    # --- Prepare Train Targets ---
    targets_train = targets_orig.copy()

    # Set labels to NaN for all validation subjects in the TRAIN dataframe
    val_mask = targets_train.index.get_level_values(0).isin(final_val_subs)
    targets_train.loc[val_mask, 'age'] = np.nan

    # --- TRAIN KEYS ---
    # Simply grab all images for the training patients
    train_keys = []
    for s in final_train_subs:
        train_keys.extend(subject_map[s])

    # --- VAL KEYS ---
    val_keys = []

    print("Building Validation Set (Strict Match: Recode + Visit)...")
    for s in final_val_subs:
        potential_keys = subject_map[s]
        for k in potential_keys:
            # Re-parse to check specific visit label
            parts = k.split('_')
            s_id = "_".join(parts[:2])

            # STRICT LOGIC from your snippet: join everything after index 2
            v_id = "_".join(parts[2:])

            # Check if this specific visit (e.g. baseline) has a value
            if (s_id, v_id) in targets_orig.index:
                val = targets_orig.loc[(s_id, v_id), 'age']
                if isinstance(val, pd.Series):
                    val = val.iloc[0]

                # Only add if not NaN
                if not pd.isna(val):
                    val_keys.append(k)

    print(f"Final Split Statistics:")
    print(f" - Train Subjects: {len(final_train_subs)}")
    print(f" - Val Subjects: {len(final_val_subs)}")
    print(f" - Train Images: {len(train_keys)}")
    print(f" - Val Images: {len(val_keys)}")

    train_key_subs = {"_".join(k.split('_')[:2]) for k in train_keys}
    val_key_subs = {"_".join(k.split('_')[:2]) for k in val_keys}
    key_overlap = train_key_subs.intersection(val_key_subs)
    if key_overlap:
        raise ValueError(f"Key-level leakage detected after split. Overlap subjects={len(key_overlap)}")

    # 3. Instantiate the Transforms
    train_aug = train_transforms(global_size=(384, 128), local_size=(96, 96))
    val_aug = val_transforms(global_size=(384, 128))

    # 4. Create Datasets
    train_dataset = LeJEPAHDF5Dataset(
        hdf5_path=HDF5_PATH,
        keys=train_keys,
        targets_df=targets_train,
        transform=train_aug,
        n_global=config["global_views"],
        n_local=config["local_views"]
    )

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4, collate_fn=lejepa_collate_fn, pin_memory=True)

    # ---------------------------------------------------------
    # 7. MODEL & LOOP

    encoder = LeJEPA_Encoder(config["model_name"], img_size=config['img_size'], proj_out_dim=64).to(DEVICE)

    sigreg_module = SIGReg(num_slices=config["sigreg_slices"]).to(DEVICE)

    opt_enc = optim.AdamW(encoder.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    total_steps = config["epochs"] * len(train_loader)
    warmup_steps = config["warmup_epochs"] * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(opt_enc, warmup_steps, total_steps)

    start_epoch = 1
    global_step = 0

    os.makedirs(CHECKPOINTS, exist_ok=True)
    os.makedirs(os.path.dirname(EMBEDDINGS_PREFIX), exist_ok=True)

    if RESUME_FROM and os.path.exists(RESUME_FROM):
        checkpoint = torch.load(RESUME_FROM, map_location=DEVICE)
        encoder.load_state_dict(checkpoint['encoder'])
        start_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint.get('global_step', 0)
        print(f"Resumed from Epoch {start_epoch}, global_step={global_step}")

    print("Starting Training...")

    for epoch in range(start_epoch, config["epochs"] + 1):
        encoder.train()
        train_stats = {'loss': 0, 'pred': 0, 'sig': 0}

        for views_data, _ in train_loader:
            bs = views_data[0].size(0)
            n_globals = config["global_views"]
            n_locals = config["local_views"]

            def compute_ssl(views):
                global_views = torch.cat(views[:n_globals], dim=0).to(DEVICE)
                _, g_projs = encoder(global_views)

                local_inputs = torch.cat(views[n_globals:], dim=0).to(DEVICE)
                _, l_projs = encoder(local_inputs)

                g_projs_v = g_projs.view(n_globals, bs, -1)
                l_projs_v = l_projs.view(n_locals, bs, -1)
                all_projs = torch.cat([g_projs_v, l_projs_v], dim=0)

                center = g_projs_v.mean(dim=0)
                l_pred = (all_projs - center.unsqueeze(0)).square().mean()

                l_sig = 0.0
                for i in range(all_projs.shape[0]):
                    l_sig += sigreg_module(all_projs[i])
                l_sig = l_sig / all_projs.shape[0]

                l_ssl = (1 - config["lambda"]) * l_pred + config["lambda"] * l_sig
                return l_ssl, l_pred, l_sig

            opt_enc.zero_grad()
            loss_ssl, loss_pred, loss_sigreg = compute_ssl(views_data)
            loss_ssl.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            opt_enc.step()
            scheduler.step()

            train_stats['loss'] += loss_ssl.item()
            train_stats['pred'] += loss_pred.item()
            train_stats['sig'] += loss_sigreg.item()
            global_step += 1

        print(f"Ep {epoch} | step {global_step} | "
              f"loss {train_stats['loss']/len(train_loader):.4f} | "
              f"pred {train_stats['pred']/len(train_loader):.4f} | "
              f"sig {train_stats['sig']/len(train_loader):.4f} | "
              f"lr {scheduler.get_last_lr()[0]:.2e}")

        if epoch % 50 == 0 or epoch == config["epochs"]:
            torch.save(
                {'epoch': epoch, 'encoder': encoder.state_dict(), 'global_step': global_step},
                os.path.join(CHECKPOINTS, f'best_model_{run_name}.pth'),
            )

    labeled_all_keys = []
    for k in all_raw_keys:
        parts = k.split('_')
        idx = ("_".join(parts[:2]), "_".join(parts[2:]))
        if idx in targets_orig.index:
            v = targets_orig.loc[idx, 'age']
            if isinstance(v, pd.Series):
                v = v.iloc[0]
            if not pd.isna(v):
                labeled_all_keys.append(k)

    print(f"Extracting embeddings for {len(labeled_all_keys)} labeled key(s)...")
    full_dataset = LeJEPAHDF5Dataset(
        hdf5_path=HDF5_PATH,
        keys=labeled_all_keys,
        targets_df=targets_orig,
        transform=val_aug,
        n_global=2,
        n_local=0,
    )
    extract_and_save_embeddings(
        encoder=encoder,
        dataset=full_dataset,
        batch_size=config["batch_size"],
        filename_prefix=EMBEDDINGS_PREFIX,
    )


if __name__ == "__main__":
    main()
