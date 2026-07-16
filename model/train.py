import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import wandb
import os
import h5py
from transformers import get_cosine_schedule_with_warmup
from model.datasets import LeJEPAHDF5Dataset
from model.model import LeJEPA_Encoder, SIGReg
from model.augmentations import train_transforms, val_transforms


config = {
    "img_size": (384, 128),
    "batch_size": 64,
    "lr": 5e-4,
    "probe_lr": 1e-3,
    "weight_decay": 5e-2,
    "epochs": 400,
    "lambda": 0.05,
    "sigreg_slices": 2048,
    "warmup_epochs": 30,
    "global_views": 2,
    "local_views": 8,
    "model_name": 'vit_small_patch16_384',
    "drop_path": 0.1,
    # [EXPERIMENTAL] SWA — per LeJEPA Table 4, small ViT boost
    "use_swa": True,
    "swa_start_epoch": 100,  # start averaging at 75% of training
    "swa_lr": 1e-4,
}


# NEW PATH CONFIGURATION
HDF5_PATH = '/data/hpp_labdata/Data/10K/aws_lab_files/dxa/dxa_dataset.h5'
TARGETS_CSV = "/data/hpp_labdata/Analyses/10K_Trajectories/body_systems/Age_Gender_BMI.csv"
CHECKPOINTS = '/data/hpp_labdata/Analyses/gilsa/checkpoints/lejepa_dexa/hpp'
EMBEDDINGS_PREFIX = '/data/hpp_labdata/Analyses/gilsa/embeddings/vits_s_4_10'


RESUME_FROM = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


run_name = f"late_fusion_2_8_withSWA"


class OnlineLinearProbe(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.probe = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.probe(x)


def lejepa_collate_fn(batch):
    data, targets = zip(*batch)
    
    if isinstance(targets[0], torch.Tensor):
        collated_targets = torch.stack(targets)
    else:
        collated_targets = targets

    if isinstance(data[0], tuple) and len(data[0]) == 2:
        bone_lists, tissue_lists = zip(*data)
        num_views = len(bone_lists[0])
        collated_bone = [torch.stack([b[i] for b in bone_lists]) for i in range(num_views)]
        collated_tissue = [torch.stack([t[i] for t in tissue_lists]) for i in range(num_views)]
        return (collated_bone, collated_tissue), collated_targets
    else:
        views_lists = data
        num_views = len(views_lists[0])
        collated_views = [torch.stack([v[i] for v in views_lists]) for i in range(num_views)]
        return collated_views, collated_targets
    views_lists = data
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
            is_dual = isinstance(views_data, tuple) and len(views_data) == 2
            if len(views_data) < 2:
                raise ValueError("Extraction requires at least 2 global views (bone and tissue).")

            if is_dual:
                bone_views, tissue_views = views_data
            else:
                # Current dataset may return a single view list where global view 0 is bone and 1 is tissue.
                if not isinstance(views_data, list) or len(views_data) < 2:
                    raise ValueError(
                        "Late-fusion extraction requires either dual-modality tuple views or at least 2 global views in list mode."
                    )
                bone_views = [views_data[0]]
                tissue_views = [views_data[1]]
            bone_view = views_data[0].to(DEVICE)
            tissue_view = views_data[1].to(DEVICE)

            # Process Bone
            b_stacked = torch.stack(bone_views).to(DEVICE)
            n_v, bs, c, h, w = b_stacked.shape
            f_b, _ = encoder(b_stacked.view(-1, c, h, w))
            f_b = f_b.view(n_v, bs, -1).mean(dim=0)
            f_b, _ = encoder(bone_view)
            embs_bone.extend(f_b.cpu().numpy())
            
            # Process Tissue
            t_stacked = torch.stack(tissue_views).to(DEVICE)
            n_t, bs_t, c_t, h_t, w_t = t_stacked.shape
            f_t, _ = encoder(t_stacked.view(-1, c_t, h_t, w_t))
            f_t = f_t.view(n_t, bs_t, -1).mean(dim=0)
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
    wandb.init(entity="your-wandb-entity", project="LeJEPA_DEXA_Scratch", name=run_name, resume="allow")

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

    # Val dataset uses targets_orig which has the values
    val_dataset = LeJEPAHDF5Dataset(
        hdf5_path=HDF5_PATH,
        keys=val_keys,
        targets_df=targets_orig,
        transform=val_aug,
        n_global=2,
        n_local=0
    )

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4, collate_fn=lejepa_collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4, collate_fn=lejepa_collate_fn, pin_memory=True)

    # ---------------------------------------------------------
    # 7. MODEL & LOOP

    encoder = LeJEPA_Encoder(config["model_name"], img_size=config['img_size'], proj_out_dim=64).to(DEVICE)

    # [EXPERIMENTAL] SWA — uniform weight averaging in the final phase of training
    if config["use_swa"]:
        from torch.optim.swa_utils import AveragedModel, update_bn
        swa_encoder = AveragedModel(encoder)
        print(f"[EXPERIMENTAL] SWA enabled (starts epoch {config['swa_start_epoch']}, lr={config['swa_lr']})")
    else:
        swa_encoder = None

    # Probe heads are detached from encoder for SSL monitoring and late fusion.
    probe_bone = nn.Linear(encoder.embed_dim, 1).to(DEVICE)
    probe_tissue = nn.Linear(encoder.embed_dim, 1).to(DEVICE)

    sigreg_module = SIGReg(num_slices=config["sigreg_slices"]).to(DEVICE)

    # Optimizer for encoder (SSL loss)
    opt_enc = optim.AdamW(encoder.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    # Separate optimizer for probes (detached for SSL monitoring)
    opt_probe = optim.AdamW(
        list(probe_bone.parameters()) + list(probe_tissue.parameters()),
        lr=config["probe_lr"],
        weight_decay=1e-5,
    )

    total_steps = config["epochs"] * len(train_loader)
    warmup_steps = config["warmup_epochs"] * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(opt_enc, warmup_steps, total_steps)

    mse_crit = nn.MSELoss()

    start_epoch = 1
    best_val_r2 = -float('inf')
    global_step = 0

    os.makedirs(CHECKPOINTS, exist_ok=True)
    os.makedirs(os.path.dirname(EMBEDDINGS_PREFIX), exist_ok=True)

    if RESUME_FROM and os.path.exists(RESUME_FROM):
        checkpoint = torch.load(RESUME_FROM, map_location=DEVICE)
        encoder.load_state_dict(checkpoint['encoder'])
        if swa_encoder is not None and 'swa_encoder' in checkpoint:
            swa_encoder.load_state_dict(checkpoint['swa_encoder'])
        if 'probe_bone' in checkpoint:
            probe_bone.load_state_dict(checkpoint['probe_bone'])
            probe_tissue.load_state_dict(checkpoint['probe_tissue'])
        elif 'probe' in checkpoint:
            # Backward compatibility for older checkpoints with a single probe.
            probe_bone.load_state_dict(checkpoint['probe'])
            probe_tissue.load_state_dict(checkpoint['probe'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_r2 = checkpoint.get('best_val_r2', -float('inf'))
        global_step = checkpoint.get('global_step', 0)  # Restore step counter
        print(f"Resumed from Epoch {start_epoch}, global_step={global_step}")

    print("Starting Training...")

    for epoch in range(start_epoch, config["epochs"] + 1):
        encoder.train()
        probe_bone.train()
        probe_tissue.train()

        train_stats = {'loss': 0, 'pred': 0, 'sig': 0, 'probe': 0}
        train_preds, train_trues = [], []

        for views_data, age_targets in train_loader:
            age_targets = age_targets.to(DEVICE).view(-1)
            bs = age_targets.size(0)

            n_globals = config["global_views"]
            n_locals = config["local_views"]

            is_dual = isinstance(views_data, tuple) and len(views_data) == 2
            if is_dual:
                bone_views, tissue_views = views_data
            else:
                bone_views, tissue_views = views_data, None
                
            def compute_ssl(views):
                global_views = torch.cat(views[:n_globals], dim=0).to(DEVICE)
                g_feats, g_projs = encoder(global_views)

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
                return l_ssl, l_pred, l_sig, g_feats.view(n_globals, bs, -1)

            opt_enc.zero_grad()
            
            if is_dual:
                l_ssl_b, l_pred_b, l_sig_b, g_feats_b = compute_ssl(bone_views)
                l_ssl_t, l_pred_t, l_sig_t, g_feats_t = compute_ssl(tissue_views)
                loss_ssl = (l_ssl_b + l_ssl_t) / 2.0
                loss_pred = (l_pred_b + l_pred_t) / 2.0
                loss_sigreg = (l_sig_b + l_sig_t) / 2.0
            else:
                loss_ssl, loss_pred, loss_sigreg, g_feats_b = compute_ssl(bone_views)
                g_feats_t = None
            loss_ssl, loss_pred, loss_sigreg, g_feats_all = compute_ssl(views_data)
                
            loss_ssl.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            opt_enc.step()

            # 2. Probe update with DETACHED features for SSL monitoring
            label_mask = (age_targets != -1.0) & (~torch.isnan(age_targets))
            loss_probe = torch.tensor(0.0, device=DEVICE)

            if label_mask.sum() > 0:
                target_labeled = age_targets[label_mask]
                
                feat_pooled_b = g_feats_b.mean(dim=0).detach()
                age_pred_b = probe_bone(feat_pooled_b[label_mask]).view(-1)
                feat_b = g_feats_all[0].detach()  # Global view 1 (Bone)
                feat_t = g_feats_all[1].detach()  # Global view 2 (Tissue)
                
                if is_dual:
                    feat_pooled_t = g_feats_t.mean(dim=0).detach()
                    age_pred_t = probe_tissue(feat_pooled_t[label_mask]).view(-1)
                    age_pred = (age_pred_b + age_pred_t) / 2.0
                else:
                    age_pred = age_pred_b
                    
                loss_probe = mse_crit(age_pred, target_labeled)
                age_pred_b = probe_bone(feat_b[label_mask]).view(-1)
                age_pred_t = probe_tissue(feat_t[label_mask]).view(-1)
                
                # Train probes simultaneously on independent modalities
                loss_probe_b = mse_crit(age_pred_b, target_labeled)
                loss_probe_t = mse_crit(age_pred_t, target_labeled)
                loss_probe = (loss_probe_b + loss_probe_t) / 2.0

                opt_probe.zero_grad()
                loss_probe.backward()
                opt_probe.step()

                age_pred = (age_pred_b.detach() + age_pred_t.detach()) / 2.0
                train_preds.extend(age_pred.detach().cpu().numpy())
                train_trues.extend(target_labeled.cpu().numpy())

            scheduler.step()

            train_stats['loss'] += loss_ssl.item()
            train_stats['pred'] += loss_pred.item()
            train_stats['sig'] += loss_sigreg.item()
            train_stats['probe'] += loss_probe.item()

            global_step += 1

        train_r2 = r2_score(train_trues, train_preds) if len(train_trues) > 1 else 0.0

        # --- VALIDATION LOOP ---
        encoder.eval()
        probe_bone.eval()
        probe_tissue.eval()
        val_preds, val_trues = [], []

        with torch.no_grad():
            for views_data, age_targets in val_loader:
                is_dual = isinstance(views_data, tuple) and len(views_data) == 2
                age_targets = age_targets.to(DEVICE).view(-1)
                
                if is_dual:
                    bone_views, tissue_views = views_data
                    
                    b_stacked = torch.stack(bone_views).to(DEVICE)
                    n_v, bs, c, h, w = b_stacked.shape
                    f_b, _ = encoder(b_stacked.view(-1, c, h, w))
                    pred_b = probe_bone(f_b.view(n_v, bs, -1).mean(dim=0)).view(-1)
                    
                    t_stacked = torch.stack(tissue_views).to(DEVICE)
                    f_t, _ = encoder(t_stacked.view(-1, c, h, w))
                    pred_t = probe_tissue(f_t.view(n_v, bs, -1).mean(dim=0)).view(-1)
                    
                    pred = (pred_b + pred_t) / 2.0
                else:
                    b_stacked = torch.stack(views_data).to(DEVICE)
                    n_v, bs, c, h, w = b_stacked.shape
                    f_b, _ = encoder(b_stacked.view(-1, c, h, w))
                    pred = probe_bone(f_b.view(n_v, bs, -1).mean(dim=0)).view(-1)
                bone_view = views_data[0].to(DEVICE)
                tissue_view = views_data[1].to(DEVICE)
                
                f_b, _ = encoder(bone_view)
                pred_b = probe_bone(f_b).view(-1)
                
                f_t, _ = encoder(tissue_view)
                pred_t = probe_tissue(f_t).view(-1)
                
                pred = (pred_b + pred_t) / 2.0

                val_preds.extend(pred.cpu().numpy())
                val_trues.extend(age_targets.cpu().numpy())

        val_r2 = r2_score(val_trues, val_preds) if len(val_trues) > 0 else 0.0

        wandb.log({
            "epoch": epoch,
            "train_r2": train_r2,
            "val_r2": val_r2,
            "train_loss": train_stats['loss'] / len(train_loader),
            "train_pred_loss": train_stats['pred'] / len(train_loader),
            "train_sig_loss": train_stats['sig'] / len(train_loader),
            "lr": scheduler.get_last_lr()[0]
        }, step=global_step)

        print(f"Ep {epoch} | Train R2: {train_r2:.4f} | Val R2: {val_r2:.4f}")

        # [EXPERIMENTAL] SWA: update averaged weights after swa_start_epoch
        if swa_encoder is not None and epoch >= config["swa_start_epoch"]:
            swa_encoder.update_parameters(encoder)
            # Switch to flat SWA learning rate
            for pg in opt_enc.param_groups:
                pg['lr'] = config["swa_lr"]

        if val_r2 > best_val_r2 or epoch % 50 == 0:
            best_val_r2 = val_r2
            ckpt = {
                'epoch': epoch,
                'encoder': encoder.state_dict(),
                'probe_bone': probe_bone.state_dict(),
                'probe_tissue': probe_tissue.state_dict(),
                'best_val_r2': best_val_r2,
                'global_step': global_step,
            }
            if swa_encoder is not None:
                ckpt['swa_encoder'] = swa_encoder.state_dict()
            torch.save(ckpt, os.path.join(CHECKPOINTS, f'best_model_{run_name}.pth'))

    # [EXPERIMENTAL] SWA: update BatchNorm stats on averaged model
    if swa_encoder is not None:
        print("[EXPERIMENTAL] Updating SWA BatchNorm statistics...")
        # Reset BN running stats
        for module in swa_encoder.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                module.reset_running_stats()
        swa_encoder.train()
        bn_dataset = LeJEPAHDF5Dataset(
            hdf5_path=HDF5_PATH, keys=train_keys, targets_df=targets_train,
            transform=val_aug, n_global=2, n_local=0,
        )
        bn_loader = DataLoader(bn_dataset, batch_size=config["batch_size"],
                               shuffle=False, num_workers=4,
                               collate_fn=lejepa_collate_fn)
        with torch.no_grad():
            for views_data, _ in bn_loader:
                x = views_data[0].to(DEVICE)
                swa_encoder(x)
        swa_encoder.eval()

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
    # Use SWA-averaged encoder for extraction if available, otherwise original
    extraction_encoder = swa_encoder if swa_encoder is not None else encoder
    extract_and_save_embeddings(
        encoder=extraction_encoder,
        dataset=full_dataset,
        batch_size=config["batch_size"],
        filename_prefix=EMBEDDINGS_PREFIX,
    )


if __name__ == "__main__":
    main()
