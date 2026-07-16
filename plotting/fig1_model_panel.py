"""Figure 1 — Model panel for the DEXA LeJEPA foundation model.

Schematic of the pretraining architecture:
  - Paired Bone + Tissue DEXA scans are the per-subject input
  - 2 global crops (384x128) + M local anatomical crops (96x96) per modality
  - Linear projection of flattened 2D patches -> ViT-S/16 encoder (weights shared
    between bone and tissue streams)
  - Projector MLP -> 64-D embeddings
  - Two objectives:
      * Prediction loss: MSE of every view projection to the centroid of the
        global projections
      * SIGReg loss: sliced empirical characteristic function matched to N(0,I)
"""
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HDF5_PATH = '/data/hpp_labdata/Data/10K/aws_lab_files/dxa/dxa_dataset.h5'
SAMPLE_KEY = '10K_1001201093_02_00_visit'
OUT_PDF = 'fig1_model_panel.pdf'
OUT_PNG = 'fig1_model_panel.png'

# ── Style ───────────────────────────────────────────────────────────────────
PUB_FS = 8
TITLE_FS = 9

BONE_COLOR = '#C62828'
TISSUE_COLOR = '#1565C0'
LOCAL_COLOR = '#2E7D32'

ENC_FC = '#B2DFDB'
ENC_EC = '#00695C'
PROJ_FC = '#D1C4E9'
PROJ_EC = '#4527A0'
CLS_FC = '#FFE082'
CLS_EC = '#B07900'
OBJ_FC = '#FAFAFA'
OBJ_EC = '#555555'
PATCH_LINE = '#FFFFFF'

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': PUB_FS,
    'axes.linewidth': 0.6,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})


# ── Helpers ─────────────────────────────────────────────────────────────────
def load_sample():
    with h5py.File(HDF5_PATH, 'r') as f:
        g = f[SAMPLE_KEY]
        bone = g['bone'][:]
        tissue = g['tissue'][:]
        crops = [g['crops'][k][:] for k in sorted(g['crops'].keys())]
    return bone, tissue, crops


def _crop_with_aspect(img, target_h, target_w):
    """Center-crop `img` to match target aspect then return it."""
    h, w = img.shape[:2]
    target_ratio = target_w / target_h
    src_ratio = w / h
    if src_ratio > target_ratio:
        new_w = int(h * target_ratio)
        x0 = (w - new_w) // 2
        return img[:, x0:x0 + new_w]
    new_h = int(w / target_ratio)
    y0 = (h - new_h) // 2
    return img[y0:y0 + new_h, :]


def _place_image(fig, img, rect, border_color, nh=None, nw=None, border_lw=1.0,
                 patch_lw=0.3):
    ax = fig.add_axes(rect)
    ax.imshow(img, cmap='gray', aspect='auto', interpolation='nearest')
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(border_color); s.set_linewidth(border_lw)
    if nh and nw:
        h, w = img.shape[:2]
        for i in range(1, nh):
            ax.axhline(i * h / nh, color=PATCH_LINE, lw=patch_lw, alpha=0.75)
        for j in range(1, nw):
            ax.axvline(j * w / nw, color=PATCH_LINE, lw=patch_lw, alpha=0.75)
    return ax


def _box(ax, x, y, w, h, label, fc, ec, fs=PUB_FS, lw=1.0, rounding=0.02,
         fontweight='bold', color='black'):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0.005,rounding_size={rounding}",
                       fc=fc, ec=ec, lw=lw)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, label, ha='center', va='center',
            fontsize=fs, fontweight=fontweight, color=color)


def _arrow(ax, p0, p1, color='#333333', lw=1.0, style='-|>', mut=9, ls='-',
           connectionstyle='arc3'):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, color=color,
                                 lw=lw, mutation_scale=mut, linestyle=ls,
                                 connectionstyle=connectionstyle))


# ── Main panel ──────────────────────────────────────────────────────────────
def build_figure():
    bone, tissue, crops = load_sample()

    # Simulate global / local crops from the real arrays
    global_bone = _crop_with_aspect(bone, 384, 128)
    global_tissue = _crop_with_aspect(tissue, 384, 128)
    local_views = [_crop_with_aspect(c, 96, 96) for c in crops[:3]]

    W, H = 14.0, 6.2
    fig = plt.figure(figsize=(W, H), facecolor='white')
    canvas = fig.add_axes([0, 0, 1, 1])
    canvas.set_xlim(0, W); canvas.set_ylim(0, H); canvas.axis('off')

    # ── Col 1: original subject ────────────────────────────────────────────
    col1_cx = 0.75 / W
    thumb_w = 0.55 / W
    thumb_h = 3.6 / H               # ≈ canvas y 1.4 – 5.0
    thumb_y0 = 1.4 / H

    _place_image(fig, global_bone,
                 [col1_cx - thumb_w - 0.005, thumb_y0, thumb_w, thumb_h],
                 BONE_COLOR, border_lw=1.2)
    _place_image(fig, global_tissue,
                 [col1_cx + 0.005, thumb_y0, thumb_w, thumb_h],
                 TISSUE_COLOR, border_lw=1.2)

    # "Bone | Tissue" header above the thumbnails
    header_y = 5.15
    canvas.text(0.72, header_y, 'Bone', ha='right', va='bottom',
                fontsize=PUB_FS, color=BONE_COLOR, fontweight='bold')
    canvas.text(0.75, header_y, '|', ha='center', va='bottom',
                fontsize=PUB_FS, color='#999')
    canvas.text(0.78, header_y, 'Tissue', ha='left', va='bottom',
                fontsize=PUB_FS, color=TISSUE_COLOR, fontweight='bold')

    canvas.text(0.75, 1.15, 'original\nDEXA subject',
                ha='center', va='top', fontsize=PUB_FS, fontweight='bold')

    # ── Col 2: view generation ─────────────────────────────────────────────
    col2_x0 = 1.85 / W
    gv_w, gv_h = 0.72 / W, 1.95 / H

    # Globals: bone top-left, tissue top-right. Patch grid 24x8.
    _place_image(fig, global_bone,
                 [col2_x0, 3.65 / H, gv_w, gv_h],
                 BONE_COLOR, nh=24, nw=8, border_lw=0.9, patch_lw=0.25)
    canvas.text(1.85 + 0.36, 5.75, 'global crop 1', ha='center',
                fontsize=PUB_FS - 0.5)

    _place_image(fig, global_tissue,
                 [col2_x0 + gv_w + 0.02, 3.65 / H, gv_w, gv_h],
                 TISSUE_COLOR, nh=24, nw=8, border_lw=0.9, patch_lw=0.25)
    canvas.text(1.85 + 0.36 + 0.72 + 0.28, 5.75, 'global crop 2',
                ha='center', fontsize=PUB_FS - 0.5)

    # Locals: three real anatomical crops at 96x96 (patch grid 6x6)
    lv_w, lv_h = 0.55 / W, 0.85 / H
    y_local = 1.85 / H
    for i, lv in enumerate(local_views):
        x = col2_x0 + i * (0.65 / W)
        _place_image(fig, lv, [x, y_local, lv_w, lv_h],
                     LOCAL_COLOR, nh=6, nw=6, border_lw=0.7, patch_lw=0.25)
    canvas.text(col2_x0 * W + 3 * 0.65 + 0.20, 2.25, '· · ·',
                ha='left', va='center', fontsize=PUB_FS + 2,
                color=LOCAL_COLOR, fontweight='bold')
    canvas.text(1.85 + 0.90, 1.58, 'local crop 1      ...      local crop M',
                ha='center', fontsize=PUB_FS - 0.5)

    canvas.text(2.75, 0.75,
                'view generation\n(2 globals + M local anatomical crops)',
                ha='center', va='top', fontsize=PUB_FS, fontweight='bold')

    # Arrows from original subject → view generation
    # right edge of col-1 thumbnails: canvas x ≈ 1.10
    _arrow(canvas, (1.12, 4.6), (1.80, 4.6), lw=0.9)
    _arrow(canvas, (1.12, 2.8), (1.80, 2.8), lw=0.9)

    # ── Col 3: patch embedding + CLS ───────────────────────────────────────
    pe_x = 4.55
    pe_y = 3.3
    _box(canvas, pe_x, pe_y, 1.15, 0.95,
         'linear projection\nof flattened\n2D patches',
         '#ECEFF1', '#607D8B', fs=PUB_FS - 0.5)

    cls_x = pe_x
    cls_y = 4.55
    _box(canvas, cls_x, cls_y, 1.15, 0.55,
         '[CLS] token', CLS_FC, CLS_EC, fs=PUB_FS - 0.5)

    # Arrows: from globals into patch-embedding box, then into encoder
    _arrow(canvas, (3.55, 4.65), (4.55, pe_y + 0.7), lw=0.9)    # bone global
    _arrow(canvas, (3.55, 4.65), (4.55, pe_y + 0.45), lw=0.9)    # tissue global
    _arrow(canvas, (3.55, 2.25), (4.55, pe_y + 0.2), lw=0.9,
           connectionstyle='arc3,rad=-0.15')                    # locals

    # ── Col 4: encoder + projector ─────────────────────────────────────────
    enc_x, enc_y, enc_w, enc_h = 6.2, 3.15, 1.60, 1.45
    _box(canvas, enc_x, enc_y, enc_w, enc_h,
         'Shared ViT-S/16\nEncoder',
         ENC_FC, ENC_EC, fs=TITLE_FS, lw=1.3)
    canvas.text(enc_x + enc_w / 2, enc_y - 0.18,
                'weights shared across\nBone + Tissue streams',
                ha='center', va='top', fontsize=PUB_FS - 1,
                color=ENC_EC, fontstyle='italic')

    # Arrows from patch-embedding + CLS into encoder
    _arrow(canvas, (pe_x + 1.15, pe_y + 0.45),
           (enc_x, enc_y + enc_h / 2 - 0.15), lw=1.0)
    _arrow(canvas, (cls_x + 1.15, cls_y + 0.28),
           (enc_x, enc_y + enc_h / 2 + 0.35), lw=1.0,
           connectionstyle='arc3,rad=0.1')

    # Projector
    proj_x, proj_y, proj_w, proj_h = 8.20, 3.40, 1.30, 0.95
    _box(canvas, proj_x, proj_y, proj_w, proj_h,
         'Projector\n(MLP: 2048→2048→64)',
         PROJ_FC, PROJ_EC, fs=PUB_FS - 0.5)
    _arrow(canvas, (enc_x + enc_w, enc_y + enc_h / 2),
           (proj_x, proj_y + proj_h / 2), lw=1.1)

    # 64-D embedding token after projector
    emb_x, emb_y = proj_x + proj_w + 0.12, proj_y + proj_h / 2 - 0.22
    _box(canvas, emb_x, emb_y, 0.55, 0.45, 'z ∈ ℝ⁶⁴',
         '#FFFFFF', '#333333', fs=PUB_FS - 0.5, fontweight='normal')
    _arrow(canvas, (proj_x + proj_w, proj_y + proj_h / 2),
           (emb_x, emb_y + 0.22), lw=1.0)

    # Two parallel "rails" showing bone and tissue pass through same encoder
    canvas.text(enc_x + enc_w / 2 + 0.05, enc_y + enc_h / 2 + 0.05,
                '', ha='center')
    # colored side-dots at encoder input to signal dual-stream
    canvas.scatter([enc_x - 0.06, enc_x - 0.06],
                   [enc_y + enc_h - 0.25, enc_y + 0.25],
                   s=24, c=[BONE_COLOR, TISSUE_COLOR], zorder=5, clip_on=False)

    # ── Col 5: Objectives (two dashed boxes) ───────────────────────────────
    obj_x = 10.45
    obj_w = 3.35

    # Prediction objective (top)
    pred_y, pred_h = 4.00, 1.55
    dashed = FancyBboxPatch((obj_x, pred_y), obj_w, pred_h,
                            boxstyle="round,pad=0.02,rounding_size=0.05",
                            fc=OBJ_FC, ec=OBJ_EC, lw=1.1, linestyle='--')
    canvas.add_patch(dashed)
    canvas.text(obj_x + 0.10, pred_y + pred_h - 0.15,
                'Prediction objective', ha='left', va='top',
                fontsize=PUB_FS, fontweight='bold')

    # Centroid + MSE diagram
    cx, cy = obj_x + 0.85, pred_y + 0.65
    for v_off_y, color in [(0.45, BONE_COLOR), (0.20, TISSUE_COLOR),
                           (-0.05, LOCAL_COLOR), (-0.25, LOCAL_COLOR)]:
        sx, sy = cx - 1.05, cy + v_off_y
        canvas.scatter([sx], [sy], s=26, color=color, zorder=5)
        _arrow(canvas, (sx + 0.06, sy), (cx - 0.15, cy), lw=0.8,
               color='#666666', mut=7)
    canvas.scatter([cx], [cy], s=70, facecolor='white',
                   edgecolor='black', linewidth=1.2, zorder=6)
    canvas.text(cx, cy, '⨁', ha='center', va='center', fontsize=PUB_FS)
    canvas.text(cx, cy - 0.30, 'centroid of\nglobal projections',
                ha='center', va='top', fontsize=PUB_FS - 1.5,
                fontstyle='italic', color='#555')

    canvas.text(obj_x + obj_w - 0.15, pred_y + 0.75,
                r'$\mathcal{L}_{\mathrm{pred}} = '
                r'\frac{1}{V}\sum_{v}\,\|z_v - \bar z_{\mathrm{glob}}\|^{2}$',
                ha='right', va='center', fontsize=PUB_FS + 0.5)

    # SIGReg objective (bottom)
    sig_y, sig_h = 1.85, 1.85
    dashed2 = FancyBboxPatch((obj_x, sig_y), obj_w, sig_h,
                             boxstyle="round,pad=0.02,rounding_size=0.05",
                             fc=OBJ_FC, ec=OBJ_EC, lw=1.1, linestyle='--')
    canvas.add_patch(dashed2)
    canvas.text(obj_x + 0.10, sig_y + sig_h - 0.15,
                'SIGReg objective',
                ha='left', va='top', fontsize=PUB_FS, fontweight='bold')
    canvas.text(obj_x + 0.10, sig_y + sig_h - 0.38,
                'random slices of z pushed to N(0, I)',
                ha='left', va='top', fontsize=PUB_FS - 1.5,
                fontstyle='italic', color='#555')

    t = np.linspace(-4, 4, 300)
    gaussian = np.exp(-0.5 * t ** 2)

    def _style_dist_ax(ax):
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.5); s.set_color('#aaa')
        ax.set_facecolor('none')
        ax.set_ylim(-0.05, 1.15)

    # "before": a clearly non-Gaussian distribution — bimodal + skewed
    ax_before = fig.add_axes([obj_x / W + 0.005, sig_y / H + 0.04,
                               0.95 / W, 0.95 / H])
    np.random.seed(42)
    before_dist = (0.55 * np.exp(-0.5 * ((t + 1.5) / 0.6) ** 2) +
                   0.45 * np.exp(-0.5 * ((t - 1.1) / 0.9) ** 2))
    before_dist /= before_dist.max()
    ax_before.plot(t, before_dist, color='#333', lw=1.3)
    ax_before.set_title('z projections', fontsize=PUB_FS - 1.5, pad=2,
                        color='#444', style='italic')
    _style_dist_ax(ax_before)

    # "→" arrow on the canvas between the two insets
    mid_x = obj_x / W + 0.005 + 0.95 / W + 0.5 / W   # midpoint
    canvas.annotate('', xy=(obj_x + 0.005 + 0.95 + 0.75, sig_y + 0.58),
                    xytext=(obj_x + 0.005 + 0.95 + 0.15, sig_y + 0.58),
                    xycoords='data', textcoords='data',
                    arrowprops=dict(arrowstyle='-|>', color='#333',
                                   lw=1.2, mutation_scale=12))
    canvas.text(obj_x + 0.005 + 0.95 + 0.45, sig_y + 0.68,
                r'$\mathcal{L}_{\mathrm{sig}}$',
                ha='center', va='bottom', fontsize=PUB_FS)

    # "after": standard Gaussian N(0,1)
    ax_after = fig.add_axes([(obj_x + 0.005 + 0.95 + 0.80) / W,
                              sig_y / H + 0.04, 0.95 / W, 0.95 / H])
    ax_after.plot(t, gaussian, color='#333', lw=1.3)
    ax_after.set_title('N(0, I)', fontsize=PUB_FS - 1.5, pad=2,
                       color='#444', style='italic')
    _style_dist_ax(ax_after)

    # SIGReg equation — sits below the push diagram
    canvas.text(obj_x + obj_w / 2, sig_y + 0.22,
                r'$\mathcal{L}_{\mathrm{sig}} = '
                r'\int |\varphi_{\mathrm{emp}}(t) '
                r'- e^{-t^2/2}|^{2}\, e^{-t^2/2}\,dt$',
                ha='center', va='center', fontsize=PUB_FS)

    # Arrow from embedding z to objectives
    _arrow(canvas, (emb_x + 0.55, emb_y + 0.22),
           (obj_x, pred_y + pred_h / 2), lw=1.0,
           connectionstyle='arc3,rad=0.15')
    _arrow(canvas, (emb_x + 0.55, emb_y + 0.22),
           (obj_x, sig_y + sig_h / 2), lw=1.0,
           connectionstyle='arc3,rad=-0.15')

    # Total loss footer
    canvas.text(obj_x + obj_w / 2, 1.45,
                r'$\mathcal{L} = (1-\lambda)\,\mathcal{L}_{\mathrm{pred}} '
                r'+ \lambda\,\mathcal{L}_{\mathrm{sig}}, \quad \lambda = 0.05$',
                ha='center', va='center', fontsize=PUB_FS + 1,
                fontweight='bold')

    # Panel label "a)"
    canvas.text(0.12, 5.95, 'a)', ha='left', va='top',
                fontsize=TITLE_FS + 3, fontweight='bold')

    return fig


if __name__ == '__main__':
    fig = build_figure()
    fig.savefig(OUT_PDF, bbox_inches='tight', pad_inches=0.1)
    fig.savefig(OUT_PNG, bbox_inches='tight', pad_inches=0.1, dpi=300)
    print(f'Saved {OUT_PDF} and {OUT_PNG}')
