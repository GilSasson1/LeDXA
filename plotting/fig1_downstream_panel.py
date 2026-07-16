"""Figure 1b — Downstream tasks panel (circular layout)."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT_PDF = 'fig1_downstream_panel.pdf'
OUT_PNG = 'fig1_downstream_panel.png'

PUB_FS  = 8
TITLE_FS = 9

BONE_COLOR   = '#C62828'
TISSUE_COLOR = '#1565C0'
ENC_FC = '#CFD8DC'
ENC_EC = '#546E7A'

TASK_COLORS = {
    'trait':   '#00695C',
    'aging':   '#1565C0',
    'disease': '#B71C1C',
    'drug':    '#6A1B9A',
    'gwas':    '#4527A0',
    'cluster': '#E65100',
}

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': PUB_FS,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

W, H = 13.0, 11.0   # figure size (inches) == data coord range


# ── Geometry helper ──────────────────────────────────────────────────────────
def _rect_edge(cx, cy, w, h, dx, dy):
    """Point on rectangle edge in direction (dx, dy) from its centre."""
    n = np.hypot(dx, dy)
    dx, dy = dx / n, dy / n
    if abs(dx) < 1e-9:
        return cx, cy + h / 2 * np.sign(dy)
    if abs(dy) < 1e-9:
        return cx + w / 2 * np.sign(dx), cy
    t = min((w / 2) / abs(dx), (h / 2) / abs(dy))
    return cx + t * dx, cy + t * dy


def _style_mini(ax):
    for s in ax.spines.values():
        s.set_linewidth(0.4); s.set_color('#ccc')
    ax.tick_params(length=0, labelsize=PUB_FS - 2)
    ax.set_facecolor('none')


# ── Mini-plot functions ──────────────────────────────────────────────────────
def _mini_scatter(ax):
    np.random.seed(1)
    n = 60
    actual = np.linspace(20, 80, n) + np.random.randn(n) * 2
    pred   = actual + np.random.randn(n) * 6
    ax.scatter(actual, pred, s=3, color=TASK_COLORS['trait'], alpha=0.6, linewidths=0)
    m, b = np.polyfit(actual, pred, 1)
    xl = np.array([actual.min(), actual.max()])
    ax.plot(xl, m * xl + b, color='#222', lw=0.9)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel('actual', fontsize=PUB_FS - 2, labelpad=1)
    ax.set_ylabel('predicted', fontsize=PUB_FS - 2, labelpad=1)
    ax.text(0.95, 0.05, r'$R^2{=}0.81$', ha='right', va='bottom',
            transform=ax.transAxes, fontsize=PUB_FS - 2)


def _mini_aging(ax):
    t = np.linspace(-3, 3, 300)
    for mu, col in zip([-1.2, -0.4, 0.4, 1.2],
                       ['#2471a3', '#76b7c8', '#f0a07a', '#c0392b']):
        kde = np.exp(-0.5 * ((t - mu) / 0.55) ** 2)
        ax.plot(t, kde / kde.max(), color=col, lw=1.0)
    ax.axvline(0, color='#555', lw=0.5, ls='--', alpha=0.6)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel('aging rate', fontsize=PUB_FS - 2, labelpad=1)
    # Minimal legend inside
    for x_pos, col, lab in zip([0.04, 0.27, 0.52, 0.75],
                                ['#2471a3','#76b7c8','#f0a07a','#c0392b'],
                                ['Q1','Q2','Q3','Q4']):
        ax.text(x_pos, 0.04, lab, transform=ax.transAxes, fontsize=PUB_FS-2.5,
                color=col, va='bottom', fontweight='bold')


def _mini_km(ax):
    t = np.linspace(0, 10, 200)
    ax.step(t, np.exp(-0.18 * t), color=TASK_COLORS['trait'],   lw=1.0)
    ax.step(t, np.exp(-0.45 * t), color=TASK_COLORS['disease'], lw=1.0)
    ax.fill_between(t, np.exp(-0.18 * t), np.exp(-0.45 * t),
                    alpha=0.10, color='#888', step='pre')
    ax.set_xticks([]); ax.set_yticks([0, 1])
    ax.set_yticklabels(['0', '1'], fontsize=PUB_FS - 2)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('time', fontsize=PUB_FS - 2, labelpad=1)


def _mini_drug(ax):
    np.random.seed(9)
    n = 18
    before = np.random.normal(0.8, 0.6, n)
    after  = before + np.random.normal(-0.55, 0.4, n)
    for b, a in zip(before, after):
        ax.plot([0, 1], [b, a], color='#888', alpha=0.22, lw=0.7)
    parts = ax.violinplot([before, after], positions=[0, 1],
                          showmedians=True, widths=0.45)
    for pc in parts['bodies']:
        pc.set_facecolor(TASK_COLORS['drug']); pc.set_alpha(0.35)
    for key in ('cmedians', 'cmins', 'cmaxes', 'cbars'):
        parts[key].set_color('#333'); parts[key].set_linewidth(0.7)
    ax.axhline(0, color='#555', lw=0.5, ls='--', alpha=0.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['pre', 'post'], fontsize=PUB_FS - 2)
    ax.set_yticks([])


def _mini_gwas(ax):
    np.random.seed(7)
    n = 180
    chrom = np.random.randint(1, 11, n)
    pvals = np.random.exponential(0.3, n)
    pvals[np.random.choice(n, 6, replace=False)] = np.random.uniform(5.5, 9, 6)
    xs    = chrom + np.random.uniform(-0.4, 0.4, n)
    cols  = ['#8E24AA' if c % 2 == 0 else '#CE93D8' for c in chrom]
    ax.scatter(xs, pvals, s=2, c=cols, alpha=0.7, linewidths=0)
    ax.axhline(5.3, color='#B71C1C', lw=0.7, ls='--')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel('chromosome', fontsize=PUB_FS - 2, labelpad=1)


def _mini_umap(ax):
    np.random.seed(3)
    for (cx, cy), col in zip([(-2.5, 1.5), (2.0, 2.0), (0.0, -2.5)],
                              ['#EF6C00', '#1565C0', '#2E7D32']):
        ax.scatter(cx + np.random.randn(40) * 0.8,
                   cy + np.random.randn(40) * 0.7,
                   s=3, color=col, alpha=0.7, linewidths=0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel('UMAP 1', fontsize=PUB_FS - 2, labelpad=1)
    ax.set_ylabel('UMAP 2', fontsize=PUB_FS - 2, labelpad=1)


# ── Task definitions (angle in degrees, math convention: 0=right, CCW) ───────
TASKS = [
    dict(key='trait',   angle=90,   label='Trait Prediction',
         sublabel='linear probe  →  R²',
         examples='BMI · bone density · lean mass'),
    dict(key='aging',   angle=30,   label='Biological Aging Rate',
         sublabel='age gap  →  longitudinal rate  →  quartiles',
         examples='slow agers (Q1)  vs  fast agers (Q4)'),
    dict(key='disease', angle=-30,  label='Disease Risk',
         sublabel='Cox survival model  →  HR',
         examples='OA · T2D · fatty liver'),
    dict(key='drug',    angle=-90,  label='Drug Response',
         sublabel='paired before / after  →  age gap shift',
         examples='statins · HRT · GLP-1 agonists'),
    dict(key='gwas',    angle=-150, label='GWAS',
         sublabel='embedding-based association',
         examples='genome-wide hits'),
    dict(key='cluster', angle=150,  label='Phenotypic Clustering',
         sublabel='UMAP · k-means',
         examples='population subgroups'),
]
MINI_FNS = [_mini_scatter, _mini_aging, _mini_km,
            _mini_drug, _mini_gwas, _mini_umap]


# ── Figure ───────────────────────────────────────────────────────────────────
def build_figure():
    fig = plt.figure(figsize=(W, H), facecolor='white')
    canvas = fig.add_axes([0, 0, 1, 1])
    canvas.set_xlim(0, W); canvas.set_ylim(0, H); canvas.axis('off')

    ENC_CX, ENC_CY = W / 2, H / 2      # 6.5, 5.5
    ENC_W,  ENC_H  = 1.6, 0.95         # smaller — just an embedding marker
    RADIUS  = 3.95
    TASK_W, TASK_H = 2.65, 1.90
    MINI_W, MINI_H = 1.05, 1.00        # data-coord size of mini-axes

    # ── Centre embeddings marker (model schema is in Panel A) ─────────────
    enc = FancyBboxPatch(
        (ENC_CX - ENC_W / 2, ENC_CY - ENC_H / 2), ENC_W, ENC_H,
        boxstyle="round,pad=0.005,rounding_size=0.10",
        fc=ENC_FC, ec=ENC_EC, lw=1.2)
    canvas.add_patch(enc)

    canvas.text(ENC_CX, ENC_CY + 0.18, 'Embeddings',
                ha='center', va='center',
                fontsize=TITLE_FS, fontweight='bold', color='#263238')
    canvas.text(ENC_CX, ENC_CY - 0.18, r'$z \in \mathbb{R}^{d}$',
                ha='center', va='center',
                fontsize=PUB_FS, color='#3949AB')

    # ── Task boxes in a circle ────────────────────────────────────────────
    for task, mini_fn in zip(TASKS, MINI_FNS):
        θ = np.radians(task['angle'])
        cos_t, sin_t = np.cos(θ), np.sin(θ)

        tcx = ENC_CX + RADIUS * cos_t
        tcy = ENC_CY + RADIUS * sin_t
        tx0 = tcx - TASK_W / 2
        ty0 = tcy - TASK_H / 2
        color = TASK_COLORS[task['key']]

        # Dashed task box
        p = FancyBboxPatch((tx0, ty0), TASK_W, TASK_H,
                           boxstyle="round,pad=0.005,rounding_size=0.06",
                           fc='#FAFAFA', ec=color, lw=1.1, linestyle='--')
        canvas.add_patch(p)

        # Labels (left portion of box)
        canvas.text(tx0 + 0.13, ty0 + TASK_H - 0.16,
                    task['label'], ha='left', va='top',
                    fontsize=PUB_FS, fontweight='bold', color=color)
        canvas.text(tx0 + 0.13, ty0 + TASK_H - 0.36,
                    task['sublabel'], ha='left', va='top',
                    fontsize=PUB_FS - 1.5, color='#444', fontstyle='italic')
        canvas.text(tx0 + 0.13, ty0 + 0.12,
                    task['examples'], ha='left', va='bottom',
                    fontsize=PUB_FS - 1.5, color='#666')

        # Mini-plot (right portion of box) — in figure-fraction coords
        mx_data = tx0 + TASK_W - MINI_W - 0.08
        my_data = tcy - MINI_H / 2
        mini_ax = fig.add_axes([mx_data / W, my_data / H,
                                 MINI_W / W, MINI_H / H])
        mini_fn(mini_ax)
        _style_mini(mini_ax)

        # Arrow: encoder-box edge  →  task-box edge
        ax0, ay0 = _rect_edge(ENC_CX, ENC_CY, ENC_W, ENC_H,  cos_t,  sin_t)
        ax1, ay1 = _rect_edge(tcx,    tcy,    TASK_W, TASK_H, -cos_t, -sin_t)
        canvas.add_patch(FancyArrowPatch(
            (ax0, ay0), (ax1, ay1),
            arrowstyle='-|>', color='#888', lw=1.0, mutation_scale=10))

    canvas.text(0.18, H - 0.18, 'b)', ha='left', va='top',
                fontsize=TITLE_FS + 3, fontweight='bold')
    return fig


if __name__ == '__main__':
    fig = build_figure()
    fig.savefig(OUT_PDF, bbox_inches='tight', pad_inches=0.1)
    fig.savefig(OUT_PNG, bbox_inches='tight', pad_inches=0.1, dpi=300)
    print(f'Saved {OUT_PDF} and {OUT_PNG}')
