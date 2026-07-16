import sys
import os
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.extend([base_path, os.path.join(base_path, 'LabData'), os.path.join(base_path, 'LabUtils'), os.path.join(base_path, 'LabQueue')])
import re
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from scipy.stats import mannwhitneyu, wilcoxon, pearsonr, ks_2samp
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')

# --- PLOTTING SETUP FOR PAPER ---
sns.set_theme(style="ticks", font_scale=1.1)
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'],
    'axes.labelsize': 13,
    'axes.titlesize': 12,
    'legend.fontsize': 11,
    'legend.title_fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# --- PAPER COLOUR PALETTE ---
PALETTE_QUARTILE = {'Q1': '#2471a3', 'Q2': '#76b7c8', 'Q3': '#f0a07a', 'Q4': '#c0392b'}
PALETTE_SEX      = {'Female': '#c0392b', 'Male': '#2471a3', None: '#555555'}
PALETTE_PERIOD   = {'Before': '#95a5a6', 'After': '#2c3e50'}

# ATC3 -> readable drug class name mapping
ATC3_NAMES = {
    'A02B': 'Antacids/PPI', 'A10B': 'Antidiabetics/GLP-1', 'A11C': 'Vitamin A/D',
    'A11D': 'Vitamin B1', 'A11G': 'Vitamin C', 'A12A': 'Calcium', 'A12C': 'Other Minerals',
    'B01A': 'Antithrombotics', 'B02B': 'Vitamin K/Hemostatics', 'B03A': 'Iron Preps',
    'B03B': 'Vitamin B12/Folic', 'C07A': 'Beta Blockers', 'C08C': 'Ca-Channel Blockers',
    'C09A': 'ACE Inhibitors', 'C09C': 'ARBs Plain', 'C09D': 'ARBs Combo',
    'C10A': 'Statins', 'C10B': 'Lipid Combo', 'G02C': 'Uterine Stimulants',
    'G03C': 'Estrogens', 'G03D': 'Progestogens', 'G03F': 'Progest+Estrogen (HRT)',
    'G03B': 'Androgens/Testosterone', 'G04C': 'BPH Drugs', 'H02A': 'Systemic Corticosteroids',
    'H05B': 'Calcitonin', 'L02A': 'GnRH Analogues', 'L02B': 'Anti-Hormonal (AIs/Anti-Androgens)',
    'M04A': 'Antigout', 'M05B': 'Bone Drugs',
    'N03A': 'Antiepileptics', 'N05C': 'Hypnotics/Sedatives', 'N06A': 'Antidepressants',
    'N06B': 'ADHD/Stimulants', 'R03A': 'Adrenergics Inhalants', 'R06A': 'Antihistamines',
    'OTHE': 'Other', 'UNKN': 'Unknown',
}
def atc3_label(code):
    name = ATC3_NAMES.get(code, code)
    return f"{code} ({name})" if name != code else code

# --- CONFIGURATION ---
LEJEPA_PATH   = "/data/hpp_labdata/Analyses/gilsa/embeddings/embeddings_with_date.pkl"
TARGETS_PATH  = "/path/to/project/targets_for_downstream_full.csv"
GLP1_PATH     = "/data/hpp_labdata/Analyses/nastya/GLP1/all_glp1_meds_logged.csv"

OUTPUT_DIR = "age_prediction_analysis"
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
PARTITIONS_OUT = os.path.join(OUTPUT_DIR, "age_prediction_rate_partitions.csv")

FOLDS = 10
AGE_PARTITION_QUANTILES = 4
MIN_TIME_GAP_YEARS = 1.0
RATE_COL = 'rate_lejepa'
RATE_COL_RTM = 'rate_lejepa_rtm_corrected'
MODEL_COL_RTM = 'bin_rate_lejepa_rtm'
TOP_N_PLOTS = 6
MIN_POPULAR_DRUG_USERS = 20
MIN_PAIRED_SUBJECTS = 10
TOP_HIST_ATC = 5

HPP_BODY_SYSTEMS_PATH = "/data/hpp_labdata/Analyses/10K_Trajectories/body_systems/"
QC_N_SD = 4.0
DEXA_SYSTEMS = ['bone_density', 'frailty']

# BMD-altering drug classes (4-char ATC codes, matching infrastructure convention)
# Increasing: used as positive controls for BMD signal validation
# Decreasing: rare (<BMD_RARITY_THRESHOLD users) → exclude from all analyses;
#             common (≥threshold) → dedicated BMD phenotype analysis
BMD_INCREASING_DRUGS = ['M05B', 'G03C', 'G03F', 'A11CC', 'H05BA', 'G03BA']
BMD_DECREASING_DRUGS = ['H02AB', 'L02AE', 'L02BG', 'L02BB', 'N06AB', 'A02BC', 'N03A']
BMD_RARITY_THRESHOLD = 50
# Always exclude regardless of prevalence (cancer/highly specific disease populations)
BMD_FORCED_EXCLUDE = ['L02BG']
HPP_SYSTEMS  = ['blood_tests_lipids', 'cardiovascular_system', 'gait', 'sleep', 'mental',
                'glycemic_status', 'immune_system']

# 1-3 representative phenotypes per HPP body system (avoids multiple-testing dilution)
HPP_REPRESENTATIVES = {
    'blood_tests_lipids':     ['bt__hdl_cholesterol', 'bt__triglycerides', 'bt__total_cholesterol'],
    'cardiovascular_system':  ['sitting_blood_pressure_systolic', 'sitting_blood_pressure_diastolic',
                                'hr_bpm', 'intima_media_th_mm_1_intima_media_thickness'],
    'gait':                   ['Walking_speed_kmh: median - TMS', 'Cadence: median - TMS',
                                'chair-rise time (ms): median - sit_to_stand'],
    'sleep':                  ['ahi', 'total_sleep_time', 'sleep_efficiency'],
    'mental':                 ['health_satisfaction', 'rds_score', 'happiness_level'],
    'glycemic_status':        ['iglu_mean', 'iglu_cv', 'iglu_gmi'],
    'immune_system':          ['bt__wbc', 'bt__neutrophils_abs', 'bt__lymphocytes_abs', 'bt__monocytes_abs'],
}

def is_atc3_code(code):
    s = str(code).strip().upper()
    return len(s) == 4 and s[0].isalpha() and s[1:].isalnum()

# =============================================================================
# CORE PIPELINE: PREDICTIONS & RATES
# =============================================================================

def clean_format(df):
    if isinstance(df, pd.Series): df = df.to_frame()
    df = df.reset_index()
    if 'date' in df.columns and 'Date' not in df.columns: df = df.rename(columns={'date': 'Date'})
    if 'RegistrationCode' in df.columns:
        df['RegistrationCode'] = df['RegistrationCode'].astype(str).apply(lambda x: f"10K_{x}" if not x.startswith("10K_") else x)
    if 'research_stage' in df.columns:
        df['research_stage'] = df['research_stage'].astype(str).apply(lambda x: 'baseline' if x == '00_00_visit' else x)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'].astype(str).str.replace('_', '-'), errors='coerce')
    return df

def generate_predictions_pipeline():
    print("\n=== STEP 1: ML Prediction (RidgeCV) ===")
    df_lejepa = clean_format(pd.read_pickle(LEJEPA_PATH))
    meta_cols = [c for c in ['RegistrationCode', 'research_stage', 'Date'] if c in df_lejepa.columns]
    emb_cols = [c for c in df_lejepa.columns if c not in meta_cols]
    df_lejepa = df_lejepa.rename(columns={c: f"lej_{i}" for i, c in enumerate(emb_cols)})
    
    df_targets = clean_format(pd.read_csv(TARGETS_PATH))
    df_targets['age'] = df_targets.get('age', df_targets.get('Age'))
        
    df_train = pd.merge(df_lejepa, df_targets[['RegistrationCode', 'research_stage', 'age']], on=['RegistrationCode', 'research_stage'], how='inner')
    df_train = df_train.dropna(subset=['age', 'Date']).drop_duplicates(subset=['RegistrationCode', 'research_stage'])

    X = df_train[[f"lej_{i}" for i in range(len(emb_cols))]].values.astype(np.float32)
    y = df_train['age'].values
    preds = np.zeros_like(y, dtype=np.float64)

    for train_ix, test_ix in GroupKFold(n_splits=FOLDS).split(X, y, df_train['RegistrationCode'].values):
        pipe = Pipeline([('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler()), ('ridge', RidgeCV(alphas=np.logspace(-3, 4, 100)))])
        y_mean, y_std = y[train_ix].mean(), y[train_ix].std()
        pipe.fit(X[train_ix], (y[train_ix] - y_mean) / y_std)
        preds[test_ix] = pipe.predict(X[test_ix]) * y_std + y_mean

    df_train['age_pred_lejepa'] = preds
    
    r_val, _ = pearsonr(y, preds)
    mae = mean_absolute_error(y, preds)
    print(f'  > Age prediction: r={r_val:.3f}, MAE={mae:.2f}')
    
    # Paper-ready Age Prediction Plot
    fig, ax = plt.subplots(figsize=(5.5, 5))
    hb = ax.hexbin(y, preds, gridsize=55, cmap='Blues', mincnt=1, linewidths=0.2)
    cb = fig.colorbar(hb, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label('Count', fontsize=11)
    cb.ax.tick_params(labelsize=10)
    lims = [min(y.min(), preds.min()), max(y.max(), preds.max())]
    ax.plot(lims, lims, 'k--', lw=1.5, alpha=0.7, zorder=3)
    ax.text(0.05, 0.93, f"$r = {r_val:.3f}$\n$MAE = {mae:.2f}$ yr",
            transform=ax.transAxes, va='top', fontsize=11, color='#333333')
    ax.set_xlabel('Chronological age (years)')
    ax.set_ylabel('Predicted age (years)')
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "age_prediction_scatter.pdf"), format='pdf')
    plt.close()
    
    return df_train.rename(columns={'age': 'age_true'})[meta_cols + ['age_true', 'age_pred_lejepa']]

def calculate_rates_and_partition(df_preds):
    print("\n=== STEP 2: Calculating Age Residuals & RTM-Corrected Rates ===")
    X_age = PolynomialFeatures(degree=2).fit_transform(df_preds[['age_true']])
    df_preds['expected_pred_age'] = LinearRegression().fit(X_age, df_preds['age_pred_lejepa']).predict(X_age)
    df_preds['age_residual'] = df_preds['age_pred_lejepa'] - df_preds['expected_pred_age']
    
    df_preds = df_preds.sort_values(by=['RegistrationCode', 'age_true'])
    rate_data = []
    
    for reg, group in df_preds.groupby('RegistrationCode'):
        if len(group) >= 2:
            first, last = group.iloc[0], group.iloc[-1]
            t_diff = last['age_true'] - first['age_true']
            if t_diff >= MIN_TIME_GAP_YEARS:
                rate_data.append({
                    'RegistrationCode': reg, 
                    RATE_COL: (last['age_residual'] - first['age_residual']) / t_diff,
                    'rate_start_date': first['Date'],
                    'rate_end_date': last['Date']
                })
                
    df_rates = pd.DataFrame(rate_data)
    
    # RTM correction
    baseline_residuals = df_preds.sort_values('age_true').drop_duplicates('RegistrationCode', keep='first')[['RegistrationCode', 'age_residual']].rename(columns={'age_residual': 'baseline_residual'})
    if not df_rates.empty:
        df_rates = df_rates.merge(baseline_residuals, on='RegistrationCode', how='left')
        
        mask = df_rates[['baseline_residual', RATE_COL]].notna().all(axis=1)
        X_bl = df_rates.loc[mask, 'baseline_residual'].values.reshape(-1, 1)
        y_rate = df_rates.loc[mask, RATE_COL].values
        reg = LinearRegression().fit(X_bl, y_rate)
        df_rates.loc[mask, RATE_COL_RTM] = y_rate - reg.predict(X_bl)
        
        valid = df_rates[RATE_COL_RTM].notna()
        if valid.sum() > 0:
            df_rates.loc[valid, MODEL_COL_RTM] = pd.qcut(
                df_rates.loc[valid, RATE_COL_RTM].rank(method='first'),
                q=AGE_PARTITION_QUANTILES, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    
    return df_preds.merge(df_rates, on='RegistrationCode', how='left')

# =============================================================================
# EXTREME PHENOTYPES (Q1 vs Q4)
# =============================================================================

def plot_phenotype_extremes(df_data, group_col, df_targets, analysis_name, sex_filter=None):
    print(f"\n=== STEP 3: Phenotype Analysis ({analysis_name} Extremes) ===")
    
    if 'gender' in df_targets.columns:
        df_gender = df_targets[['RegistrationCode', 'gender']].drop_duplicates()
    else:
        _meta = clean_format(pd.read_csv(TARGETS_PATH))
        df_gender = _meta[['RegistrationCode', 'gender']].drop_duplicates()
    df_gender['Gender'] = df_gender['gender'].map({0: 'Female', 1: 'Male'})

    # Replace physiologically impossible zeros with NaN (these encode missing data in HPP)
    ZERO_IS_MISSING = {'bt__hba1c', 'bt__glucose'}
    for col in ZERO_IS_MISSING:
        if col in df_targets.columns:
            df_targets[col] = df_targets[col].replace(0, np.nan)

    if sex_filter in ['Female', 'Male']:
        sex_code = 0 if sex_filter == 'Female' else 1
        allowed_regs = set(df_gender[df_gender['gender'] == sex_code]['RegistrationCode'])
        df_data = df_data[df_data['RegistrationCode'].isin(allowed_regs)].copy()
        df_targets = df_targets[df_targets['RegistrationCode'].isin(allowed_regs)].copy()
        df_gender = df_gender[df_gender['gender'] == sex_code].copy()
    
    num_cols = [c for c in df_targets.select_dtypes(include=np.number).columns if c not in ['age', 'Age', 'gender', 'index', 'level_0']]
    
    is_rate_analysis = analysis_name.startswith("Aging Rate")

    if is_rate_analysis:
        sort_col = 'Date' if 'Date' in df_targets.columns else 'age'
        df_targets = df_targets.sort_values(['RegistrationCode', sort_col])
        first = df_targets.groupby('RegistrationCode')[num_cols + ['age']].first()
        last = df_targets.groupby('RegistrationCode')[num_cols + ['age']].last()
        tdiff = last['age'] - first['age']
        valid = tdiff[tdiff >= MIN_TIME_GAP_YEARS].index
        pheno_vals = (last.loc[valid, num_cols] - first.loc[valid, num_cols]).div(tdiff.loc[valid], axis=0).reset_index()
        y_label = "Annualized Rate of Change"
    else:
        pheno_vals = df_targets[['RegistrationCode', 'research_stage'] + num_cols]
        y_label = "Raw Baseline Value"

    if is_rate_analysis:
        df_merged = df_data.merge(pheno_vals, on=['RegistrationCode']).merge(df_gender, on='RegistrationCode')
    else:
        df_merged = df_data.merge(pheno_vals, on=['RegistrationCode', 'research_stage']).merge(df_gender, on='RegistrationCode')
        
    df_extremes = df_merged[df_merged[group_col].isin(['Q1', 'Q4'])]

    results = []
    for pheno in num_cols:
        data = df_extremes[[group_col, pheno]].dropna()
        q1, q4 = data[data[group_col] == 'Q1'][pheno], data[data[group_col] == 'Q4'][pheno]
        if len(q1) >= 50 and len(q4) >= 50 and data[pheno].std() > 1e-6:
            _, p_val = mannwhitneyu(q1, q4)
            pooled_std = np.sqrt((q1.std()**2 + q4.std()**2) / 2)
            cohens_d = (q4.mean() - q1.mean()) / pooled_std if pooled_std > 1e-6 else 0
            row = {'Phenotype': pheno, 'Mean_Q1': q1.mean(), 'Mean_Q4': q4.mean(), 'Median_Q1': q1.median(), 'Median_Q4': q4.median(), 'Median_Diff': q4.median() - q1.median(), 'Mean_Diff': q4.mean() - q1.mean(), 'Cohens_d': cohens_d, 'N_Q1': len(q1), 'N_Q4': len(q4), 'P_Value': p_val}
            results.append(row)
            
    if not results: return
    df_res = pd.DataFrame(results).sort_values('P_Value')
    _, df_res['Adjusted_P_Value'], _, _ = multipletests(df_res['P_Value'], alpha=0.05, method='fdr_bh')
    
    res_out = os.path.join(OUTPUT_DIR, f"{analysis_name.replace(' ', '_').lower()}_phenotype_results.csv")
    df_res.to_csv(res_out, index=False)
    
    top_phenos = df_res.head(TOP_N_PLOTS)['Phenotype'].tolist()
    n_cols = 3
    n_rows = (len(top_phenos) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4.2 * n_rows))
    axes = axes.flatten()

    box_color = PALETTE_SEX.get(sex_filter, '#555555')

    for i, pheno in enumerate(top_phenos):
        ax = axes[i]
        subset = df_extremes[[group_col, pheno]].dropna()
        row_stats = df_res[df_res['Phenotype'] == pheno].iloc[0]
        q_val = row_stats['Adjusted_P_Value']
        cohens_d_val = row_stats['Cohens_d']

        # Trim outliers for y-axis window only
        p02, p98 = subset[pheno].quantile(0.02), subset[pheno].quantile(0.98)
        padding = (p98 - p02) * 0.12
        ax.set_ylim(p02 - padding, p98 + padding * 2.5)

        sns.boxplot(data=subset, x=group_col, y=pheno, order=['Q1', 'Q4'],
                    color=box_color, ax=ax, showfliers=False, width=0.45,
                    boxprops=dict(alpha=0.55, edgecolor='black', linewidth=1.4),
                    medianprops=dict(color='black', linewidth=2),
                    whiskerprops=dict(color='black', linewidth=1.2),
                    capprops=dict(color='black', linewidth=1.2))

        # Significance bracket
        y_top = p98 + padding * 1.2
        y_bar = p98 + padding * 1.8
        ax.plot([0, 0, 1, 1], [y_top, y_bar, y_bar, y_top], lw=1.2, color='#333333')
        sig_label = '***' if q_val < 0.001 else ('**' if q_val < 0.01 else ('*' if q_val < 0.05 else 'ns'))
        ax.text(0.5, y_bar + padding * 0.1, sig_label, ha='center', va='bottom', fontsize=11, color='#333333')

        pheno_name = pheno.replace('_', ' ').title()
        if len(pheno_name) > 28: pheno_name = pheno_name[:25] + "…"
        ax.set_title(f"{pheno_name}\nadj. $P={q_val:.1e}$, $d={cohens_d_val:.2f}$",
                     fontsize=10, color='#333333', pad=4)
        ax.set_ylabel(y_label if i % n_cols == 0 else '', fontsize=11)
        ax.set_xlabel('')
        ax.set_xticklabels(['Q1 (slow)', 'Q4 (fast)'] if is_rate_analysis else ['Q1 (young)', 'Q4 (old)'])
        sns.despine(ax=ax)

    for j in range(i + 1, len(axes)): axes[j].axis('off')
    plt.tight_layout(h_pad=2.5, w_pad=1.5)
    plt.savefig(os.path.join(OUTPUT_DIR, f"{analysis_name.replace(' ', '_').lower()}_phenotypes.pdf"), format='pdf')
    plt.close()


# =============================================================================
# MEDICATIONS: LMM & PAIRED
# =============================================================================

def load_medications():
    """Loads medications and injects GLP-1s; returns (start_events, meds_loader)."""
    meds_loader = None
    try:
        from LabData.DataLoaders.Medications10KLoader import Medications10KLoader
        meds_loader = Medications10KLoader(gen_cache=False)
        meds_df = meds_loader.get_data().df.reset_index()
        meds_df['Date'] = pd.to_datetime(meds_df['Date'], errors='coerce', utc=True).dt.tz_localize(None)
        start_events = meds_df[(meds_df['Start'] == True) & meds_df['Date'].notna()].copy()

        # Enrich ATC3_NAMES with the loader's official ATC ontology labels (fills gaps in our manual dict)
        if hasattr(meds_loader, '_atc_to_label'):
            for _code, _label in meds_loader._atc_to_label.items():
                if len(_code) == 4 and _code not in ATC3_NAMES:
                    ATC3_NAMES[_code] = _label

        meds_meta = getattr(meds_loader, '_columns_metadata', pd.DataFrame())
        if isinstance(meds_meta, pd.DataFrame) and not meds_meta.empty and 'column_name' in meds_meta.columns:
            atc_source_col = 'ATC' if 'ATC' in meds_meta.columns else ('ATC4' if 'ATC4' in meds_meta.columns else None)
            if atc_source_col is not None:
                atc_map = meds_meta[['column_name', atc_source_col]].copy()
                atc_map.columns = ['medication', 'atc_raw']
                atc_map['medication'] = atc_map['medication'].astype(str).str.strip()
                atc_map['atc_raw'] = atc_map['atc_raw'].astype(str).str.strip().str.upper()
                atc_map = atc_map.dropna(subset=['medication', 'atc_raw']).drop_duplicates('medication')
                atc_map = atc_map[~atc_map['atc_raw'].isin(['', 'EMPTY', 'NAN'])]
                start_events['medication'] = start_events['medication'].astype(str).str.strip()
                start_events = start_events.merge(atc_map, on='medication', how='left')
                start_events['atc3'] = start_events['atc_raw'].str[:4]
                start_events.rename(columns={'atc_raw': 'atc_full'}, inplace=True)
    except Exception as e:
        print(f"  > Could not load Medications10KLoader: {e}")
        start_events = pd.DataFrame(columns=['RegistrationCode', 'Date', 'medication', 'Start'])

    try:
        glp1_df = clean_format(pd.read_csv(GLP1_PATH)).dropna(subset=['Date'])
        glp1_df['Date'] = pd.to_datetime(glp1_df['Date'], errors='coerce', utc=True).dt.tz_localize(None)
        glp1_starts = glp1_df.groupby('RegistrationCode')['Date'].min().reset_index()
        glp1_starts['medication'] = 'GLP-1 Agonists'
        glp1_starts['atc3'] = 'A10B'
        glp1_starts['Start'] = True
        start_events = pd.concat([start_events, glp1_starts], ignore_index=True)
    except Exception as e:
        pass

    if 'Date' in start_events.columns:
        start_events['Date'] = pd.to_datetime(start_events['Date'], errors='coerce', utc=True).dt.tz_localize(None)
        start_events = start_events.dropna(subset=['Date'])

    if start_events.empty:
        return None, meds_loader

    atc_col = next((c for c in ['atc', 'ATC', 'atc_code', 'ATC_code'] if c in start_events.columns), None)
    if atc_col is not None and 'atc3' not in start_events.columns:
        start_events['atc3'] = start_events[atc_col].astype(str).str.strip().str.upper().str[:4]
    start_events['atc3'] = start_events['atc3'].astype(str).str.strip()
    return start_events[start_events['atc3'].str.len() == 4].copy(), meds_loader

def _build_analyzable_drugs(start_events):
    """Return list of {drug, label, first_start} for ATC3 codes with enough users."""
    popular = (start_events.groupby('atc3')['RegistrationCode'].nunique()
               .pipe(lambda s: s[s >= MIN_POPULAR_DRUG_USERS]).index.tolist())
    return [
        {'drug': code, 'label': atc3_label(code),
         'first_start': start_events[start_events['atc3'] == code]
                        .groupby('RegistrationCode')['Date'].min()}
        for code in popular
    ]

def run_paired_medication_analysis(df_final, df_targets, analyzable_drugs):
    print("\n=== STEP 4B: Paired Crossover Analysis (Strict Before/After) ===")
    visits = df_final[['RegistrationCode', 'Date', 'age_residual']].dropna().copy()
    visits['Date'] = pd.to_datetime(visits['Date'])
    gender_df = df_targets[['RegistrationCode', 'gender']].drop_duplicates() if 'gender' in df_targets.columns else None
    
    paired_results = []
    
    for group in analyzable_drugs:
        drug = group['drug']
        first_start = group['first_start']
        drug_visits = visits[visits['RegistrationCode'].isin(first_start.index)].copy()
        
        demo = "All"
        if gender_df is not None:
            genders = drug_visits.merge(gender_df, on='RegistrationCode')['gender']
            female_ratio = (genders == 0).mean() if len(genders) > 0 else 0.5
            if female_ratio >= 0.90: demo, allowed = "Female", 0
            elif female_ratio <= 0.10: demo, allowed = "Male", 1
            else: allowed = None
            if allowed is not None: drug_visits = drug_visits[drug_visits['RegistrationCode'].isin(gender_df[gender_df['gender'] == allowed]['RegistrationCode'])]
        
        drug_visits['start_date'] = drug_visits['RegistrationCode'].map(first_start)
        drug_visits = drug_visits.dropna(subset=['start_date'])
        drug_visits['period'] = np.where(drug_visits['Date'] < drug_visits['start_date'], 'Before', 'After')
        
        subj_means = drug_visits.groupby(['RegistrationCode', 'period'])['age_residual'].mean().unstack()
        if 'Before' not in subj_means.columns or 'After' not in subj_means.columns: continue
            
        subj_means = subj_means.dropna(subset=['Before', 'After'])
        
        if len(subj_means) >= MIN_PAIRED_SUBJECTS:
            delta = subj_means['After'] - subj_means['Before']
            _, pval = wilcoxon(subj_means['Before'], subj_means['After'])
            paired_results.append({
                'drug': drug, 'demo': demo, 'N_paired': len(subj_means),
                'mean_before': subj_means['Before'].mean(), 'mean_after': subj_means['After'].mean(), 
                'mean_delta': delta.mean(), 'p_value': pval,
            })

    paired_df = pd.DataFrame(paired_results)
    if paired_df.empty: return pd.DataFrame(), []
    
    paired_df = paired_df.sort_values('p_value')
    _, paired_df['adjusted_p_value'], _, _ = multipletests(paired_df['p_value'], alpha=0.05, method='fdr_bh')
    paired_df.to_csv(os.path.join(OUTPUT_DIR, "medications_atc3_paired_before_after.csv"), index=False)
    
    sig_paired_drugs = paired_df[(paired_df['adjusted_p_value'] < 0.05) | (paired_df['p_value'] < 0.10)]['drug'].tolist()
    sig_paired_drugs = list(dict.fromkeys([d for d in sig_paired_drugs if is_atc3_code(d)]))[:TOP_HIST_ATC]
    
    # Plot top 6 paired crossovers
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5))
    for i, (_, row) in enumerate(paired_df.head(6).iterrows()):
        ax, drug = axes.flatten()[i], row['drug']
        first_start = next(g['first_start'] for g in analyzable_drugs if g['drug'] == drug)
        dvis = visits[visits['RegistrationCode'].isin(first_start.index)].copy()
        dvis['start_date'] = dvis['RegistrationCode'].map(first_start)
        dvis['period'] = np.where(dvis['Date'] < dvis['start_date'], 'Before', 'After')
        dvis = dvis[dvis.groupby('RegistrationCode')['period'].transform('nunique') == 2]

        sns.boxplot(data=dvis, x='period', y='age_residual', order=['Before', 'After'],
                    palette=PALETTE_PERIOD, showfliers=False, ax=ax, width=0.4,
                    boxprops=dict(alpha=0.85, edgecolor='black', linewidth=1.4),
                    medianprops=dict(color='black', linewidth=2),
                    whiskerprops=dict(color='black', linewidth=1.2),
                    capprops=dict(color='black', linewidth=1.2))

        # Per-subject spaghetti lines
        for _, p_row in dvis.groupby(['RegistrationCode', 'period'])['age_residual'].mean().unstack().dropna().iterrows():
            ax.plot([0, 1], [p_row['Before'], p_row['After']],
                    color='#555555', alpha=0.18, linewidth=0.8, zorder=1)

        ax.axhline(0, color='#888888', linestyle='--', linewidth=1, alpha=0.6, zorder=0)

        title_color = '#c0392b' if row['adjusted_p_value'] < 0.05 else '#333333'
        ax.set_title(
            f"{atc3_label(drug)}\n"
            f"$n={row['N_paired']}$, $\Delta={row['mean_delta']:+.2f}$ yr,  adj.$P={row['adjusted_p_value']:.1e}$",
            fontsize=9.5, color=title_color, pad=5)
        ax.set_ylabel('Age gap (years)' if i % 3 == 0 else '')
        ax.set_xlabel('')
        ax.set_xticklabels(['Before', 'After'])
        sns.despine(ax=ax)

    for j in range(i + 1, 6): axes.flatten()[j].axis('off')
    plt.tight_layout(h_pad=2.5, w_pad=1.5)
    plt.savefig(os.path.join(OUTPUT_DIR, "medications_atc3_top_paired_crossovers.pdf"), format='pdf')
    plt.close()
    
    return paired_df, sig_paired_drugs


def run_crosssectional_drug_analysis(df_final, df_targets, start_events, drug_codes):
    """Compare bioage gap (age_residual) between ever-users and non-users at baseline.
    Uses the first visit per subject; adjusts for sex by stratification."""
    print("\n=== STEP 4E: Cross-Sectional Users vs Non-Users (baseline gap) ===")

    # Baseline visit = first scan per subject
    baseline = (df_final[['RegistrationCode', 'Date', 'age_residual']]
                .dropna()
                .sort_values('Date')
                .drop_duplicates('RegistrationCode', keep='first')
                .set_index('RegistrationCode'))

    gender_map = {}
    if 'gender' in df_targets.columns:
        gender_map = df_targets[['RegistrationCode', 'gender']].drop_duplicates().set_index('RegistrationCode')['gender'].to_dict()

    results = []
    for drug in drug_codes:
        drug = str(drug)
        if len(drug) == 5 and 'atc_full' in start_events.columns:
            users = set(start_events[start_events['atc_full'].str.startswith(drug, na=False)]['RegistrationCode'])
        else:
            users = set(start_events[start_events['atc3'] == drug[:4]]['RegistrationCode'])

        for sex_label, sex_code in [('All', None), ('Female', 0), ('Male', 1)]:
            if sex_code is not None:
                keep = {rc for rc, g in gender_map.items() if g == sex_code}
                sub = baseline[baseline.index.isin(keep)]
            else:
                sub = baseline

            user_gap   = sub.loc[sub.index.isin(users),     'age_residual'].dropna()
            nonuser_gap = sub.loc[~sub.index.isin(users),    'age_residual'].dropna()

            if len(user_gap) < 10 or len(nonuser_gap) < 50:
                continue

            _, p_val = mannwhitneyu(user_gap, nonuser_gap, alternative='two-sided')
            pooled_std = np.sqrt((user_gap.std()**2 + nonuser_gap.std()**2) / 2)
            d = (user_gap.mean() - nonuser_gap.mean()) / pooled_std if pooled_std > 1e-6 else 0.0
            results.append({
                'drug': drug,
                'label': ATC3_NAMES.get(drug[:4], drug),
                'demo': sex_label,
                'N_users': len(user_gap),
                'N_nonusers': len(nonuser_gap),
                'mean_gap_users': user_gap.mean(),
                'mean_gap_nonusers': nonuser_gap.mean(),
                'mean_diff': user_gap.mean() - nonuser_gap.mean(),
                'cohens_d': d,
                'p_value': p_val,
            })

    if not results:
        print("  No drug passed minimum-N threshold.")
        return pd.DataFrame()

    df_res = pd.DataFrame(results).sort_values('p_value')
    _, df_res['adjusted_p_value'], _, _ = multipletests(df_res['p_value'], alpha=0.05, method='fdr_bh')
    out_csv = os.path.join(OUTPUT_DIR, 'medications_crosssectional_users_vs_nonusers.csv')
    df_res.to_csv(out_csv, index=False)

    print(f"  Results saved → {out_csv}")
    sig = df_res[df_res['adjusted_p_value'] < 0.05]
    print(f"  FDR-significant: {len(sig)}/{len(df_res)}")
    print(df_res[['drug','demo','N_users','N_nonusers','mean_diff','cohens_d','p_value','adjusted_p_value']].to_string(index=False))
    return df_res


def run_significant_drug_phenotype_change_analysis(df_targets, start_events, significant_drugs, df_final,
                                                    out_prefix='medications'):
    print(f"\n=== STEP 4C: Phenotype Changes for Significant Paired Drugs ({out_prefix}) ===")
    if not significant_drugs: return

    targets = df_targets.copy()
    
    if 'Date' not in targets.columns:
        if 'Date' in df_final.columns and 'research_stage' in targets.columns:
            date_map = df_final[['RegistrationCode', 'research_stage', 'Date']].dropna(subset=['Date']).drop_duplicates(['RegistrationCode', 'research_stage'])
            targets = targets.merge(date_map, on=['RegistrationCode', 'research_stage'], how='left')
        else:
            return

    targets['Date'] = pd.to_datetime(targets['Date'], errors='coerce')
    targets = targets.dropna(subset=['RegistrationCode', 'Date'])

    num_cols = [c for c in targets.select_dtypes(include=np.number).columns if c not in ['age', 'Age', 'gender', 'index', 'level_0']]
    drug_codes = [d for d in significant_drugs if is_atc3_code(d) or (len(str(d)) == 5 and str(d)[0].isalpha())]

    results = []
    for drug in drug_codes:
        if len(drug) == 4 or 'atc_full' not in start_events.columns:
            first_start = start_events[start_events['atc3'] == drug].groupby('RegistrationCode')['Date'].min()
        else:
            first_start = start_events[start_events['atc_full'].str.startswith(drug, na=False)].groupby('RegistrationCode')['Date'].min()
        sub = targets[targets['RegistrationCode'].isin(first_start.index)].copy()
        sub['start_date'] = sub['RegistrationCode'].map(first_start)
        sub = sub.dropna(subset=['start_date'])
        sub['period'] = np.where(sub['Date'] < sub['start_date'], 'Before', 'After')

        per_subj = sub.groupby(['RegistrationCode', 'period'])[num_cols].mean().unstack('period')
        if ('Before' not in per_subj.columns.get_level_values(1)) or ('After' not in per_subj.columns.get_level_values(1)): continue

        subj_sex = sub.groupby('RegistrationCode')['gender'].first() if 'gender' in sub.columns else pd.Series(index=per_subj.index, dtype='float64')

        for sex_code, sex_label in [(0, 'Female'), (1, 'Male')]:
            sex_idx = subj_sex[subj_sex == sex_code].index
            sex_frame = per_subj.loc[per_subj.index.intersection(sex_idx)]
            sex_rows = []
            for ph in num_cols:
                b = sex_frame[(ph, 'Before')].dropna()
                a = sex_frame[(ph, 'After')].dropna()
                common_idx = b.index.intersection(a.index)
                if len(common_idx) < 5: continue
                b, a = b.loc[common_idx], a.loc[common_idx]
                d = a - b
                if np.nanstd(d.values) < 1e-8: p_val = np.nan
                else: _, p_val = wilcoxon(b.values, a.values)
                d_std = np.nanstd(d.values)
                cohens_d = float(np.mean(d.values)) / d_std if d_std > 1e-8 else 0.0

                sex_rows.append({
                    'drug': drug, 'drug_label': atc3_label(drug), 'sex': sex_label, 'phenotype': ph,
                    'N_paired': len(common_idx), 'cohens_d': cohens_d, 'p_value': p_val,
                })

            if sex_rows:
                sex_df = pd.DataFrame(sex_rows).sort_values('p_value', na_position='last')
                valid = sex_df['p_value'].notna()
                sex_df['adjusted_p_value'] = np.nan
                if valid.sum() > 0: _, sex_df.loc[valid, 'adjusted_p_value'], _, _ = multipletests(sex_df.loc[valid, 'p_value'], alpha=0.05, method='fdr_bh')
                results.extend(sex_df.to_dict('records'))

    if results:
        out_df = pd.DataFrame(results).sort_values(['drug', 'sex', 'p_value'], na_position='last')
        out_csv = os.path.join(OUTPUT_DIR, f'{out_prefix}_significant_hits_phenotype_changes_by_sex.csv')
        out_df.to_csv(out_csv, index=False)
        plot_significant_drug_phenotype_forest(out_csv, out_prefix=out_prefix)

def plot_significant_drug_phenotype_forest(pheno_csv_path, out_prefix='medications'):
    df = pd.read_csv(pheno_csv_path)
    sig_df = df[df['adjusted_p_value'] < 0.10].copy()
    if sig_df.empty: return
    top_drugs = sig_df['drug'].unique().tolist()
    
    for drug in top_drugs:
        plot_df = df[df['drug'] == drug].copy()
        sig_phenos = plot_df[plot_df['adjusted_p_value'] < 0.10]['phenotype'].unique()
        plot_df = plot_df[plot_df['phenotype'].isin(sig_phenos)]
        if plot_df.empty: continue
            
        plot_df['abs_effect'] = plot_df['cohens_d'].abs()
        pheno_order = plot_df.groupby('phenotype')['abs_effect'].mean().sort_values(ascending=False).index
        
        fig, ax = plt.subplots(figsize=(8, max(4, len(pheno_order) * 0.7)))
        ax.set_yticks(np.arange(len(pheno_order)))
        ax.set_yticklabels([p.replace('_', ' ').title() for p in pheno_order])
        
        ax.axvline(0, color='black', linestyle='-', alpha=0.8, linewidth=1.5, zorder=1)
        
        for sex, color, offset in [('Female', '#d62728', -0.15), ('Male', '#1f77b4', 0.15)]:
            sex_data = plot_df[plot_df['sex'] == sex].set_index('phenotype')
            for i, pheno in enumerate(pheno_order):
                if pheno in sex_data.index:
                    row = sex_data.loc[pheno]
                    strict_sig = row['adjusted_p_value'] < 0.05
                    ax.scatter(row['cohens_d'], i + offset, color=color, s=80 if strict_sig else 40,
                               edgecolors=color if strict_sig else 'none', linewidth=1.5,
                               facecolors=color if strict_sig else 'white', zorder=3)
                    align = 'left' if row['cohens_d'] > 0 else 'right'
                    ax.text(row['cohens_d'] + (0.05 if row['cohens_d'] > 0 else -0.05), i + offset, f"$n={int(row['N_paired'])}$", va='center', ha=align, fontsize=10, color='dimgray')

        ax.set_xlabel("Effect size (Cohen's $d$)")
        ax.invert_yaxis()
        ax.axvline(0, color='black', linestyle='-', alpha=0.5, linewidth=1, zorder=0)
        ax.grid(axis='x', alpha=0.2, linestyle='--')

        from matplotlib.lines import Line2D
        ax.legend(handles=[
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62728', markersize=9, label='Female'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', markersize=9, label='Male'),
        ], frameon=False, fontsize=10, loc='lower right')

        sns.despine(ax=ax)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{out_prefix}_phenotype_changes_{drug}.pdf"), format='pdf')
        plt.close()

# =============================================================================
# QUARTILE DISTRIBUTIONS — new figure
# =============================================================================

def plot_quartile_distributions(df_final):
    """KDE of RTM-corrected aging rate by quartile — makes the partitioning tangible."""
    if MODEL_COL_RTM not in df_final.columns or RATE_COL_RTM not in df_final.columns:
        return

    fig, ax = plt.subplots(figsize=(6, 4))

    # Drop to one row per subject (first visit with a valid quartile)
    df_plot = (df_final.dropna(subset=[MODEL_COL_RTM, RATE_COL_RTM])
               .drop_duplicates('RegistrationCode', keep='first'))

    for q, color in PALETTE_QUARTILE.items():
        subset = df_plot[df_plot[MODEL_COL_RTM] == q][RATE_COL_RTM].dropna()
        if len(subset) > 10:
            sns.kdeplot(subset, ax=ax, color=color, linewidth=2.2,
                        label=f'{q}  ($n={len(subset):,}$)')

    ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.55)
    ax.set_xlabel('Aging rate (RTM-corrected, Δ biological years / year)')
    ax.set_ylabel('Density')
    ax.legend(frameon=False, title='Aging quartile', fontsize=10, title_fontsize=10)
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'quartile_rate_distributions.pdf'), format='pdf')
    plt.close()




# =============================================================================
# COMPOSITE PAPER FIGURE — helpers
# =============================================================================

def _select_phenotypes(analysis_prefix, n_per_sex=2):
    """
    Return top-n phenotypes per sex as [(phenotype, sex_key, {sex: row}), ...],
    female entries first then male.
    """
    out = []
    for sex in ['female', 'male']:
        path = os.path.join(OUTPUT_DIR, f'{analysis_prefix}_{sex}_phenotype_results.csv')
        if not os.path.exists(path):
            continue
        for _, r in pd.read_csv(path).head(n_per_sex).iterrows():
            out.append((r['Phenotype'], sex, {sex: r}))
    return out


def _draw_pheno_panel(ax, df_extremes, quartile_col, pheno, scope, rows,
                      xticklabels, ylabel, is_first):
    """
    Draw a Q1 vs Q4 boxplot for the single most significant sex.
    When scope == 'both', the sex with the lower adjusted p-value is chosen;
    the sex is always noted in the panel title.
    """
    sub = df_extremes[[quartile_col, 'Gender', pheno]].dropna()
    if sub.empty:
        ax.axis('off')
        return

    # Resolve which sex to display
    if scope == 'both':
        p_f = rows['female']['Adjusted_P_Value'] if 'female' in rows else np.inf
        p_m = rows['male']['Adjusted_P_Value']   if 'male'   in rows else np.inf
        sex_key = 'female' if p_f <= p_m else 'male'
    else:
        sex_key = scope

    gender_label = 'Female' if sex_key == 'female' else 'Male'
    sub_sex = sub[sub['Gender'] == gender_label]
    if sub_sex.empty:
        ax.axis('off')
        return

    p02, p98 = sub_sex[pheno].quantile(0.02), sub_sex[pheno].quantile(0.98)
    pad = (p98 - p02) * 0.15
    ax.set_ylim(p02 - pad * 0.5, p98 + pad * 2.5)

    BOX_KW = dict(showfliers=False, width=0.52,
                  boxprops=dict(alpha=0.65, edgecolor='black', linewidth=1.2),
                  medianprops=dict(color='black', linewidth=1.8),
                  whiskerprops=dict(color='black', linewidth=1.0),
                  capprops=dict(color='black', linewidth=1.0))

    sns.boxplot(data=sub_sex, x=quartile_col, y=pheno,
                order=['Q1', 'Q4'], color=PALETTE_SEX[gender_label],
                ax=ax, **BOX_KW)

    # Single significance bracket
    r = rows.get(sex_key)
    if r is not None:
        q_val = r['Adjusted_P_Value']
        sig   = ('***' if q_val < 0.001 else '**' if q_val < 0.01
                 else '*' if q_val < 0.05 else 'ns')
        color = PALETTE_SEX[gender_label]
        y_br  = p98 + pad * 1.55
        y_ft  = y_br - pad * 0.38
        ax.plot([0, 0, 1, 1], [y_ft, y_br, y_br, y_ft], lw=1.0, color=color)
        ax.text(0.5, y_br + pad * 0.05, sig,
                ha='center', va='bottom', fontsize=10, color=color)

    pname = pheno.replace('_', ' ').title()
    if len(pname) > 26:
        pname = pname[:24] + '…'
    ax.set_title(f'{pname}  ({gender_label})', fontsize=10, color='#333333', pad=4)
    ax.set_ylabel(ylabel if is_first else '')
    ax.set_xlabel('')
    ax.set_xticklabels(xticklabels)
    sns.despine(ax=ax)


def _load_results_table(analysis_prefix, sex):
    path = os.path.join(OUTPUT_DIR, f'{analysis_prefix}_{sex}_phenotype_results.csv')
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    df = df.copy()
    df['sex'] = sex
    return df


def _build_forest_table(analysis_prefix, top_n=12):
    """Build a sex-aware phenotype table with effect size CIs and ranking score."""
    df_f = _load_results_table(analysis_prefix, 'female')
    df_m = _load_results_table(analysis_prefix, 'male')
    if df_f.empty and df_m.empty:
        return pd.DataFrame()

    df = pd.concat([df_f, df_m], ignore_index=True)
    df = df.dropna(subset=['Phenotype', 'Cohens_d', 'N_Q1', 'N_Q4', 'Adjusted_P_Value'])
    if df.empty:
        return pd.DataFrame()

    n1 = df['N_Q1'].astype(float).clip(lower=2)
    n2 = df['N_Q4'].astype(float).clip(lower=2)
    d = df['Cohens_d'].astype(float)

    # Approximate standard error for standardized mean difference.
    se_d = np.sqrt((n1 + n2) / (n1 * n2) + (d ** 2) / (2 * (n1 + n2 - 2)))
    df['ci_low'] = d - 1.96 * se_d
    df['ci_high'] = d + 1.96 * se_d
    df['n_total'] = n1 + n2
    df['adj_p'] = df['Adjusted_P_Value'].clip(lower=1e-300)
    df['score'] = np.abs(d) * np.sqrt(df['n_total']) * (-np.log10(df['adj_p']))

    # Keep one row per phenotype: whichever sex has stronger adjusted evidence.
    df = (df.sort_values(['Phenotype', 'adj_p', 'score'], ascending=[True, True, False])
            .drop_duplicates('Phenotype', keep='first')
            .sort_values(['score', 'adj_p'], ascending=[False, True])
            .head(top_n)
            .reset_index(drop=True))
    return df


def _draw_forest_panel(ax, forest_df, title, show_legend=True):
    if forest_df.empty:
        ax.text(0.5, 0.5, 'No phenotype results available', ha='center', va='center', fontsize=11)
        ax.axis('off')
        return

    y = np.arange(len(forest_df))
    colors = forest_df['sex'].map({'female': PALETTE_SEX['Female'], 'male': PALETTE_SEX['Male']}).fillna('#555555').values
    sig = forest_df['adj_p'] < 0.05

    ax.axvline(0, color='black', linewidth=1.0, alpha=0.65, zorder=0)
    for i, row in forest_df.iterrows():
        ax.plot([row['ci_low'], row['ci_high']], [i, i], color=colors[i], lw=1.8, alpha=0.8)
        ax.scatter(row['Cohens_d'], i, s=60,
                   facecolor=colors[i] if sig.iloc[i] else 'white',
                   edgecolor=colors[i], linewidth=1.5, zorder=3)

    names = [p.replace('_', ' ').title() for p in forest_df['Phenotype'].tolist()]
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("Effect size (Cohen's d)")
    ax.set_title(title, fontsize=10.5)
    ax.grid(axis='x', alpha=0.2, linestyle='--')

    xmax = np.nanmax(np.abs(np.r_[forest_df['ci_low'].values, forest_df['ci_high'].values]))
    xmax = max(xmax, 0.25)
    ax.set_xlim(-xmax * 1.35, xmax * 1.35)

    from matplotlib.lines import Line2D
    if show_legend:
        ax.legend(handles=[
            Line2D([0], [0], marker='o', color='w', markeredgecolor=PALETTE_SEX['Female'],
                   markerfacecolor=PALETTE_SEX['Female'], markersize=7, label='Female (adj. P<0.05 filled)'),
            Line2D([0], [0], marker='o', color='w', markeredgecolor=PALETTE_SEX['Male'],
                   markerfacecolor=PALETTE_SEX['Male'], markersize=7, label='Male (adj. P<0.05 filled)'),
        ], frameon=False, fontsize=7.5, loc='lower left')
    sns.despine(ax=ax)


def _draw_summary_panel(ax, df_preds, df_final, gap_forest, rate_forest, paired_df):
    ax.axis('off')

    n_subjects = int(df_final['RegistrationCode'].nunique()) if 'RegistrationCode' in df_final.columns else 0
    n_visits = int(df_final[['RegistrationCode', 'Date']].drop_duplicates().shape[0]) if 'Date' in df_final.columns else int(len(df_final))
    r_val, _ = pearsonr(df_preds['age_true'].values, df_preds['age_pred_lejepa'].values)
    mae = mean_absolute_error(df_preds['age_true'].values, df_preds['age_pred_lejepa'].values)

    sex_counts = {'Female': 0, 'Male': 0}
    if 'gender' in df_final.columns:
        g = df_final.drop_duplicates('RegistrationCode')['gender'].value_counts().to_dict()
        sex_counts['Female'] = int(g.get(0, 0))
        sex_counts['Male'] = int(g.get(1, 0))

    n_gap_sig = int((gap_forest['adj_p'] < 0.05).sum()) if not gap_forest.empty else 0
    n_rate_sig = int((rate_forest['adj_p'] < 0.05).sum()) if not rate_forest.empty else 0
    n_drug_sig = int((paired_df['adjusted_p_value'] < 0.05).sum()) if paired_df is not None and not paired_df.empty else 0

    txt = (
        "Study summary\n"
        f"Subjects: {n_subjects:,}   Visits: {n_visits:,}\n"
        f"Sex: Female {sex_counts['Female']:,} | Male {sex_counts['Male']:,}\n"
        f"Model (10-fold GroupKFold): r={r_val:.3f}, MAE={mae:.2f} yr\n"
        f"Baseline-gap phenotypes adj. P<0.05: {n_gap_sig}\n"
        f"Aging-rate phenotypes adj. P<0.05: {n_rate_sig}\n"
        f"Paired drugs adj. P<0.05: {n_drug_sig}"
    )
    ax.text(0.02, 0.98, txt, va='top', ha='left', fontsize=10.2,
            bbox=dict(boxstyle='round,pad=0.45', facecolor='#f7f7f7', edgecolor='#bfbfbf'))


_PHENO_LABELS = {
    'body_comp_total_lean_mass': 'Total lean mass',
    'body_comp_total_tissue_mass': 'Total tissue mass',
    'total_scan_vat_volume': 'VAT volume',
    'total_scan_sat_volume': 'SAT volume',
    'femur_neck_mean_bmd': 'Femur neck BMD',
    'body_total_bmd': 'Total BMD',
    'body_spine_bmd': 'Spine BMD',
    'spine_l1_l4_bmd': 'Spine L1–L4 BMD',
    'hand_grip_left': 'Grip strength',
    'liver_attenuation': 'Liver attenuation',
    'bt__neutrophils_abs': 'Neutrophils',
    'bt__lymphocytes_abs': 'Lymphocytes',
    'bt__monocytes_abs': 'Monocytes',
    'bt__hemoglobin': 'Hemoglobin',
    'bt__hba1c': 'HbA1c',
    'bt__wbc': 'WBC',
    'bt__rbc': 'RBC',
    'bt__mcv': 'MCV',
    'bt__mchc': 'MCHC',
    'bt__alt_gpt': 'ALT',
    'bt__creatinine': 'Creatinine',
    'bt__glucose': 'Glucose',
    'bt__platelets': 'Platelets',
}


def _clean_pheno_label(name):
    if name in _PHENO_LABELS:
        return _PHENO_LABELS[name]
    return name.replace('bt__', '').replace('_', ' ').title()


def _draw_lollipop_panel(ax, analysis_prefix, sex, n=8):
    """
    Horizontal lollipop plot of the top FDR-significant phenotypes for one sex.
    Reads pre-saved CSV from OUTPUT_DIR. Cohen's d on x-axis, phenotype on y-axis.
    """
    path = os.path.join(OUTPUT_DIR, f'{analysis_prefix}_{sex}_phenotype_results.csv')
    if not os.path.exists(path):
        ax.axis('off')
        return

    df = pd.read_csv(path)
    sig = df[df['Adjusted_P_Value'] < 0.05].head(n).copy()
    if sig.empty:
        ax.text(0.5, 0.5, 'No FDR-significant phenotypes',
                ha='center', va='center', fontsize=9, color='#888888')
        ax.axis('off')
        return

    sig['label'] = sig['Phenotype'].apply(_clean_pheno_label)
    sig = sig.sort_values('Cohens_d', ascending=True).reset_index(drop=True)

    color = PALETTE_SEX['Female'] if sex == 'female' else PALETTE_SEX['Male']

    for i, row in sig.iterrows():
        d = row['Cohens_d']
        ax.plot([0, d], [i, i], color=color, lw=2.2,
                solid_capstyle='round', zorder=2, alpha=0.85)
        ax.scatter(d, i, color=color, s=65, zorder=3, edgecolors='none')

    ax.axvline(0, color='#444444', lw=0.9, linestyle='--', alpha=0.5, zorder=1)
    ax.set_yticks(range(len(sig)))
    ax.set_yticklabels(sig['label'].tolist(), fontsize=9)
    ax.set_xlabel("Cohen's $d$", fontsize=10)
    ax.grid(axis='x', alpha=0.15, linestyle='--', zorder=0)

    gender_label = 'Female' if sex == 'female' else 'Male'
    ax.set_title(gender_label, fontsize=11, color=color, fontweight='bold', pad=5)
    sns.despine(ax=ax)


def create_paper_figure_v2(df_preds, df_final, df_targets, paired_df, analyzable_drugs):
    """High-information paper figure integrating global evidence and exemplar trajectories."""
    print("\n=== Assembling integrated paper figure (v2) ===")

    fig = plt.figure(figsize=(18, 10.5))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.28, 1.28, 1.34], height_ratios=[1.0, 1.25],
                          wspace=0.30, hspace=0.34)

    ax_A = fig.add_subplot(gs[0, 0])
    ax_B = fig.add_subplot(gs[0, 1])
    ax_D = fig.add_subplot(gs[1, 0])
    ax_E = fig.add_subplot(gs[1, 1])
    ax_F = fig.add_subplot(gs[:, 2])

    # A: age prediction performance + calibration inset.
    y = df_preds['age_true'].values
    p = df_preds['age_pred_lejepa'].values
    ok = ~(np.isnan(y) | np.isnan(p))
    y_v, p_v = y[ok], p[ok]
    r_val, _ = pearsonr(y_v, p_v)
    mae = mean_absolute_error(y_v, p_v)
    slope, intercept = np.polyfit(y_v, p_v, 1)

    hb = ax_A.hexbin(y_v, p_v, gridsize=54, cmap='Blues', mincnt=1, linewidths=0.2)
    cb = fig.colorbar(hb, ax=ax_A, shrink=0.77, pad=0.02)
    cb.set_label('Count', fontsize=9)
    cb.ax.tick_params(labelsize=8)
    lims = [min(y_v.min(), p_v.min()), max(y_v.max(), p_v.max())]
    ax_A.plot(lims, lims, 'k--', lw=1.2, alpha=0.65)
    ax_A.text(0.04, 0.95,
              f"$r={r_val:.3f}$\n$MAE={mae:.2f}$ yr\n$slope={slope:.2f}$",
              transform=ax_A.transAxes, va='top', fontsize=9.5)
    ax_A.set_xlabel('Chronological age (years)')
    ax_A.set_ylabel('Predicted age (years)')
    ax_A.set_title('Age prediction performance', fontsize=11)
    sns.despine(ax=ax_A)

    inset = ax_A.inset_axes([0.58, 0.08, 0.38, 0.30])
    bins = np.quantile(y_v, np.linspace(0, 1, 8))
    bins[0] = bins[0] - 1e-6
    idx = np.digitize(y_v, bins, right=True)
    mids, means, ses = [], [], []
    for b in sorted(set(idx)):
        m = idx == b
        if m.sum() < 10:
            continue
        e = p_v[m] - y_v[m]
        mids.append(np.median(y_v[m]))
        means.append(e.mean())
        ses.append(e.std(ddof=1) / np.sqrt(max(len(e), 1)))
    if len(mids) >= 2:
        mids, means, ses = np.array(mids), np.array(means), np.array(ses)
        inset.plot(mids, means, color='#2c3e50', lw=1.4)
        inset.fill_between(mids, means - 1.96 * ses, means + 1.96 * ses,
                           color='#95a5a6', alpha=0.4)
        inset.axhline(0, color='black', lw=0.8, ls='--', alpha=0.6)
        inset.set_title('Calibration\nerror', fontsize=7)
        inset.tick_params(labelsize=7)
        sns.despine(ax=inset)

    # B: quartile KDE + top phenotype quartile trend inset.
    df_q = (df_final.dropna(subset=[MODEL_COL_RTM, RATE_COL_RTM])
                  .drop_duplicates('RegistrationCode', keep='first'))
    for q, color in PALETTE_QUARTILE.items():
        sub = df_q[df_q[MODEL_COL_RTM] == q][RATE_COL_RTM].dropna()
        if len(sub) > 10:
            sns.kdeplot(sub, ax=ax_B, color=color, lw=2.1, label=q)
    ax_B.axvline(0, color='black', ls='--', lw=1.0, alpha=0.55)
    ax_B.set_xlabel('RTM-corrected aging rate (Δ years/year)')
    ax_B.set_ylabel('Density')
    ax_B.set_title('Quartile partition validity', fontsize=11)
    ax_B.legend(frameon=False, fontsize=7.5, title='Quartile', title_fontsize=7.5)
    sns.despine(ax=ax_B)

    rate_forest = _build_forest_table('aging_rate', top_n=12)
    if not rate_forest.empty and 'Phenotype' in rate_forest.columns:
        top_pheno = rate_forest.iloc[0]['Phenotype']
        sex_pick = rate_forest.iloc[0]['sex']
        if top_pheno in df_targets.columns:
            df_gender = df_targets[['RegistrationCode', 'gender']].drop_duplicates()
            df_gender['Gender'] = df_gender['gender'].map({0: 'Female', 1: 'Male'})
            num_cols = [c for c in df_targets.select_dtypes(include=np.number).columns
                        if c not in ['age', 'Age', 'gender', 'index', 'level_0']]
            df_ts = (df_targets.sort_values(['RegistrationCode', 'age'])
                     if 'age' in df_targets.columns else df_targets)
            if 'age' in df_ts.columns and top_pheno in num_cols:
                first_t = df_ts.groupby('RegistrationCode')[num_cols + ['age']].first()
                last_t = df_ts.groupby('RegistrationCode')[num_cols + ['age']].last()
                tdiff = last_t['age'] - first_t['age']
                valid_idx = tdiff[tdiff >= MIN_TIME_GAP_YEARS].index
                pheno_rates = ((last_t.loc[valid_idx, num_cols] - first_t.loc[valid_idx, num_cols])
                               .div(tdiff.loc[valid_idx], axis=0).reset_index())
                df_tmp = (df_q[['RegistrationCode', MODEL_COL_RTM]]
                          .merge(df_gender[['RegistrationCode', 'Gender']], on='RegistrationCode', how='left')
                          .merge(pheno_rates[['RegistrationCode', top_pheno]], on='RegistrationCode', how='left'))
                desired_gender = 'Female' if sex_pick == 'female' else 'Male'
                df_tmp = df_tmp[df_tmp['Gender'] == desired_gender]
                if not df_tmp.empty:
                    means = df_tmp.groupby(MODEL_COL_RTM)[top_pheno].mean().reindex(['Q1', 'Q2', 'Q3', 'Q4'])
                    sems = df_tmp.groupby(MODEL_COL_RTM)[top_pheno].sem().reindex(['Q1', 'Q2', 'Q3', 'Q4'])
                    inset_b = ax_B.inset_axes([0.55, 0.57, 0.42, 0.36])
                    x = np.arange(4)
                    inset_b.plot(x, means.values, marker='o', color=PALETTE_SEX[desired_gender], lw=1.5)
                    inset_b.fill_between(x,
                                         (means - 1.96 * sems).values,
                                         (means + 1.96 * sems).values,
                                         color=PALETTE_SEX[desired_gender], alpha=0.18)
                    inset_b.set_xticks(x)
                    inset_b.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4'], fontsize=7)
                    inset_b.tick_params(labelsize=7)
                    inset_b.set_title(f"{top_pheno.replace('_', ' ').title()}\n({desired_gender})", fontsize=7)
                    sns.despine(ax=inset_b)

    gap_forest = _build_forest_table('baseline_gap', top_n=12)

    # D/E: forest panels.
    _draw_forest_panel(ax_D, gap_forest, 'Baseline-gap phenotype effects (top hits)', show_legend=False)
    _draw_forest_panel(ax_E, rate_forest, 'Aging-rate phenotype effects (top hits)', show_legend=True)

    # F: paired medication box plots (top hits) without bubble-map clutter.
    if paired_df is None or paired_df.empty:
        ax_F.text(0.5, 0.5, 'No paired medication results', ha='center', va='center', fontsize=11)
        ax_F.axis('off')
    else:
        mdf = paired_df.copy().dropna(subset=['adjusted_p_value'])
        sig = mdf.sort_values('adjusted_p_value').head(3)
        visits = df_final[['RegistrationCode', 'Date', 'age_residual']].dropna().copy()
        visits['Date'] = pd.to_datetime(visits['Date'])
        inset_positions = [[0.02, 0.12, 0.31, 0.76], [0.35, 0.12, 0.31, 0.76], [0.68, 0.12, 0.31, 0.76]]
        ax_F.set_title('Paired before/after shifts in age gap (top drugs)', fontsize=10)
        ax_F.axis('off')
        drawn = 0
        for _, row in sig.iterrows():
            if drawn >= len(inset_positions):
                break
            fs = next((g['first_start'] for g in analyzable_drugs if g['drug'] == row['drug']), None)
            if fs is None:
                continue
            dvis = visits[visits['RegistrationCode'].isin(fs.index)].copy()
            dvis['start_date'] = dvis['RegistrationCode'].map(fs)
            dvis['period'] = np.where(dvis['Date'] < dvis['start_date'], 'Before', 'After')
            dvis = dvis[dvis.groupby('RegistrationCode')['period'].transform('nunique') == 2]
            if dvis.empty:
                continue
            dmean = dvis.groupby(['RegistrationCode', 'period'])['age_residual'].mean().unstack().dropna()
            if dmean.empty:
                continue
            iax = ax_F.inset_axes(inset_positions[drawn])
            sns.boxplot(
                data=dvis, x='period', y='age_residual', order=['Before', 'After'],
                palette=PALETTE_PERIOD, showfliers=False, width=0.45, ax=iax,
                boxprops=dict(alpha=0.85, edgecolor='black', linewidth=1.1),
                medianprops=dict(color='black', linewidth=1.6),
                whiskerprops=dict(color='black', linewidth=0.9),
                capprops=dict(color='black', linewidth=0.9)
            )
            # Light paired trajectories keep directional signal while preserving readability.
            for _, pr in dmean.head(80).iterrows():
                iax.plot([0, 1], [pr['Before'], pr['After']], color='#666666', alpha=0.10, lw=0.6, zorder=1)
            iax.axhline(0, color='black', ls='--', lw=0.8, alpha=0.55)
            iax.set_xticks([0, 1])
            iax.set_xticklabels(['before', 'after'], fontsize=8)
            iax.tick_params(labelsize=8)
            if drawn == 0:
                iax.set_ylabel('Age gap (years)', fontsize=9)
            else:
                iax.set_ylabel('')
            iax.set_xlabel('')
            iax.set_title(
                f"{atc3_label(row['drug'])}\n$\\Delta={row['mean_delta']:+.2f}$ yr, adj. $P={row['adjusted_p_value']:.1e}$",
                fontsize=7.6
            )
            sns.despine(ax=iax)
            drawn += 1

        if drawn == 0:
            ax_F.text(0.5, 0.5, 'No drugs with valid paired before/after visits',
                      ha='center', va='center', fontsize=10, color='#666666')

    # Panel labels.
    label_kw = dict(fontsize=16, fontweight='bold', va='top', ha='right', clip_on=False)
    ax_A.text(-0.16, 1.06, 'a', transform=ax_A.transAxes, **label_kw)
    ax_B.text(-0.16, 1.06, 'b', transform=ax_B.transAxes, **label_kw)
    ax_D.text(-0.16, 1.06, 'c', transform=ax_D.transAxes, **label_kw)
    ax_E.text(-0.16, 1.06, 'd', transform=ax_E.transAxes, **label_kw)
    ax_F.text(-0.16, 1.03, 'e', transform=ax_F.transAxes, **label_kw)

    fig.subplots_adjust(top=0.96, bottom=0.06, left=0.05, right=0.98)

    out_base = os.path.join(OUTPUT_DIR, 'paper_figure_v2')
    plt.savefig(out_base + '.pdf', format='pdf', bbox_inches='tight')
    plt.savefig(out_base + '.png', dpi=800, bbox_inches='tight')
    plt.close()
    print(f"  > Saved {out_base}.pdf / .png")


# =============================================================================
# COMPOSITE PAPER FIGURE
# =============================================================================

def create_paper_figure(df_preds, df_final, paired_df, analyzable_drugs):
    """
    Composite paper figure:
      Row A|B : age prediction hexbin | aging-rate quartile KDE
      Row C   : baseline GAP phenotypes — female lollipop | male lollipop
      Row D   : aging RATE phenotypes  — female lollipop | male lollipop
      Row E   : drug before/after paired panels  [only if paired_df is non-empty]

    Phenotype rows read pre-saved CSVs from OUTPUT_DIR; no raw data needed here.
    """
    print("\n=== Assembling composite paper figure ===")

    has_drugs = paired_df is not None and not paired_df.empty
    row_h     = [2.0, 3.8, 3.8] + ([2.8] if has_drugs else [])
    fig       = plt.figure(figsize=(15.0, sum(row_h) + 0.5))
    gs_master = fig.add_gridspec(len(row_h), 1, hspace=0.55, height_ratios=row_h)

    # Row 0: A  B
    gs_AB = gs_master[0].subgridspec(1, 2, wspace=0.40)
    ax_A  = fig.add_subplot(gs_AB[0])
    ax_B  = fig.add_subplot(gs_AB[1])

    # Row 1: C  (gap lollipops — female | male)
    gs_C   = gs_master[1].subgridspec(1, 2, wspace=0.65)
    axes_C = [fig.add_subplot(gs_C[i]) for i in range(2)]

    # Row 2: D  (rate lollipops — female | male)
    gs_D   = gs_master[2].subgridspec(1, 2, wspace=0.65)
    axes_D = [fig.add_subplot(gs_D[i]) for i in range(2)]

    # Row 3 (optional): E  (drug panels)
    axes_E = []
    if has_drugs:
        n_sig_E = min(3, int((paired_df['adjusted_p_value'] < 0.05).sum()))
        if n_sig_E == 2:
            # Center 2 plots: 4-column grid, occupy middle two slots
            gs_E   = gs_master[3].subgridspec(1, 4, wspace=0.44)
            axes_E = [fig.add_subplot(gs_E[1]), fig.add_subplot(gs_E[2])]
        elif n_sig_E == 1:
            # Center 1 plot: 3-column grid, occupy middle slot
            gs_E   = gs_master[3].subgridspec(1, 3, wspace=0.44)
            axes_E = [fig.add_subplot(gs_E[1])]
        else:
            gs_E   = gs_master[3].subgridspec(1, 3, wspace=0.44)
            axes_E = [fig.add_subplot(gs_E[i]) for i in range(max(n_sig_E, 1))]

    # =========================================================
    # PANEL A — age prediction hexbin
    # =========================================================
    y_all = df_preds['age_true'].values
    p_all = df_preds['age_pred_lejepa'].values
    ok    = ~(np.isnan(y_all) | np.isnan(p_all))
    y_v, p_v = y_all[ok], p_all[ok]
    r_val, _ = pearsonr(y_v, p_v)
    mae       = mean_absolute_error(y_v, p_v)
    hb = ax_A.hexbin(y_v, p_v, gridsize=52, cmap='Blues', mincnt=1, linewidths=0.2)
    cb = fig.colorbar(hb, ax=ax_A, shrink=0.78, pad=0.02)
    cb.set_label('Count', fontsize=9); cb.ax.tick_params(labelsize=8)
    lims = [min(y_v.min(), p_v.min()), max(y_v.max(), p_v.max())]
    ax_A.plot(lims, lims, 'k--', lw=1.2, alpha=0.65)
    ax_A.text(0.05, 0.93, f"$r={r_val:.3f}$\n$MAE={mae:.2f}$ yr",
              transform=ax_A.transAxes, va='top', fontsize=10, color='#333333')
    ax_A.set_xlabel('Chronological age (years)')
    ax_A.set_ylabel('Predicted age (years)', labelpad=8)
    sns.despine(ax=ax_A)

    # =========================================================
    # PANEL B — aging-rate quartile KDE
    # =========================================================
    df_q = (df_final.dropna(subset=[MODEL_COL_RTM, RATE_COL_RTM])
                    .drop_duplicates('RegistrationCode', keep='first'))
    for q, color in PALETTE_QUARTILE.items():
        sub = df_q[df_q[MODEL_COL_RTM] == q][RATE_COL_RTM].dropna()
        if len(sub) > 10:
            sns.kdeplot(sub, ax=ax_B, color=color, linewidth=2.2,
                        label=f'{q}  ($n={len(sub):,}$)')
    ax_B.axvline(0, color='black', linestyle='--', lw=1, alpha=0.5)
    ax_B.set_xlabel('Aging rate (Δ years / year)')
    ax_B.set_ylabel('Density')
    ax_B.legend(frameon=False, title='Aging quartile', fontsize=9, title_fontsize=9)
    sns.despine(ax=ax_B)

    # =========================================================
    # PANELS C — baseline GAP lollipops (female | male)
    # =========================================================
    _draw_lollipop_panel(axes_C[0], 'baseline_gap', 'female')
    _draw_lollipop_panel(axes_C[1], 'baseline_gap', 'male')

    # =========================================================
    # PANELS D — aging RATE lollipops (female | male)
    # =========================================================
    _draw_lollipop_panel(axes_D[0], 'aging_rate', 'female')
    _draw_lollipop_panel(axes_D[1], 'aging_rate', 'male')


    # =========================================================
    # PANELS E — drug before/after paired panels
    # =========================================================
    drug_plotted = 0
    if axes_E:
        visits = df_final[['RegistrationCode', 'Date', 'age_residual']].dropna().copy()
        visits['Date'] = pd.to_datetime(visits['Date'])
        sig_rows = paired_df[paired_df['adjusted_p_value'] < 0.05].head(3)
        for _, row in sig_rows.iterrows():
            if drug_plotted >= 3: break
            drug = row['drug']
            fs   = next((g['first_start'] for g in analyzable_drugs if g['drug'] == drug), None)
            if fs is None: continue
            dvis = visits[visits['RegistrationCode'].isin(fs.index)].copy()
            dvis['start_date'] = dvis['RegistrationCode'].map(fs)
            dvis['period']     = np.where(dvis['Date'] < dvis['start_date'], 'Before', 'After')
            dvis = dvis[dvis.groupby('RegistrationCode')['period'].transform('nunique') == 2]
            if dvis.empty: continue
            ax = axes_E[drug_plotted]
            sns.boxplot(data=dvis, x='period', y='age_residual', order=['Before', 'After'],
                        palette=PALETTE_PERIOD, showfliers=False, ax=ax, width=0.40,
                        boxprops=dict(alpha=0.85, edgecolor='black', linewidth=1.3),
                        medianprops=dict(color='black', linewidth=2),
                        whiskerprops=dict(color='black', linewidth=1.1),
                        capprops=dict(color='black', linewidth=1.1))
            for _, pr in (dvis.groupby(['RegistrationCode', 'period'])['age_residual']
                              .mean().unstack().dropna().iterrows()):
                ax.plot([0, 1], [pr['Before'], pr['After']],
                        color='#555555', alpha=0.18, lw=0.7, zorder=1)
            ax.axhline(0, color='#888888', linestyle='--', lw=1, alpha=0.5)
            tc = '#c0392b' if row['adjusted_p_value'] < 0.05 else '#333333'
            ax.set_title(
                f"{atc3_label(drug)}\n$n={row['N_paired']}$, "
                f"$\Delta={row['mean_delta']:+.2f}$ yr,  adj.$P={row['adjusted_p_value']:.1e}$",
                fontsize=9, color=tc, pad=4)
            ax.set_ylabel('Age gap (years)' if drug_plotted == 0 else '')
            ax.set_xlabel(''); ax.set_xticklabels(['Before', 'After'])
            sns.despine(ax=ax)
            drug_plotted += 1
        for ax in axes_E[drug_plotted:]: ax.axis('off')

    plt.tight_layout()

    # =========================================================
    # Panel labels — placed in figure coordinates after tight_layout
    # so every letter sits the same absolute distance above its panel top.
    # =========================================================
    _kw = dict(fontsize=17, fontweight='bold', va='bottom', ha='left',
               clip_on=False, transform=fig.transFigure)
    _dx, _dy = -0.013, 0.004          # offset from panel top-left corner (figure fraction)
    for lbl, ax in [('a', ax_A), ('b', ax_B), ('c', axes_C[0]), ('d', axes_D[0])]:
        pos = ax.get_position()
        fig.text(pos.x0 + _dx, pos.y1 + _dy, lbl, **_kw)
    if axes_E and drug_plotted:
        pos = axes_E[0].get_position()
        fig.text(pos.x0 + _dx, pos.y1 + _dy, 'e', **_kw)
    out_base = os.path.join(OUTPUT_DIR, 'paper_figure')
    plt.savefig(out_base + '.pdf', format='pdf', bbox_inches='tight', pad_inches=0.5)
    plt.savefig(out_base + '.png', dpi=800, bbox_inches='tight', pad_inches=0.5)
    plt.close()
    print(f"  > Saved {out_base}.pdf / .png")


def load_hpp_targets(systems_list, multivisit=False):
    """Load and merge body-system CSVs from the HPP body_systems directory.

    multivisit=False: baseline only (for cross-sectional gap analysis).
    multivisit=True:  concatenate all available visits so the rate analysis
                      can compute first→last change per subject.
    """
    visits = ['baseline', '02_00_visit', '04_00_visit', '06_00_visit'] if multivisit else ['baseline']
    merged = None
    for name in systems_list:
        dfs = []
        for visit in visits:
            path = os.path.join(HPP_BODY_SYSTEMS_PATH, f"{name}_{visit}.csv")
            try:
                df_v = pd.read_csv(path)
                if 'research_stage' not in df_v.columns:
                    df_v['research_stage'] = visit
                df_v['research_stage'] = df_v['research_stage'].astype(str).apply(
                    lambda x: 'baseline' if x == '00_00_visit' else x)
                dfs.append(df_v)
            except FileNotFoundError:
                pass
        if not dfs:
            print(f"  > Warning: no files found for {name} — skipping")
            continue
        df = pd.concat(dfs, ignore_index=True)
        if merged is None:
            merged = df
        else:
            dup_cols = [c for c in df.columns if c not in ['RegistrationCode', 'research_stage'] and c in merged.columns]
            df = df.drop(columns=dup_cols)
            merged = merged.merge(df, on=['RegistrationCode', 'research_stage'], how='outer')
        print(f"  > Loaded {name} ({'multi-visit' if multivisit else 'baseline'}): "
              f"{df.shape[1]-2} phenotype cols, {len(df)} rows")
    if merged is not None:
        age_df = clean_format(pd.read_csv(TARGETS_PATH))[['RegistrationCode', 'research_stage', 'age']]
        merged = merged.merge(age_df, on=['RegistrationCode', 'research_stage'], how='left')
    return merged if merged is not None else pd.DataFrame()


def filter_to_representatives(df, systems_list, representatives_dict):
    """Keep only representative phenotype columns + metadata."""
    meta = ['RegistrationCode', 'research_stage', 'age']
    keep = [c for c in meta if c in df.columns]
    for system in systems_list:
        for col in representatives_dict.get(system, []):
            if col in df.columns and col not in keep:
                keep.append(col)
    return df[keep]


def qc_aging_scores(df, n_sd=QC_N_SD):
    """Flag and remove extreme outliers in age_residual and rate_lejepa_rtm_corrected."""
    print(f"\n=== QC: Outlier Removal (±{n_sd} SD) ===")
    score_cols = ['age_residual', RATE_COL_RTM]
    outlier_mask = pd.Series(False, index=df.index)
    thresholds = {}
    for col in score_cols:
        vals = df[col].dropna()
        mu, sigma = vals.mean(), vals.std()
        lo, hi = mu - n_sd * sigma, mu + n_sd * sigma
        lo3, hi3 = mu - 3 * sigma, mu + 3 * sigma
        n_out_4sd = ((df[col] < lo) | (df[col] > hi)).sum()
        n_out_3sd = ((df[col] < lo3) | (df[col] > hi3)).sum()
        print(f"  {col}: mean={mu:.2f}, std={sigma:.2f}, range=[{vals.min():.2f}, {vals.max():.2f}]")
        print(f"    ±3 SD=[{lo3:.2f}, {hi3:.2f}]  → {n_out_3sd} outliers")
        print(f"    ±4 SD=[{lo:.2f}, {hi:.2f}]    → {n_out_4sd} outliers (removed)")
        outlier_mask |= (df[col].notna() & ((df[col] < lo) | (df[col] > hi)))
        thresholds[col] = (mu, sigma, lo3, hi3, lo, hi)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, col in zip(axes, score_cols):
        vals_all = df[col].dropna()
        mu, sigma, lo3, hi3, lo, hi = thresholds[col]
        vals_all.plot.kde(ax=ax, color='#2471a3', lw=2)
        ax.axvline(lo3, color='#f0a07a', lw=1.4, ls='--', label=f'±3 SD (n={int(((vals_all<lo3)|(vals_all>hi3)).sum())})')
        ax.axvline(hi3, color='#f0a07a', lw=1.4, ls='--')
        ax.axvline(lo,  color='#c0392b', lw=1.4, ls='-',  label=f'±4 SD (n={int(((vals_all<lo)|(vals_all>hi)).sum())})')
        ax.axvline(hi,  color='#c0392b', lw=1.4, ls='-')
        xlabel = 'Biological age gap (years)' if col == 'age_residual' else 'Pace of aging (years/year)'
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Density')
        ax.set_title(col.replace('_', ' '))
        ax.legend(fontsize=9)
        sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "aging_score_qc.pdf"), format='pdf')
    plt.close()

    df_outliers = df[outlier_mask].copy()
    df_clean = df[~outlier_mask].copy()
    print(f"  > QC removed {outlier_mask.sum()} rows total; {len(df_clean)} remain")
    return df_clean, df_outliers


def plot_score_value_ranges(df_final):
    """Violin plots of actual age_gap and pace values by Q1–Q4 quartile."""
    print("\n=== Plotting Aging Score Value Ranges ===")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Panel 1: age_residual by Gap_Quartile
    df_gap = df_final.sort_values('age_true').drop_duplicates('RegistrationCode', keep='first').copy()
    df_gap['Gap_Quartile'] = pd.qcut(df_gap['age_residual'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    ax = axes[0]
    sns.violinplot(data=df_gap, x='Gap_Quartile', y='age_residual',
                   order=['Q1', 'Q2', 'Q3', 'Q4'],
                   palette=PALETTE_QUARTILE, inner='box', ax=ax, linewidth=1.2)
    ax.axhline(0, color='#888888', ls='--', lw=1, alpha=0.6)
    for i, q in enumerate(['Q1', 'Q2', 'Q3', 'Q4']):
        vals = df_gap[df_gap['Gap_Quartile'] == q]['age_residual'].dropna()
        q25, med, q75 = vals.quantile([0.25, 0.5, 0.75])
        ax.text(i, ax.get_ylim()[1] * 0.92, f"med={med:.1f}\nIQR={q75-q25:.1f}",
                ha='center', va='top', fontsize=8, color='#333333')
    ax.set_xlabel('Quartile (biological age gap)')
    ax.set_ylabel('Biological age gap (years)')
    ax.set_xticklabels(['Q1\n(young)', 'Q2', 'Q3', 'Q4\n(old)'])
    sns.despine(ax=ax)

    # Panel 2: rate by bin_rate_lejepa_rtm
    df_rate = df_final.dropna(subset=[RATE_COL_RTM, MODEL_COL_RTM]).drop_duplicates('RegistrationCode', keep='first')
    ax = axes[1]
    sns.violinplot(data=df_rate, x=MODEL_COL_RTM, y=RATE_COL_RTM,
                   order=['Q1', 'Q2', 'Q3', 'Q4'],
                   palette=PALETTE_QUARTILE, inner='box', ax=ax, linewidth=1.2)
    ax.axhline(0, color='#888888', ls='--', lw=1, alpha=0.6)
    for i, q in enumerate(['Q1', 'Q2', 'Q3', 'Q4']):
        vals = df_rate[df_rate[MODEL_COL_RTM] == q][RATE_COL_RTM].dropna()
        q25, med, q75 = vals.quantile([0.25, 0.5, 0.75])
        ax.text(i, ax.get_ylim()[1] * 0.92, f"med={med:.2f}\nIQR={q75-q25:.2f}",
                ha='center', va='top', fontsize=8, color='#333333')
    ax.set_xlabel('Quartile (pace of aging)')
    ax.set_ylabel('Pace of aging (years/year)')
    ax.set_xticklabels(['Q1\n(slow)', 'Q2', 'Q3', 'Q4\n(fast)'])
    sns.despine(ax=ax)

    plt.tight_layout()
    base = os.path.join(OUTPUT_DIR, "aging_score_value_ranges")
    plt.savefig(base + ".pdf", format='pdf')
    plt.savefig(base + ".png", dpi=200)
    plt.close()
    print(f"  > Saved {base}.pdf/.png")


def report_bmd_drug_prevalence(start_events, meds_loader=None):
    """Count cohort prevalence of BMD-altering drug classes and decide action per drug.

    Uses the loader's _atc_filter for precise ATC prefix matching (supports 4- and 5-char codes)
    and _atc_to_label for official ontology labels.

    Returns:
        eids_to_exclude    -- set of RegistrationCodes on rare BMD-decreasing drugs
        common_confounders -- list of ATC codes (decreasing, ≥ threshold) for dedicated analysis
        prevalence_df      -- summary DataFrame
    """
    print("\n=== BMD Drug Prevalence Report ===")

    def _med_names_for(code):
        """Medication names matching ATC prefix, via loader if available."""
        if meds_loader is not None and hasattr(meds_loader, '_atc_filter'):
            result = meds_loader._atc_filter([code])
            return set(result.tolist()) if result is not None else set()
        # Fallback: match on the 4-char atc3 column
        return set(start_events[start_events['atc3'] == code[:4]]['medication'].unique())

    def _label_for(code):
        if meds_loader is not None and hasattr(meds_loader, '_atc_to_label'):
            return meds_loader._atc_to_label.get(code, ATC3_NAMES.get(code, code))
        return ATC3_NAMES.get(code, code)

    rows = []
    med_names_cache = {}
    for code in BMD_INCREASING_DRUGS + BMD_DECREASING_DRUGS:
        category = 'increasing' if code in BMD_INCREASING_DRUGS else 'decreasing'
        med_names = _med_names_for(code)
        med_names_cache[code] = med_names
        n_users = int(start_events[start_events['medication'].isin(med_names)]['RegistrationCode'].nunique()
                      if med_names else 0)
        if category == 'increasing':
            action = 'positive_control'
        elif code in BMD_FORCED_EXCLUDE or n_users < BMD_RARITY_THRESHOLD:
            action = 'exclude'
        else:
            action = 'dedicated_analysis'
        rows.append({'atc': code, 'label': _label_for(code),
                     'n_users': n_users, 'category': category, 'action': action})

    prevalence_df = pd.DataFrame(rows).sort_values(['category', 'n_users'], ascending=[True, False])

    col_w = max(len(r['label']) for r in rows) + 2
    print(f"  {'ATC':<7} {'Label':<{col_w}} {'N users':>8}  {'Category':<12}  {'Action'}")
    print(f"  {'-'*7} {'-'*col_w} {'-'*8}  {'-'*12}  {'-'*20}")
    for _, r in prevalence_df.iterrows():
        print(f"  {r['atc']:<7} {r['label']:<{col_w}} {r['n_users']:>8}  {r['category']:<12}  {r['action']}")

    out_csv = os.path.join(OUTPUT_DIR, 'bmd_drug_prevalence.csv')
    prevalence_df.to_csv(out_csv, index=False)
    print(f"  > Saved {out_csv}")

    exclude_codes = prevalence_df[prevalence_df['action'] == 'exclude']['atc'].tolist()
    exclude_meds = set().union(*[med_names_cache[c] for c in exclude_codes if med_names_cache.get(c)])
    eids_to_exclude = set(
        start_events[start_events['medication'].isin(exclude_meds)]['RegistrationCode'].unique()
    )
    common_confounders = prevalence_df[prevalence_df['action'] == 'dedicated_analysis']['atc'].tolist()

    excluded_drugs = prevalence_df[prevalence_df['action'] == 'exclude'][['atc', 'label', 'n_users']]
    if not excluded_drugs.empty:
        print(f"  > Will exclude {len(eids_to_exclude)} participants using rare BMD-decreasing drugs:")
        for _, r in excluded_drugs.iterrows():
            print(f"      {r['atc']} ({r['label']}): {r['n_users']} users")
    else:
        print("  > No rare BMD-decreasing drug users to exclude (all above threshold).")

    return eids_to_exclude, common_confounders, prevalence_df


BODYCOMP_DRUGS = ['N06A', 'G03F']   # drugs of interest for decomposition

BODYCOMP_COLS = {
    'bmi':                         'BMI',
    'body_comp_total_lean_mass':   'Lean mass',
    'body_comp_total_tissue_mass': 'Total tissue',
    'total_scan_vat_volume':       'VAT',
    'total_scan_sat_volume':       'SAT',
    'femur_neck_mean_bmd':         'Femur-neck BMD',
    'body_total_bmd':              'Total BMD',
    'hand_grip_left':              'Grip strength',
}


def run_drug_bodycomp_decomposition(df_final, df_targets, start_events):
    """
    For significant drugs (SSRIs N06A, HRT G03F):
      1. Compute paired Δ (After − Before) for bio-age gap + body-comp columns.
      2. Age-adjust each Δ by residualising on age at drug initiation.
      3. Report partial correlations: r(Δgap, Δcomponent | age at initiation).
      4. Export CSV + multi-panel figure per drug × sex.

    Rationale: the paired Wilcoxon already controls for between-subject age
    differences, but within the ~2-year window the rate of body-comp change
    depends on baseline age.  Residualising Δ on age-at-initiation isolates
    the drug-attributable component.  Partial correlations then reveal which
    body-comp axis mediates the gap change.
    """
    print("\n=== STEP 4F: Body-Comp Decomposition for SSRI / HRT ===")

    gap_df = (df_final[['RegistrationCode', 'Date', 'age_residual']]
              .dropna().copy())
    gap_df['Date'] = pd.to_datetime(gap_df['Date'])

    targets = df_targets.copy()
    if 'Date' not in targets.columns:
        date_map = (df_final[['RegistrationCode', 'research_stage', 'Date']]
                    .dropna(subset=['Date'])
                    .drop_duplicates(['RegistrationCode', 'research_stage']))
        targets = targets.merge(date_map,
                                on=['RegistrationCode', 'research_stage'], how='left')
    targets['Date'] = pd.to_datetime(targets['Date'], errors='coerce')
    targets = targets.dropna(subset=['RegistrationCode', 'Date'])
    targets = targets.merge(
        gap_df[['RegistrationCode', 'Date', 'age_residual']],
        on=['RegistrationCode', 'Date'], how='left')

    if ('body_comp_total_tissue_mass' in targets.columns and
            'body_comp_total_lean_mass' in targets.columns):
        targets['fat_mass_derived'] = (targets['body_comp_total_tissue_mass']
                                       - targets['body_comp_total_lean_mass'])

    avail_pheno = {k: v for k, v in BODYCOMP_COLS.items() if k in targets.columns}
    if 'fat_mass_derived' in targets.columns:
        avail_pheno['fat_mass_derived'] = 'Fat mass (derived)'
    outcome_cols = ['age_residual'] + list(avail_pheno.keys())

    for drug in BODYCOMP_DRUGS:
        label = ATC3_NAMES.get(drug, drug)
        print(f"\n  -- {label} ({drug}) --")

        first_start = (start_events[start_events['atc3'] == drug]
                       .groupby('RegistrationCode')['Date'].min())
        if len(first_start) < MIN_PAIRED_SUBJECTS:
            print(f"    n={len(first_start)} < {MIN_PAIRED_SUBJECTS}; skipping.")
            continue

        sub = targets[targets['RegistrationCode'].isin(first_start.index)].copy()
        sub['start_date'] = sub['RegistrationCode'].map(first_start)
        sub = sub.dropna(subset=['start_date'])
        sub['period'] = np.where(sub['Date'] < sub['start_date'], 'Before', 'After')

        age_at_init = (sub[sub['period'] == 'After']
                       .sort_values('Date')
                       .drop_duplicates('RegistrationCode', keep='first')
                       [['RegistrationCode', 'age']]
                       .set_index('RegistrationCode')['age'])

        per_subj = (sub.groupby(['RegistrationCode', 'period'])[outcome_cols]
                    .mean().unstack('period'))

        gender_df = (df_targets[['RegistrationCode', 'gender']].drop_duplicates()
                     if 'gender' in df_targets.columns else None)
        sex_map = (gender_df.set_index('RegistrationCode')['gender'].reindex(per_subj.index)
                   if gender_df is not None
                   else pd.Series(np.nan, index=per_subj.index))

        female_ratio = (sex_map == 0).mean()
        if drug == 'G03F' or female_ratio >= 0.90:
            sex_groups = [('Female', 0)]
        elif female_ratio <= 0.10:
            sex_groups = [('Male', 1)]
        else:
            sex_groups = [('Female', 0), ('Male', 1)]

        for sex_label, sex_code in sex_groups:
            idx   = per_subj.index[sex_map == sex_code]
            frame = per_subj.loc[idx]
            ages  = age_at_init.reindex(idx)

            if len(frame) < MIN_PAIRED_SUBJECTS:
                print(f"    {sex_label}: n={len(frame)} < {MIN_PAIRED_SUBJECTS}; skipping.")
                continue

            rows_out, delta_raw, delta_adj = [], {}, {}

            for col in outcome_cols:
                if col not in frame.columns.get_level_values(0):
                    continue
                try:
                    b_vals = frame[(col, 'Before')]
                    a_vals = frame[(col, 'After')]
                except KeyError:
                    continue
                common = b_vals.dropna().index.intersection(a_vals.dropna().index)
                if len(common) < 5:
                    continue

                delta = a_vals.loc[common] - b_vals.loc[common]
                delta_raw[col] = delta

                age_vals = ages.reindex(common).dropna()
                shared   = common.intersection(age_vals.index)
                d_shared = delta.loc[shared]
                a_shared = age_vals.loc[shared]
                if len(shared) >= 5 and a_shared.std() > 1e-6:
                    coef  = np.polyfit(a_shared.values, d_shared.values, 1)
                    d_adj = d_shared - np.polyval(coef, a_shared.values)
                else:
                    d_adj = d_shared.copy()
                delta_adj[col] = d_adj

                def _wtest(b, a):
                    d = a - b
                    return (np.nan if d.std() < 1e-8 or len(d) < 5
                            else wilcoxon(b.values, a.values)[1])

                p_raw  = _wtest(b_vals.loc[common], a_vals.loc[common])
                p_adj  = (np.nan if d_adj.std() < 1e-8
                          else wilcoxon(d_adj.values)[1])
                d_std  = delta.std()
                cohens = delta.mean() / d_std if d_std > 1e-8 else 0.0

                rows_out.append({
                    'drug': drug, 'drug_label': label, 'sex': sex_label,
                    'phenotype': col,
                    'phenotype_label': (avail_pheno.get(col, col)
                                        if col != 'age_residual' else 'Bio-age gap'),
                    'N_paired':          len(common),
                    'mean_before':       b_vals.loc[common].mean(),
                    'mean_after':        a_vals.loc[common].mean(),
                    'mean_delta_raw':    delta.mean(),
                    'cohens_d':          cohens,
                    'p_raw':             p_raw,
                    'mean_delta_ageadj': d_adj.mean(),
                    'p_ageadj':          p_adj,
                })

            if not rows_out:
                continue

            res_df = pd.DataFrame(rows_out)
            for p_col, q_col in [('p_raw', 'q_raw'), ('p_ageadj', 'q_ageadj')]:
                valid = res_df[p_col].notna()
                res_df[q_col] = np.nan
                if valid.sum() > 1:
                    _, res_df.loc[valid, q_col], _, _ = multipletests(
                        res_df.loc[valid, p_col], method='fdr_bh')

            out_csv = os.path.join(OUTPUT_DIR,
                                   f'drug_bodycomp_{drug}_{sex_label.lower()}.csv')
            res_df.to_csv(out_csv, index=False)
            print(f"\n    {sex_label} (n={len(frame)}): {out_csv}")
            print(res_df[['phenotype_label', 'N_paired',
                           'mean_before', 'mean_after',
                           'mean_delta_raw', 'cohens_d', 'q_raw',
                           'mean_delta_ageadj', 'q_ageadj']].to_string(index=False))

            # Partial correlations: Δgap vs each Δcomponent | age at initiation
            pcorr_rows = []
            if 'age_residual' in delta_raw:
                dg = delta_raw['age_residual']
                for col, col_lbl in avail_pheno.items():
                    if col not in delta_raw:
                        continue
                    dc = delta_raw[col]
                    shared = (dg.index.intersection(dc.index)
                              .intersection(ages.dropna().index))
                    if len(shared) < 8:
                        continue

                    def _resid(y, x):
                        c_ = np.polyfit(x, y, 1)
                        return y - np.polyval(c_, x)

                    a_v = ages.reindex(shared).values
                    r, p = pearsonr(_resid(dg.loc[shared].values, a_v),
                                    _resid(dc.loc[shared].values, a_v))
                    pcorr_rows.append({'component': col_lbl,
                                       'partial_r': round(r, 3),
                                       'p': round(p, 4),
                                       'n': len(shared)})

            if pcorr_rows:
                pcorr_df = (pd.DataFrame(pcorr_rows)
                            .sort_values('partial_r', key=abs, ascending=False))
                print(f"\n    Partial r  (Δ bio-age gap vs Δcomponent | age at initiation):")
                print(pcorr_df.to_string(index=False))
                pcorr_df.to_csv(
                    os.path.join(OUTPUT_DIR,
                                 f'drug_bodycomp_pcorr_{drug}_{sex_label.lower()}.csv'),
                    index=False)

            # Figure: row 0 = bar Δraw vs Δage-adj per component
            #         row 1 = scatter Δgap(age-adj) vs Δcomponent(age-adj)
            plot_cols = [c for c in ['bmi', 'body_comp_total_lean_mass',
                                     'fat_mass_derived', 'total_scan_vat_volume',
                                     'total_scan_sat_volume', 'femur_neck_mean_bmd',
                                     'hand_grip_left']
                         if c in delta_raw]
            scat_cols = [c for c in ['body_comp_total_lean_mass', 'fat_mass_derived',
                                     'bmi', 'total_scan_vat_volume']
                         if c in delta_adj and 'age_residual' in delta_adj]
            if not plot_cols:
                continue

            ncols = max(len(plot_cols), len(scat_cols), 3)
            fig, axes = plt.subplots(2, ncols, figsize=(ncols * 2.5, 7))

            def _sig(q):
                if np.isnan(q): return 'ns'
                return '**' if q < 0.01 else ('*' if q < 0.05 else 'ns')

            for j, col in enumerate(plot_cols):
                ax = axes[0, j]
                row_info = res_df[res_df['phenotype'] == col]
                q_r = row_info['q_raw'].values[0]    if len(row_info) else np.nan
                q_a = row_info['q_ageadj'].values[0] if len(row_info) else np.nan
                d_r = delta_raw.get(col, pd.Series(dtype=float))
                d_a = delta_adj.get(col, pd.Series(dtype=float))
                means = [d_r.mean(), d_a.mean() if len(d_a) else np.nan]
                sems  = [d_r.sem(),  d_a.sem()  if len(d_a) else np.nan]
                ax.bar([0, 1], means, yerr=sems, capsize=4,
                       color=['#4878CF', '#D65F5F'], alpha=0.85, width=0.5,
                       ecolor='black', error_kw={'linewidth': 1.2})
                ax.axhline(0, color='black', linewidth=0.8)
                ax.set_xticks([0, 1])
                ax.set_xticklabels(['Raw\nDelta', 'Age-adj\nDelta'], fontsize=7)
                ax.set_title(f"{avail_pheno.get(col, col)}\n"
                             f"{_sig(q_r)} / {_sig(q_a)}", fontsize=8)
                ax.set_ylabel('Mean Δ (after − before)' if j == 0 else '')
                sns.despine(ax=ax)
            for j in range(len(plot_cols), ncols):
                axes[0, j].axis('off')

            for j, col in enumerate(scat_cols[:ncols]):
                ax = axes[1, j]
                dg_a = delta_adj.get('age_residual', pd.Series(dtype=float))
                dc_a = delta_adj.get(col, pd.Series(dtype=float))
                shared = dg_a.index.intersection(dc_a.index)
                if len(shared) < 5:
                    ax.axis('off')
                    continue
                x = dc_a.loc[shared].values
                y = dg_a.loc[shared].values
                ax.scatter(x, y, alpha=0.45, s=22, color='#555555')
                m_fit, b_fit = np.polyfit(x, y, 1)
                xl = np.linspace(x.min(), x.max(), 100)
                ax.plot(xl, m_fit * xl + b_fit, color='#c0392b', linewidth=1.8)
                r_s, p_s = pearsonr(x, y)
                ax.set_xlabel(f'Delta {avail_pheno.get(col, col)} (age-adj)', fontsize=8)
                ax.set_ylabel('Delta bio-age gap (age-adj)' if j == 0 else '', fontsize=8)
                ax.set_title(f'r = {r_s:+.2f}  p = {p_s:.3f}', fontsize=8)
                sns.despine(ax=ax)
            for j in range(len(scat_cols), ncols):
                axes[1, j].axis('off')

            plt.suptitle(
                f"{label} ({sex_label}, n={len(frame)}) — "
                "Body-composition decomposition of Delta bio-age gap",
                fontsize=10, y=1.01)
            plt.tight_layout()
            out_fig = os.path.join(OUTPUT_DIR,
                                   f'drug_bodycomp_{drug}_{sex_label.lower()}.pdf')
            fig.savefig(out_fig, format='pdf', bbox_inches='tight')
            plt.close()
            print(f"    Figure saved: {out_fig}")

    print("\n  Body-comp decomposition complete.")


def main():
    # 1. Base Predictions & Pace
    df_preds = generate_predictions_pipeline()
    df_final = calculate_rates_and_partition(df_preds)

    # 1b. QC: remove extreme outliers, inspect value ranges
    df_final, _ = qc_aging_scores(df_final)
    df_final.to_csv(PARTITIONS_OUT, index=False)
    plot_score_value_ranges(df_final)

    # 2. Quartile rate distribution (new: makes the partitioning tangible)
    plot_quartile_distributions(df_final)

    df_targets = clean_format(pd.read_csv(TARGETS_PATH))

    # Load medications early so BMD exclusions can be applied before all phenotype analyses
    start_events, meds_loader = load_medications()
    eids_to_exclude, common_confounders = set(), []
    if start_events is not None:
        eids_to_exclude, common_confounders, _ = report_bmd_drug_prevalence(start_events, meds_loader)
        if eids_to_exclude:
            n_full = df_final['RegistrationCode'].nunique()
            df_final   = df_final[~df_final['RegistrationCode'].isin(eids_to_exclude)].copy()
            df_targets = df_targets[~df_targets['RegistrationCode'].isin(eids_to_exclude)].copy()
            n_excl = df_final['RegistrationCode'].nunique()
            print(f"  > BMD exclusion applied: {n_full} → {n_excl} participants "
                  f"({n_full - n_excl} excluded on rare BMD-altering drugs)")
            df_final.to_csv(PARTITIONS_OUT.replace('.csv', '_bmd_excl.csv'), index=False)

    # 3. Extreme Phenotypes (Baseline Gap & RTM-Corrected Pace) — existing targets
    df_gap = df_final.sort_values('age_true').drop_duplicates('RegistrationCode', keep='first').copy()
    df_gap['Gap_Quartile'] = pd.qcut(df_gap['age_residual'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    plot_phenotype_extremes(df_gap, 'Gap_Quartile', df_targets, "Baseline Gap Female", sex_filter='Female')
    plot_phenotype_extremes(df_gap, 'Gap_Quartile', df_targets, "Baseline Gap Male", sex_filter='Male')

    df_rate_rtm = df_final.dropna(subset=[MODEL_COL_RTM]).drop_duplicates('RegistrationCode', keep='first')
    if len(df_rate_rtm) > 0:
        plot_phenotype_extremes(df_rate_rtm, MODEL_COL_RTM, df_targets, "Aging Rate Female", sex_filter='Female')
        plot_phenotype_extremes(df_rate_rtm, MODEL_COL_RTM, df_targets, "Aging Rate Male", sex_filter='Male')

    # 3b. DXA/Skeletal phenotype analysis (full bone density + frailty)
    print("\n=== DXA/Skeletal Phenotype Analysis ===")
    df_dexa_bl = load_hpp_targets(DEXA_SYSTEMS, multivisit=False)
    if not df_dexa_bl.empty:
        plot_phenotype_extremes(df_gap, 'Gap_Quartile', df_dexa_bl, "Baseline Gap DXA Female", sex_filter='Female')
        plot_phenotype_extremes(df_gap, 'Gap_Quartile', df_dexa_bl, "Baseline Gap DXA Male",   sex_filter='Male')
    df_dexa_mv = pd.DataFrame()
    if len(df_rate_rtm) > 0:
        df_dexa_mv = load_hpp_targets(DEXA_SYSTEMS, multivisit=True)
        if not df_dexa_mv.empty:
            plot_phenotype_extremes(df_rate_rtm, MODEL_COL_RTM, df_dexa_mv, "Aging Rate DXA Female", sex_filter='Female')
            plot_phenotype_extremes(df_rate_rtm, MODEL_COL_RTM, df_dexa_mv, "Aging Rate DXA Male",   sex_filter='Male')

    # 3c. HPP general phenotype analysis — representative phenotypes only to preserve power
    print("\n=== HPP General Phenotype Analysis ===")
    df_hpp_bl = load_hpp_targets(HPP_SYSTEMS, multivisit=False)
    if not df_hpp_bl.empty:
        df_hpp_bl_rep = filter_to_representatives(df_hpp_bl, HPP_SYSTEMS, HPP_REPRESENTATIVES)
        plot_phenotype_extremes(df_gap, 'Gap_Quartile', df_hpp_bl_rep, "Baseline Gap HPP Female", sex_filter='Female')
        plot_phenotype_extremes(df_gap, 'Gap_Quartile', df_hpp_bl_rep, "Baseline Gap HPP Male",   sex_filter='Male')
    if len(df_rate_rtm) > 0:
        df_hpp_mv = load_hpp_targets(HPP_SYSTEMS, multivisit=True)
        if not df_hpp_mv.empty:
            df_hpp_mv_rep = filter_to_representatives(df_hpp_mv, HPP_SYSTEMS, HPP_REPRESENTATIVES)
            plot_phenotype_extremes(df_rate_rtm, MODEL_COL_RTM, df_hpp_mv_rep, "Aging Rate HPP Female", sex_filter='Female')
            plot_phenotype_extremes(df_rate_rtm, MODEL_COL_RTM, df_hpp_mv_rep, "Aging Rate HPP Male",   sex_filter='Male')

    # 4. Medications (Paired Before/After)
    paired_df, analyzable_drugs = pd.DataFrame(), []
    if start_events is not None:
        analyzable_drugs = _build_analyzable_drugs(start_events)

        # Paired Wilcoxon before/after
        paired_df, sig_drugs = run_paired_medication_analysis(df_final, df_targets, analyzable_drugs)

        # Phenotype changes for significant paired hits (general targets)
        run_significant_drug_phenotype_change_analysis(df_targets, start_events, sig_drugs, df_final)

        # Body-comp decomposition + age-adjustment for SSRI / HRT
        run_drug_bodycomp_decomposition(df_final, df_targets, start_events)

        # 4D. Dedicated BMD drug phenotype analysis on DXA phenotypes:
        #     M05B = positive control; common decreasing confounders (≥ threshold) = dedicated
        bmd_analysis_drugs = list(dict.fromkeys(['M05B'] + common_confounders))
        if bmd_analysis_drugs:
            print("\n=== STEP 4D: Dedicated BMD Drug Phenotype Analysis ===")
            print(f"  Drugs: {bmd_analysis_drugs}")
            run_significant_drug_phenotype_change_analysis(
                df_targets, start_events, bmd_analysis_drugs, df_final,
                out_prefix='bmd_dedicated'
            )

        # Cross-sectional analysis: users vs non-users at baseline
        all_drugs = list(dict.fromkeys(
            [g['drug'] for g in analyzable_drugs] + bmd_analysis_drugs +
            ['M05B', 'G03C', 'G03F', 'A11CC', 'N06AB', 'A02BC', 'N03A']
        ))
        run_crosssectional_drug_analysis(df_final, df_targets, start_events, all_drugs)

    # 5. Composite paper figure
    create_paper_figure(df_preds, df_final, paired_df, analyzable_drugs)
    create_paper_figure_v2(df_preds, df_final, df_targets, paired_df, analyzable_drugs)

    print("\n" + "="*60)
    print("PIPELINE COMPLETE — Output files:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.endswith('.png') or f.endswith('.pdf') or f.endswith('.csv'):
            print(f"  {f:55s} ({os.path.getsize(os.path.join(OUTPUT_DIR, f)) / 1024:.1f} KB)")
    print("="*60)

if __name__ == "__main__":
    main()
