"""Regression-to-the-mean (RTM) detrending and quartile binning.

A single source of truth for the biological-age-gap definition used across
the section. Mirrors the canonical implementation that previously lived
inline in ``ukbb_aging_pace_v2v3.py:load_and_detrend`` (poly-2 RTM on
chronological age).

The gap is orthogonal-by-construction to the polynomial of ``age_col``:

    gap = pred − E[pred | age_col]    where E[·] is poly-degree fit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


__all__ = ['detrend_gap', 'bin_quartiles', 'rtm_correct_pace']


def detrend_gap(
    df: pd.DataFrame,
    age_col: str = 'age_true',
    pred_col: str = 'age_pred_lejepa',
    degree: int = 2,
) -> pd.Series:
    """Return the RTM-detrended biological-age gap.

    Fits ``pred ~ poly(age_col, degree)`` on the full input, then returns
    ``pred − fitted``. The resulting series has the same index as ``df``.
    """
    sub = df[[age_col, pred_col]].dropna()
    X = PolynomialFeatures(degree=degree).fit_transform(sub[[age_col]].values)
    expected = LinearRegression().fit(X, sub[pred_col].values).predict(X)
    gap = pd.Series(np.nan, index=df.index, name='bioage_gap')
    gap.loc[sub.index] = sub[pred_col].values - expected
    return gap


def bin_quartiles(series: pd.Series, labels=('Q1', 'Q2', 'Q3', 'Q4')) -> pd.Series:
    """Stable quartile bin (rank-based to avoid duplicate-edge errors)."""
    out = pd.Series(pd.Categorical([None] * len(series), categories=list(labels)),
                    index=series.index, name='gap_quartile')
    mask = series.notna()
    binned = pd.qcut(series.loc[mask].rank(method='first'),
                     q=len(labels), labels=list(labels))
    out.loc[mask] = binned.astype(str)
    out = out.astype(pd.CategoricalDtype(categories=list(labels), ordered=True))
    return out


def rtm_correct_pace(pace: pd.Series, baseline_gap: pd.Series) -> pd.Series:
    """Linear RTM correction for aging pace (regress out baseline-gap dependence)."""
    df = pd.concat([pace.rename('pace'), baseline_gap.rename('bl')], axis=1).dropna()
    if df.empty:
        return pd.Series(np.nan, index=pace.index, name='aging_pace_rtm')
    rtm = LinearRegression().fit(df[['bl']].values, df['pace'].values)
    out = pd.Series(np.nan, index=pace.index, name='aging_pace_rtm')
    out.loc[df.index] = df['pace'].values - rtm.predict(df[['bl']].values)
    return out


if __name__ == '__main__':
    # Smoke check: verify against the original inline RTM in ukbb_aging_pace_v2v3.py
    import os
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
    from downstream.bioage.paths import PRED_CSV
    df = pd.read_csv(PRED_CSV).head(5000)

    # Inline reference implementation (verbatim from ukbb_aging_pace_v2v3.py)
    X = PolynomialFeatures(degree=2).fit_transform(df[['age_true']])
    expected = LinearRegression().fit(X, df['age_pred_lejepa']).predict(X)
    ref_gap = df['age_pred_lejepa'] - expected

    new_gap = detrend_gap(df)
    diff = (new_gap.values - ref_gap.values)
    print(f'max |diff| = {np.abs(diff).max():.3e}   (expected < 1e-10)')
    assert np.abs(diff).max() < 1e-9, 'RTM helper drifted from inline reference'
    print('rtm.detrend_gap matches reference: OK')
    q = bin_quartiles(new_gap)
    print('quartile sizes:', q.value_counts().sort_index().to_dict())
