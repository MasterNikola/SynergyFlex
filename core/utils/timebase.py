# -*- coding: utf-8 -*-
"""
Created on Mon Jan  5 15:35:43 2026

@author: infor
"""

# core/utils/timebase.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Optional, List, Tuple
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any


# ----------------------------------------------------------------------
# Parsing / time utilities
# ----------------------------------------------------------------------

def _try_parse_datetime_col(df: pd.DataFrame, col: str) -> Optional[pd.Series]:
    """Essaye de parser une colonne en datetime. Retourne Series[datetime] ou None."""
    try:
        dt = pd.to_datetime(df[col], errors="coerce")
        if dt.notna().sum() >= max(2, int(0.5 * len(dt))):  # au moins 2 points + raisonnable
            return dt
    except Exception:
        return None
    return None


def find_time_col(df: pd.DataFrame, explicit: Optional[str] = None) -> Optional[str]:
    """
    Trouve une colonne temps.
    - priorité : explicit (si fourni)
    - sinon : noms usuels
    - sinon : première colonne parsable en datetime
    """
    if df is None or df.empty:
        return None

    if explicit and explicit in df.columns:
        return explicit

    preferred = ["time", "date", "datetime", "timestamp", "date_periode", "Date_periode"]
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    for key in preferred:
        real = lower_map.get(key.strip().lower())
        if real and _try_parse_datetime_col(df, real) is not None:
            return real

    for c in df.columns:
        if _try_parse_datetime_col(df, c) is not None:
            return c

    return None


def infer_dt_hours_from_index(index) -> float:
    """
    Déduit le pas de temps "typique" en heures à partir d'un index datetime.
    - robustesse : médiane des deltas (pas seulement index[1]-index[0])
    - fallback : 1h si impossible
    """
    try:
        if not isinstance(index, pd.DatetimeIndex):
            return 1.0
        if len(index) < 2:
            return 1.0
        deltas_h = index.to_series().diff().dropna().dt.total_seconds() / 3600.0
        deltas_h = deltas_h[deltas_h > 0]
        if deltas_h.empty:
            return 1.0
        return float(deltas_h.median())
    except Exception:
        return 1.0


def dt_hours_series(index: pd.DatetimeIndex) -> pd.Series:
    """
    Retourne une Series alignée sur l'index : dt_hours par pas.
    Utile si le pas est variable (mensuel).
    """
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 2:
        return pd.Series([1.0] * len(index), index=index)

    d = index.to_series().diff().dt.total_seconds() / 3600.0
    # premier pas : backfill
    d = d.bfill().fillna(1.0)
    d = d.clip(lower=1e-9)
    return d


# ----------------------------------------------------------------------
# Monthly aggregation helper (used today by calculations.py & pv_module.py)
# ----------------------------------------------------------------------

def build_monthly_kwh_from_df(
    df: pd.DataFrame,
    energy_col: str,
    time_col: Optional[str] = None,
) -> Optional[pd.Series]:
    """
    Construit une série mensuelle (12 mois) à partir d'un DataFrame:
    - energy_col: colonne énergie (kWh)
    - time_col: colonne temps (optionnelle). Si None -> auto-détection.

    Retourne une Series indexée 1..12 (mois) ou None si impossible.
    """
    if df is None or df.empty or energy_col not in df.columns:
        return None

    s = pd.to_numeric(df[energy_col], errors="coerce")
    if s.notna().sum() == 0:
        return None

    # Temps
    real_time_col = find_time_col(df, explicit=time_col)
    if real_time_col is None:
        # fallback : si on a déjà 12 valeurs sans dates (cas PV mesuré mensuel)
        ser = s.dropna().reset_index(drop=True)
        if len(ser) >= 12:
            ser = ser.iloc[:12]
            ser.index = range(1, 13)
            return ser.reindex(range(1, 13), fill_value=0.0)
        return None

    dt = _try_parse_datetime_col(df, real_time_col)
    if dt is None:
        return None

    mask = dt.notna() & s.notna()
    if not mask.any():
        return None

    grouped = s[mask].groupby(dt[mask].dt.month).sum()
    grouped = grouped.reindex(range(1, 13)).fillna(0.0)
    grouped.index = list(range(1, 13))
    return grouped


def extract_time_info(df: pd.DataFrame, time_col: Optional[str]) -> Optional[dict]:
    """
    Petit helper JSON-friendly pour debug : start/end/dt_median.
    """
    if df is None or df.empty:
        return None
    real = find_time_col(df, explicit=time_col)
    if not real:
        return None
    dt = _try_parse_datetime_col(df, real)
    if dt is None:
        return None
    idx = pd.DatetimeIndex(dt.dropna()).sort_values()
    if len(idx) == 0:
        return None
    return {
        "time_col_used": str(real),
        "start": idx[0].isoformat(),
        "end": idx[-1].isoformat(),
        "dt_hours_median": float(infer_dt_hours_from_index(idx)),
        "n_points_time": int(len(idx)),
    }

def build_timeseries_kwh(
    df: pd.DataFrame,
    energy_col: str,
    time_col: Optional[str] = None,
) -> Optional[pd.Series]:
    """
    Retourne une série kWh/step indexée datetime (pas forcément régulier).
    - auto-détecte time_col si None
    - nettoie NaN -> 0
    """
    if df is None or df.empty or energy_col not in df.columns:
        return None

    tcol = find_time_col(df, explicit=time_col)
    if tcol is None:
        return None

    dt = _try_parse_datetime_col(df, tcol)
    if dt is None:
        return None

    v = pd.to_numeric(df[energy_col], errors="coerce").fillna(0.0)
    mask = dt.notna()
    if not mask.any():
        return None

    s = pd.Series(v[mask].values, index=pd.DatetimeIndex(dt[mask]))
    s = s.sort_index()
    s = s[~s.index.duplicated(keep="first")]
    return s


def _upsample_energy_equal_split(coarse: pd.Series, target_index: pd.DatetimeIndex) -> pd.Series:
    """
    Upsampling déterministe : si on a une énergie par pas plus grossier,
    on la répartit uniformément sur les pas fins inclus dans l’intervalle.
    """
    if coarse is None or coarse.empty:
        return pd.Series(0.0, index=target_index)

    coarse = coarse.sort_index()
    out = pd.Series(0.0, index=target_index)

    # on suppose que chaque valeur coarse vaut pour l’intervalle [t_i, t_{i+1})
    idx = coarse.index
    if len(idx) == 1:
        # cas dégénéré : tout sur tous les points
        out[:] = float(coarse.iloc[0]) / max(len(out), 1)
        return out

    for i in range(len(idx) - 1):
        t0, t1 = idx[i], idx[i + 1]
        mask = (target_index >= t0) & (target_index < t1)
        n = int(mask.sum())
        if n > 0:
            out.loc[target_index[mask]] += float(coarse.iloc[i]) / n

    # dernier point : on ignore (ou on propage 0) -> choix conservatif
    return out


def align_energy_series_to_index(
    s: Optional[pd.Series],
    target_index: pd.DatetimeIndex,
) -> pd.Series:
    """
    Aligne une série kWh/step sur un index cible.
    - si déjà sur l’index : reindex + fill 0
    - si plus fin : on agrège (groupby exact timestamps) + reindex
    - si plus grossier : on split uniformément (déterministe)
    """
    if s is None or len(target_index) == 0:
        return pd.Series(0.0, index=target_index)

    s = s.sort_index()
    # Cas simple : mêmes timestamps
    if s.index.equals(target_index):
        return s.reindex(target_index).fillna(0.0)

    # Si s a une fréquence plus fine : on “snap” sur target par somme des valeurs tombant sur même timestamp
    # (cas rare dans ton usage, mais safe)
    if len(s.index) > len(target_index):
        s2 = s.groupby(s.index).sum()
        return s2.reindex(target_index).fillna(0.0)

    # Sinon : on suppose plus grossier -> upsample split
    return _upsample_energy_equal_split(s, target_index)


def pick_finest_index(series_list: list[pd.Series]) -> Optional[pd.DatetimeIndex]:
    """Renvoie l’index ayant le dt médian le plus petit (plus fin)."""
    best = None
    best_dt = None
    for s in series_list:
        if s is None or s.empty or not isinstance(s.index, pd.DatetimeIndex):
            continue
        dt = infer_dt_hours_from_index(s.index)
        if best is None or dt < (best_dt or 1e99):
            best = s.index
            best_dt = dt
    return best


def common_sim_window_intersection(index_list: list[pd.DatetimeIndex]) -> Optional[pd.DatetimeIndex]:
    """
    Fenêtre commune stricte = intersection temporelle des index (sur min/max),
    puis on garde l’index le plus fin tronqué à cette fenêtre.
    """
    idxs = [i for i in index_list if isinstance(i, pd.DatetimeIndex) and len(i) > 0]
    if not idxs:
        return None

    start = max(i.min() for i in idxs)
    end = min(i.max() for i in idxs)

    finest = min(idxs, key=lambda i: infer_dt_hours_from_index(i))
    finest_cut = finest[(finest >= start) & (finest <= end)]
    if len(finest_cut) == 0:
        return None
    return finest_cut