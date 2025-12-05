# core/climate_data.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd

# Ton fichier est dans /data
CLIMATE_XLS_PATH = Path(__file__).resolve().parent.parent / "data" / "2028_2015_f_kompaktdaten.xls"
SHEET_NAME = "val. mois"

# Noms exacts des stations (doivent correspondre à ce qui apparaît dans la colonne 0 de val. mois)
STATION_NAMES = [
    "Adelboden", "Aigle", "Altdorf", "Basel-Binningen", "Bern-Liebefeld",
    "Buchs-Aarau", "Chur", "Davos", "Disentis", "Engelberg", "Genève-Cointrin",
    "Glarus", "Grand-St-Bernard", "Güttingen", "Interlaken",
    "La Chaux-de-Fonds", "La Frétaz", "Locarno-Monti", "Lugano", "Luzern",
    "Magadino", "Montana", "Neuchâtel", "Payerne", "Piotta", "Pully",
    "Robbia", "Rünenberg", "Samedan", "San Bernardino", "St. Gallen",
    "Schaffhausen", "Scuol", "Sion", "Ulrichen", "Vaduz", "Wynau",
    "Zermatt", "Zürich-Kloten", "Zürich-MeteoSchweiz"
]


# ---------------------------------------------------------------------------
# 1) Charger la feuille brute
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _load_raw() -> pd.DataFrame:
    if not CLIMATE_XLS_PATH.exists():
        raise FileNotFoundError(f"Fichier climatique introuvable : {CLIMATE_XLS_PATH}")

    df = pd.read_excel(CLIMATE_XLS_PATH, sheet_name=SHEET_NAME, header=None)
    return df


# ---------------------------------------------------------------------------
# 2) Repérer les blocs par station
# ---------------------------------------------------------------------------
def _find_station_blocks(df: pd.DataFrame) -> Dict[str, Tuple[int, int]]:
    """
    Repère les blocs de lignes correspondant à chaque station dans val. mois.
    Retourne : { "Adelboden": (start_row, end_row), ... }
    """
    name_rows = []

    for i, value in enumerate(df.iloc[:, 0].astype(str)):
        if value.strip() in STATION_NAMES:
            name_rows.append((i, value.strip()))

    blocks = {}
    for idx, (row, name) in enumerate(name_rows):
        start = row
        end = name_rows[idx + 1][0] if idx + 1 < len(name_rows) else len(df)
        blocks[name] = (start, end)

    return blocks


# ---------------------------------------------------------------------------
# 3) Extraire le tableau mensuel pour un bloc de station
# ---------------------------------------------------------------------------
def _extract_monthly(df_block: pd.DataFrame) -> pd.DataFrame:
    """
    Extrait un tableau mensuel complet pour une station :
        Mois | T_ext | G_horiz_MJm2 | G_E_MJm2 | G_S_MJm2 | G_O_MJm2 | G_N_MJm2

    Basé sur la structure du fichier :
    - une ligne contient les mois + Année
    - 1 ligne T_ext
    - 1 ligne 'Rayonnement global, somme'
    - les 5 lignes suivantes correspondent à: horizontal, E, S, O, N (dans cet ordre)
    """

    # 1) ligne des mois
    header_row_idx = None
    annee_col_idx = None
    for i in range(len(df_block)):
        row = df_block.iloc[i, :]
        for j, cell in enumerate(row):
            if str(cell).strip() == "Année":
                header_row_idx = i
                annee_col_idx = j
                break
        if header_row_idx is not None:
            break

    if header_row_idx is None:
        raise ValueError("Impossible de trouver 'Année'.")

    header_row = df_block.iloc[header_row_idx, :]
    month_cols = [
        j for j, cell in enumerate(header_row)
        if pd.notna(cell) and str(cell).strip() != "Année"
    ]
    months = [str(header_row[j]).strip() for j in month_cols]

    # 2) ligne T_ext
    temp_row_idx = None
    for i in range(header_row_idx + 1, len(df_block)):
        if "Température de l’air" in str(df_block.iloc[i, 0]):
            temp_row_idx = i
            break

    if temp_row_idx is None:
        raise ValueError("Impossible de trouver la ligne température.")

    # 3) ligne Rayonnement global, somme
    ray_row_idx = None
    for i in range(temp_row_idx + 1, len(df_block)):
        if "Rayonnement global" in str(df_block.iloc[i, 0]):
            ray_row_idx = i
            break

    if ray_row_idx is None:
        raise ValueError("Impossible de trouver 'Rayonnement global, somme'.")

    # 4) Lignes des orientations fixes
    orient_rows = {
        "G_horiz_MJm2": ray_row_idx - 2,
        "G_E_MJm2":     ray_row_idx - 1,
        "G_S_MJm2":     ray_row_idx + 0,
        "G_O_MJm2":     ray_row_idx + 1,
        "G_N_MJm2":     ray_row_idx + 2,
    }

    # Extraction valeurs
    data = {
        "Mois": months,
        "T_ext": pd.to_numeric(
            [df_block.iloc[temp_row_idx, j] for j in month_cols],
            errors="coerce"
        )
    }

    for colname, row_idx in orient_rows.items():
        data[colname] = pd.to_numeric(
            [df_block.iloc[row_idx, j] for j in month_cols],
            errors="coerce"
        )

    return pd.DataFrame(data)





# ---------------------------------------------------------------------------
# 4) Fonction publique : récupérer les données mensuelles d'une station
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_climate_monthly() -> Dict[str, pd.DataFrame]:
    df_raw = _load_raw()
    blocks = _find_station_blocks(df_raw)

    out = {}
    for station_name, (start, end) in blocks.items():
        df_block = df_raw.iloc[start:end, :].reset_index(drop=True)
        try:
            out[station_name] = _extract_monthly(df_block)
        except Exception as e:
            print(f"[climate_data] Erreur extraction pour {station_name}: {e}")

    return out


def get_climate_station_options() -> List[Tuple[str, str]]:
    """Pour Phase 1 : renvoie [(label, station_name), ...]."""
    stations = sorted(load_climate_monthly().keys())
    return [(name, name) for name in stations]


def get_station_monthly(station_name: str) -> pd.DataFrame:
    monthly = load_climate_monthly()
    if station_name not in monthly:
        raise KeyError(f"Station inconnue : {station_name}")
    return monthly[station_name].copy()


# ---------------------------------------------------------------------------
# 5) MODE TEST — exécutable en dehors de Streamlit
# ---------------------------------------------------------------------------
# if __name__ == "__main__":
#     print("=== TEST CLIMAT SynergyFlex ===")
#     stations = list(load_climate_monthly().keys())

#     print("\nStations disponibles :")
#     for s in stations:
#         print(" -", s)

#     choice = input("\nEntrez le nom EXACT de la station à afficher : ")

#     if choice not in stations:
#         print("❌ Station inconnue ! Vérifie la casse et l'orthographe.")
#     else:
#         df_month = get_station_monthly(choice)
#         print(f"\n=== Données mensuelles pour {choice} ===\n")
#         print(df_month)
