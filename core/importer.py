# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 09:47:49 2025

@author: vujic
"""

# core/importer.py
import io
from typing import Literal
import pandas as pd


def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    """
    Lit un fichier uploadé (csv, xlsx, txt) et renvoie un DataFrame.
    À compléter / adapter selon tes besoins.
    """
    filename = uploaded_file.name.lower()

    if filename.endswith(".csv"):
        return pd.read_csv(uploaded_file, sep=None, engine="python")
    elif filename.endswith(".xlsx") or filename.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    elif filename.endswith(".txt"):
        # essai auto du séparateur
        content = uploaded_file.read().decode("utf-8")
        return pd.read_csv(io.StringIO(content), sep=None, engine="python")
    else:
        raise ValueError("Format de fichier non supporté pour le moment.")


def build_project_from_dataframe(
    df: pd.DataFrame,
    params: dict,
    categories_ouvrages: list[dict] | None = None,
    col_batiment: str | None = None,
    col_type_ouvrage: str | None = None,
    col_puissance: str | None = None,
):
    """
    Convertit le DataFrame + éventuel mapping de colonnes en structure Projet.

    Deux modes possibles :
    1) Mode "ancien" : on fournit col_batiment, col_type_ouvrage, col_puissance
       -> on construit batiments / ouvrages à partir du DF (comme avant).
    2) Mode "nouveau" : on ne fournit que df, params, categories_ouvrages
       -> on construit une structure simple basée sur les catégories d’ouvrage.

    Ça te permet de présenter la structure en Phase 2
    même si les calculs détaillés ne sont pas encore implémentés.
    """

    project_dict: dict = {
        "nom": params.get("nom_projet", "Projet sans nom"),
        "params": params,
        "results": {},  # sera rempli plus tard par run_calculations
    }

    # ------------------------------------------------------------------
    # MODE 1 : ancien comportement avec mapping global
    # ------------------------------------------------------------------
    if col_batiment is not None and col_type_ouvrage is not None:
        batiments = sorted(df[col_batiment].unique())
        ouvrages = []

        for _, row in df.iterrows():
            ouvrages.append(
                {
                    "batiment": row[col_batiment],
                    "type_ouvrage": row[col_type_ouvrage],
                    "puissance_kw": float(row[col_puissance])
                    if col_puissance
                    else None,
                }
            )

        project_dict["batiments"] = [{"nom": b} for b in batiments]
        project_dict["ouvrages"] = ouvrages

    # ------------------------------------------------------------------
    # MODE 2 : nouveau mode basé sur categories_ouvrages
    # ------------------------------------------------------------------
    else:
        # Un seul bâtiment "logique" basé sur les infos générales
        nom_bat = params.get("batiment_nom") or "Bâtiment principal"

        project_dict["batiments"] = [
            {
                "nom": nom_bat,
                "adresse": params.get("batiment_adresse", ""),
                "npa_ville": params.get("batiment_npa_ville", ""),
            }
        ]

        ouvrages = []
        if categories_ouvrages:
            for cat in categories_ouvrages:
                ouvrages.append(
                    {
                        "batiment": nom_bat,
                        "type_ouvrage": cat.get("type"),
                        "nom_ouvrage": cat.get("nom"),
                        "sheet_name": cat.get("sheet_name"),
                        "sre_m2": cat.get("sre"),
                        "surface_enveloppe_m2": cat.get("surface_enveloppe"),
                        "time_col": cat.get("time_col"),
                        "prod_cols": cat.get("prod_cols", []),
                        "cons_cols": cat.get("cons_cols", []),
                    }
                )

        project_dict["ouvrages"] = ouvrages
        project_dict["categories_ouvrages"] = categories_ouvrages or []

        # petit aperçu des données (optionnel)
        try:
            project_dict["raw_data_preview"] = df.head(100).to_dict(orient="list")
        except Exception:
            project_dict["raw_data_preview"] = {}

    return project_dict


