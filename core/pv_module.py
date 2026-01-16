# core/pv_module.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Dict, Any, Optional, Tuple

import pandas as pd
import streamlit as st
import math
from core.climate_data import get_station_monthly
from core.economics import compute_tech_economics

from core.nodes_ontology import node_pv, node_elec_load, node_elec_grid

from core.utils.timebase import build_monthly_kwh_from_df




def render_pv_extra_params(
    df: pd.DataFrame,
    existing_config: Optional[Dict[str, Any]] = None,
    widget_key_prefix: str = "",
) -> Dict[str, Any]:
    """
    UI simple pour les paramètres PV supplémentaires :
    - colonne d'autoconsommation SI elle existe
    - sinon, taux d'autoconsommation estimé [%]

    Retourne un dict du type :
    {
        "col_pv_auto": "NomColonne" ou None,
        "default_selfc_pct": 20 ou autre,
    }
    """
    st.markdown("##### Paramètres PV supplémentaires")
    
    cfg = existing_config.copy() if existing_config else {}
    
    orientation_deg = float(cfg.get("orientation_deg", 0.0))
    inclinaison_deg = float(cfg.get("inclinaison_deg", 30.0))

    col1, col2 = st.columns(2)
    with col1:
        orientation_deg = st.number_input(
            "Orientation du champ PV [°]\n(0° = Sud, -90° = Ouest, 180° ou -180° = Nord, 90° = Est)",
            min_value=0.0,
            max_value=360.0,
            value=orientation_deg,
            step=1.0,
            key=f"{widget_key_prefix}_pv_orientation_deg",
        )
    with col2:
        inclinaison_deg = st.number_input(
            "Inclinaison du champ PV [°]",
            min_value=0.0,
            max_value=90.0,
            value=inclinaison_deg,
            step=1.0,
            key=f"{widget_key_prefix}_pv_inclinaison_deg",
        )


    col_pv_auto = cfg.get("col_pv_auto")
    default_selfc_pct = cfg.get("default_selfc_pct", 20)

    # 1) Checkbox : est-ce qu'il a une colonne d'autocons ?
    has_pv_auto_default = col_pv_auto is not None
    has_pv_auto = st.checkbox(
        "Avez-vous une colonne d'autoconsommation PV ?",
        value=has_pv_auto_default,
        key=f"{widget_key_prefix}_has_pv_auto",
    )

    if has_pv_auto:
        cols_num = df.select_dtypes(include=["number"]).columns.tolist()

        if not cols_num:
            st.warning("Aucune colonne numérique trouvée dans le fichier.")
            col_pv_auto = None
            default_selfc_pct = None
        else:
            if col_pv_auto not in cols_num:
                col_pv_auto = cols_num[0]

            col_pv_auto = st.selectbox(
                "Sélectionnez la colonne d'autoconsommation PV :",
                options=cols_num,
                index=cols_num.index(col_pv_auto) if col_pv_auto in cols_num else 0,
                key=f"{widget_key_prefix}_pv_auto_col",
            )
            # si colonne réelle → plus besoin d'un % par défaut
            default_selfc_pct = None
    else:
        col_pv_auto = None
        # 2) Slider pour le taux d'autoconsommation estimé
        default_selfc_pct = st.slider(
            "Taux d'autoconsommation PV estimé [%]",
            min_value=0,
            max_value=100,
            value=int(default_selfc_pct),
            step=1,
            key=f"{widget_key_prefix}_pv_selfc_pct",
        )

    return {
        "col_pv_auto": col_pv_auto,
        "default_selfc_pct": default_selfc_pct,
        "orientation_deg": orientation_deg,
        "inclinaison_deg": inclinaison_deg,
    }

def extract_pv_standard_inputs(prod: Dict[str, Any], project_meta: Dict[str, Any]) -> tuple[Optional[Dict[str, Any]], list[str]]:
    """
    Prépare toutes les infos nécessaires pour le calcul du profil PV standard.
    Retourne (inputs, missing) :
      - inputs = dict si tout est OK
      - missing = liste de libellés manquants
    """
    missing: list[str] = []

    techno_params = prod.get("parametres", {}) or {}

    p_module_kw = _get_param_val(techno_params, "Puissance du module", 0.0)
    area_module_m2 = _get_param_val(techno_params, "Surface du module", 0.0)
    eta_mod_pct = _get_param_val(techno_params, "Rendement", 0.0)

    pv_cfg = prod.get("pv_flux_config", {}) or {}
    orientation = pv_cfg.get("orientation_deg")
    inclinaison = pv_cfg.get("inclinaison_deg")

    station = project_meta.get("station_meteo") or project_meta.get("station")  # selon ce que tu utilises

    if not station:
        missing.append("Station météo (Phase 1)")
    if orientation is None:
        missing.append("Orientation champ PV [°]")
    if inclinaison is None:
        missing.append("Inclinaison champ PV [°]")
    if p_module_kw <= 0:
        missing.append("Puissance du module (kW)")
    if area_module_m2 <= 0:
        missing.append("Surface du module (m²)")
    if eta_mod_pct <= 0:
        missing.append("Rendement (%)")

    if missing:
        return None, missing

    inputs = {
        "station_name": station,
        "orientation_deg": float(orientation),
        "tilt_deg": float(inclinaison),
        "p_module_kw": float(p_module_kw),
        "area_module_m2": float(area_module_m2),
        "eta_mod_pct": float(eta_mod_pct),
    }
    return inputs, []


def _get_param_val(techno_params: Dict[str, Any], key: str, default: float = 0.0) -> float:
    """
    Récupère une valeur numérique depuis techno_params.
    techno_params vient typiquement de prod["parametres"] dans Phase 1.

    techno_params ressemble à :
    {
        "Capex (CHF/kW)": {"valeur": 1200, "unite": "CHF/kW"},
        "Opex (CHF/an)": {"valeur": 20, "unite": "CHF/an"},
        ...
    }
    """
    raw = techno_params.get(key, {})
    if isinstance(raw, dict):
        val = raw.get("valeur", default)
    else:
        val = raw
    try:
        return float(val)
    except Exception:
        return float(default)

def _get_param_numeric(
    techno_params: Dict[str, Any],
    candidate_keys,
    default: float = 0.0
) -> float:
    """
    Cherche une valeur numérique dans techno_params en testant plusieurs clés possibles.
    techno_params ressemble à :
        { "Capex (CHF/kW)": {"valeur": 1200, "unite": "CHF/kW"}, ... }
    """
    if not techno_params:
        return float(default)

    for key in candidate_keys:
        raw = techno_params.get(key)
        if isinstance(raw, dict):
            val = raw.get("valeur", None)
        else:
            val = raw

        if val is None:
            continue

        try:
            return float(val)
        except Exception:
            continue

    return float(default)


def compute_pv_economics(
    annual_prod_kwh: float,
    installed_kw: float,
    tech_params: Dict[str, Any],
) -> Dict[str, float]:
    """
    Calcule des indicateurs économiques simples pour le PV.

    On accepte plusieurs écritures possibles dans l'Excel :
      - Capex (CHF/kW), Capex [CHF/kW], CAPEX (CHF/kW), ...
      - Capex (CHF), Capex total (CHF), ...
      - Opex (CHF/an), Opex [CHF/an], ...
      - Opex (CHF/kW/an), Opex [CHF/kW/an], ...
      - Durée de vie (ans), Durée de vie [ans], Lifetime (a), ...
    """

    annual_prod_kwh = float(annual_prod_kwh or 0.0)
    installed_kw = float(installed_kw or 0.0)

    if installed_kw <= 0 or annual_prod_kwh <= 0:
        return {
            "capex_total_CHF": 0.0,
            "opex_annual_CHF": 0.0,
            "pv_lcoe_CHF_kWh": 0.0,
            "lifetime_years": 25.0,
            "lcoe_machine_CHF_kWh": 0.0,
        }

    # --- CAPEX ---
    capex_spec_kw = _get_param_numeric(
        tech_params,
        [
            "Capex (CHF/kW)",
            "Capex [CHF/kW]",
            "CAPEX (CHF/kW)",
            "Investissement spécifique (CHF/kW)",
            "Capex",
        ],
        default=0.0,
    )
    capex_total_direct = _get_param_numeric(
        tech_params,
        [
            "Capex (CHF)",
            "Capex total (CHF)",
            "Investissement total (CHF)",
        ],
        default=0.0,
    )

    if capex_total_direct > 0:
        capex_total = capex_total_direct
    else:
        capex_total = capex_spec_kw * installed_kw

    # --- OPEX ---
    opex_global = _get_param_numeric(
        tech_params,
        [
            "Opex (CHF/an)",
            "Opex [CHF/an]",
            "OPEX (CHF/an)",
            "Coûts fixes (CHF/an)",
            "Opex",
        ],
        default=0.0,
    )
    opex_spec_kw_an = _get_param_numeric(
        tech_params,
        [
            "Opex (CHF/kW/an)",
            "Opex [CHF/kW/an]",
            "Opex spécifique (CHF/kW/an)",
        ],
        default=0.0,
    )

    opex_annual = 0.0
    if opex_global > 0:
        opex_annual += opex_global
    if opex_spec_kw_an > 0 and installed_kw > 0:
        opex_annual += opex_spec_kw_an * installed_kw

    # --- Durée de vie ---
    lifetime_years = _get_param_numeric(
        tech_params,
        [
            "Durée de vie (ans)",
            "Durée de vie [ans]",
            "Lifetime (a)",
            "Durée de vie",
        ],
        default=25.0,
    )
    if lifetime_years <= 0:
        lifetime_years = 25.0

    # --- LCOE simplifié (pas d'actualisation) ---
    if annual_prod_kwh > 0:
        lcoe = (capex_total / lifetime_years + opex_annual) / annual_prod_kwh
    else:
        lcoe = 0.0

    return {
        "capex_total_CHF": capex_total,
        "opex_annual_CHF": opex_annual,
        "pv_lcoe_CHF_kWh": lcoe,          # spécifique PV
        "lifetime_years": lifetime_years, # générique
        "lcoe_machine_CHF_kWh": lcoe,     # générique pour l'agrégateur

        # --- Détails (utiles pour Phase 2 / debug) ---
        "capex_spec_CHF_kW": capex_spec_kw,
        "capex_total_direct_CHF": capex_total_direct,
        "opex_global_CHF_an": opex_global,
        "opex_spec_CHF_kW_an": opex_spec_kw_an,
    }

# ================================================================
# Rendement géométrique PV (orientation / inclinaison)
# Basé sur les polynômes fournis par Nikola
# ================================================================

def _poly_val(x: float, coeffs):
    """
    Évalue un polynôme a0 + a1*x + a2*x² + ... pour des coeffs = [a0, a1, ...].
    """
    y = 0.0
    for i, a in enumerate(coeffs):
        y += a * (x ** i)
    return y


# Polynomiales EN FONCTION DE L'INCLINAISON pour des orientations fixes
# y = a2 * tilt² + a1 * tilt + a0
# (on utilise |orientation| pour classer dans ces familles)
_ORIENT_TILT_POLYS = {
    0:   [0.901,   0.0062,   -1e-4],   # 0° (Sud)
    45:  [0.903,   0.0021,   -6e-5],   # ±45° (SE / SO)
    90:  [0.9016, -0.0043,   -2e-7],   # ±90° (E / O)
    135: [0.9004, -0.0094,    4e-5],   # ±135° (NE / NO)
    180: [0.9062, -0.0113,    5e-5],   # ±180° (N)
}

# Polynomiales EN FONCTION DE L'ORIENTATION pour des inclinaisons fixes
# y = a_n * ori^n + ... + a1 * ori + a0
# On stocke les coeffs en ordre [a0, a1, ..., a_n].
_TILT_ORIENT_POLYS = {
    0:  [0.9],  # horizontal = constant 0.9

    30: [0.993,  -2e-12, -4e-5,  -1e-17,  1e-9,   8e-22,  -1e-14],
    # y = -1E-14 x^6 + 8E-22 x^5 + 1E-09 x^4 - 1E-17 x^3 - 4E-05 x^2 - 2E-12 x + 0.993

    45: [0.9651, -2e-12, -4e-5,   2e-17,  1e-9,   1e-21,  -2e-14],
    # y = -2E-14 x^6 + 1E-21 x^5 + 1E-09 x^4 + 2E-17 x^3 - 4E-05 x^2 - 2E-12 x + 0.9651

    60: [0.893,  -2e-12, -4e-5,  -3e-17,  1e-9,   9e-22,  -2e-14],
    # y = -2E-14 x^6 + 9E-22 x^5 + 1E-09 x^4 - 3E-17 x^3 - 4E-05 x^2 - 2E-12 x + 0.893

    90: [0.6501,  4e-14, -2e-5,   5e-18,  3e-10, -9e-23],
    # y = -9E-23 x^5 + 3E-10 x^4 + 5E-18 x^3 - 2E-05 x^2 + 4E-14 x + 0.6501
}


def _closest_key(keys, value):
    """Renvoie la clé de 'keys' la plus proche de 'value'."""
    return min(keys, key=lambda k: abs(k - value))


def compute_pv_geom_efficiency(orientation_deg: float, tilt_deg: float) -> float:
    """
    Calcule un rendement géométrique PV (facteur relatif ~ [0..1])
    à partir de l'orientation et de l'inclinaison, en combinant
    les deux familles de polynômes fournies.

    - orientation_deg : 0° = Sud, 90° = Ouest, 180° = Nord, 270° = Est
    - tilt_deg        : 0° = horizontal, 90° = vertical
    """

    if orientation_deg is None or tilt_deg is None:
        return 1.0  # fallback neutre

    # Normaliser orientation dans [-180, 180] pour gérer Est/Ouest proprement
    ori = float(orientation_deg)
    ori_norm = ((ori + 180.0) % 360.0) - 180.0

    # Clamp de l'inclinaison
    tilt = max(0.0, min(float(tilt_deg), 90.0))

    # ---------- 1) Rendement via polynômes (orientation -> fonction du tilt) ----------
    abs_ori = abs(ori_norm)
    orient_key = _closest_key(_ORIENT_TILT_POLYS.keys(), abs_ori)
    a0, a1, a2 = _ORIENT_TILT_POLYS[orient_key]
    eff_orient_tilt = _poly_val(tilt, [a0, a1, a2])

    # ---------- 2) Rendement via polynômes (inclinaison -> fonction de l'orientation) ----------
    tilt_key = _closest_key(_TILT_ORIENT_POLYS.keys(), tilt)
    coeffs_tilt = _TILT_ORIENT_POLYS[tilt_key]
    eff_tilt_orient = _poly_val(ori_norm, coeffs_tilt)

    # ---------- 3) Combinaison (moyenne) ----------
    eff = 0.5 * (eff_orient_tilt + eff_tilt_orient)

    # Sécurité : on borne dans [0, 1.1]
    eff = max(0.0, min(float(eff), 1.1))

    return eff

# ================================================================
# CALCUL DE L'ÉNERGIE PV PAR m² À PARTIR DE LA MÉTÉO SIA + REND. GÉOMÉTRIQUE
# ================================================================

def compute_pv_monthly_energy_m2(
    station_name: str,
    orientation_deg: float,
    tilt_deg: float,
) -> pd.Series:
    """
    Retourne une série pandas de 12 valeurs (kWh/m² par mois)
    en appliquant :
    - la météo SIA de la station
    - ta règle : horizontal si tilt < 45°, sinon orientation
    - correction par le rendement géométrique calculé
    
    orientation_deg : 0 = Sud, 90 = Ouest, -90 = Est, 180 = Nord
    tilt_deg        : 0 = horizontal, 90 = vertical
    """

    df = get_station_monthly(station_name)

    # 1) Calcul du rendement géométrique
    eta_geom = compute_pv_geom_efficiency(orientation_deg, tilt_deg)

    tilt = float(tilt_deg)
    ori = float(orientation_deg)

    # 2) Détermination du rayonnement de base
    if tilt < 45:
        # ----- CAS HORIZONTAL -----
        G = df["G_horiz_MJm2"]
        ref_eff = 0.90  # ton rendement horizontal
    else:
        # Normalisation orientation dans [-180, +180]
        ori_norm = ((ori + 180) % 360) - 180

        if -45 <= ori_norm <= 45:
            G = df["G_S_MJm2"]
            ref_eff = 0.65  # Sud vertical
        elif 45 < ori_norm <= 135:
            G = df["G_O_MJm2"]
            ref_eff = 0.51  # Ouest vertical
        elif -135 <= ori_norm < -45:
            G = df["G_E_MJm2"]
            ref_eff = 0.51  # Est vertical
        else:
            G = df["G_N_MJm2"]
            ref_eff = 0.27  # Nord vertical

    # 3) Conversion MJ/m² → kWh/m² & correction rendement
    #    E = (G / ref_eff) * eta_geom
    E_kWh = (G / ref_eff) * eta_geom / 3.6

    return E_kWh  # Série pandas de 12 mois

def compute_pv_monthly_energy_per_kw(
    station_name: str,
    orientation_deg: float,
    tilt_deg: float,
    p_module_kw: float,
    area_module_m2: float,
    panel_efficiency: float,
) -> pd.Series:
    """
    Retourne une série pandas de 12 valeurs (kWh/kW par mois).

    Inputs :
        - station_name : station SIA (ex: "Sion")
        - orientation_deg : 0 = Sud, 90 = Ouest, -90 = Est, 180 = Nord
        - tilt_deg : 0 = horizontal, 90 = vertical
        - p_module_kw : puissance nominale d'un module [kW]
        - area_module_m2 : surface d'un module [m²]

    Logique :
        1) On calcule d'abord l'énergie solaire utile par m² :
           E_m2 [kWh/m²]
        2) On passe à l'énergie électrique par kW en suivant ta logique :
           - surface totale pour 1 kW : A_tot = area_module_m2 / p_module_kw
           - rendement module implicite : eta_mod = p_module_kw / area_module_m2
           - E_kWh_per_kW = E_m2 * eta_mod * A_tot

        Mathématiquement, eta_mod * A_tot = 1, donc E_kWh_per_kW = E_m2.
        Mais on garde la formule complète pour être aligné avec ta démarche.
    """

    if p_module_kw <= 0 or area_module_m2 <= 0:
        raise ValueError("p_module_kw et area_module_m2 doivent être > 0.")

    # Étape 1 : énergie sur le plan du module, par m²
    E_m2 = compute_pv_monthly_energy_m2(
        station_name=station_name,
        orientation_deg=orientation_deg,
        tilt_deg=tilt_deg,
    )

    # Étape 2 : chaîne de calcul "physique" que tu as décrite
    nbr_module = 1/p_module_kw
    area_per_kw = area_module_m2 * nbr_module    # m² de modules par kW installé

    E_per_kw = E_m2 * area_per_kw * panel_efficiency /100  # kWh/kW

    # mathématiquement, E_per_kw == E_m2, mais on garde la structure explicite
    return E_per_kw

def aggregate_pv_profile_monthly(
    df: pd.DataFrame,
    prod_profile_col: str,
    time_col: Optional[str] = None,
) -> pd.Series:
    """
    Agrège la production utilisateur en kWh/mois, indexé 1..12.

    Priorité :
      1) time_col (ou auto-détection) -> datetime -> groupby mois   [PATCH 1 via timebase]
      2) colonnes (Annee, Mois)
      3) 12 valeurs sans date -> assumé Jan→Déc                    [PATCH 1 via timebase / fallback final]
    """
    if prod_profile_col not in df.columns:
        raise KeyError(f"Colonne '{prod_profile_col}' introuvable.")

    # 1) PATCH 1 : cas time_col (ou auto-détection) + fallback "12 valeurs"
    monthly = build_monthly_kwh_from_df(df, energy_col=prod_profile_col, time_col=time_col)
    if monthly is not None:
        return monthly.reindex(range(1, 13), fill_value=0.0)

    # 2) Fallback : colonnes "Annee", "Mois"
    if {"Annee", "Mois"}.issubset(df.columns):
        serie = pd.to_numeric(df[prod_profile_col], errors="coerce").fillna(0.0)
        dt = pd.to_datetime(
            df["Annee"].astype(str) + "-" + df["Mois"].astype(str) + "-01",
            errors="coerce"
        )
        mois = dt.dt.month
        monthly = serie.groupby(mois).sum()
        return monthly.reindex(range(1, 13), fill_value=0.0)

    # 3) Dernier recours : 12 valeurs -> Jan→Déc (si timebase n'a pas pu)
    serie = pd.to_numeric(df[prod_profile_col], errors="coerce").fillna(0.0).reset_index(drop=True)
    if len(serie) > 12:
        serie = serie.iloc[:12]
    serie.index = range(1, len(serie) + 1)
    return serie.reindex(range(1, 13), fill_value=0.0)


def compare_pv_measured_vs_theoretical_monthly(
    df_prod: pd.DataFrame,
    prod_profile_col: str,
    station_name: str,
    orientation_deg: float,
    tilt_deg: float,
    p_module_kw: float,
    area_module_m2: float,
    panel_efficiency: float,
    installed_kw: float,
    time_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compare la production mesurée vs la production théorique PV.

    Retourne un DataFrame index mois 1..12 :
        - pv_measured_kWh
        - pv_theoretical_kWh
        - ratio_theoretical_over_measured
    """

    # 1) Mesuré
    monthly_meas = aggregate_pv_profile_monthly(
        df=df_prod,
        prod_profile_col=prod_profile_col,
        time_col=time_col,
    )

    # 2) Théorique par kW
    monthly_per_kw = compute_pv_monthly_energy_per_kw(
        station_name=station_name,
        orientation_deg=orientation_deg,
        tilt_deg=tilt_deg,
        p_module_kw=p_module_kw,
        area_module_m2=area_module_m2,
        panel_efficiency=panel_efficiency,
    )
    monthly_per_kw.index = range(1, 13)

    # 3) Théorique pour la puissance installée
    monthly_th = monthly_per_kw * installed_kw

    # 4) Assemblage
    comp = pd.DataFrame({
        "pv_measured_kWh": monthly_meas,
        "pv_theoretical_kWh": monthly_th,
    })

    comp["ratio_theoretical_over_measured"] = comp.apply(
        lambda r: r["pv_theoretical_kWh"] / r["pv_measured_kWh"]
        if r["pv_measured_kWh"] > 0 else None,
        axis=1
    )

    comp.index.name = "month"
    return comp


def compute_pv_flow_block(
    df: pd.DataFrame,
    prod_profile_col: str,
    prod: Dict[str, Any],
    contexte: Optional[Dict[str, Any]] = None,
    techno_params: Optional[Dict[str, Any]] = None,
) -> pd.Series:
    """
    Calcule un bloc de flux PV (flow_block) pour un producteur donné.
    """

    if df is None or df.empty:
        raise ValueError("Impossible de calculer un bloc PV : df vide.")

    if prod_profile_col not in df.columns:
        raise KeyError(f"Colonne de production PV '{prod_profile_col}' introuvable dans le DataFrame.")

    pv_cfg = prod.get("pv_flux_config", {}) or {}
    col_pv_auto = pv_cfg.get("col_pv_auto")
    default_selfc_pct = pv_cfg.get("default_selfc_pct", 20)

    # --- Production totale PV ---
    prod_pv = float(pd.to_numeric(df[prod_profile_col], errors="coerce").fillna(0).sum())
    if prod_pv <= 0:
        raise ValueError("Production PV totale nulle ou négative, bloc non pertinent.")

    # --- Autoconsommation ---
    if col_pv_auto and col_pv_auto in df.columns:
        pv_auto = float(pd.to_numeric(df[col_pv_auto], errors="coerce").fillna(0).sum())
        has_pv_auto_profile = True
    else:
        pv_auto = prod_pv * (float(default_selfc_pct) / 100.0)
        has_pv_auto_profile = False

    # --- Injection réseau ---
    pv_inj = max(prod_pv - pv_auto, 0.0)

    meta = (contexte or {}).copy()
    taux_autocons_pct = float(pv_auto / prod_pv * 100) if prod_pv > 0 else 0.0

    # --- Partie énergétique (avec ontologie de nœuds) ---
    # TODO plus tard : index PV réel si plusieurs champs PV
    pv_node_id = node_pv(index=1)
    load_node_id = node_elec_load()
    grid_node_id = node_elec_grid()

    flow_block = {
        "name": meta.get("pv_block_label") or f"PV – {meta.get('ouvrage_nom', '')}",
        "type": "pv",
        "meta": meta,
        "nodes": [
            {
                "id": pv_node_id,
                "label": meta.get("pv_label", "PV"),
                "group": "prod_elec",
            },
            {
                "id": load_node_id,
                "label": meta.get(
                    "elec_use_label",
                    "Consommation élec. bâtiment"
                ),
                "group": "use_elec",
            },
            {
                "id": grid_node_id,
                "label": meta.get("grid_inj_label", "Réseau électrique"),
                "group": "reseau",
            },
        ],
        "links": [
            {"source": pv_node_id,   "target": load_node_id, "value": pv_auto},
            {"source": pv_node_id,   "target": grid_node_id, "value": pv_inj},
        ],
        "totals": {
            "pv_prod_kWh": prod_pv,
            "pv_auto_kWh": pv_auto,
            "pv_inj_kWh": pv_inj,
            "pv_selfc_pct": taux_autocons_pct,
            "pv_has_auto_profile": has_pv_auto_profile,
            "production_machine_kWh": prod_pv,
        },
    }

    # --- Partie économique : CAPEX / OPEX / LCOE ---
    techno_params = techno_params or prod.get("parametres", {}) or {}
    installed_kw = float((meta or {}).get("puissance_kw", prod.get("puissance_kw", 0.0)) or 0.0)

    eco = compute_tech_economics(
        annual_prod_kwh=prod_pv,
        installed_kw=installed_kw,
        tech_params=techno_params,
        default_lifetime_years=25.0,  # valeur par défaut, mais sera override si "Durée de vie" est renseigné
    )

    # On met les clés NORMALISÉES attendues par compute_economics_from_flow_blocks
    flow_block["totals"].update(
        {
            "capex_total_CHF": eco["capex_total_CHF"],
            "opex_annual_CHF": eco["opex_annual_CHF"],
            "lifetime_years": eco["lifetime_years"],
            "lcoe_machine_CHF_kWh": eco["lcoe_machine_CHF_kWh"],
            "production_machine_kWh": prod_pv,
            # pour rétrocompatibilité PV :
            "pv_lcoe_CHF_kWh": eco["lcoe_machine_CHF_kWh"],
        }
    )

    # Pour debug : on garde le snapshot des paramètres techno utilisés
    flow_block["meta"]["techno_params_used"] = techno_params

    return flow_block


def build_pv_standard_profile(prod: Dict[str, Any], project_meta: Dict[str, Any]) -> tuple[Optional[Dict[str, Any]], list[str]]:
    """
    Fabrique un profil mensuel standard (théorique) pour un producteur PV.
    Retourne (profil, missing_params).
    """
    inputs, missing = extract_pv_standard_inputs(prod, project_meta)
    if missing:
        return None, missing

    E_per_kw = compute_pv_monthly_energy_per_kw(
        station_name=inputs["station_name"],
        orientation_deg=inputs["orientation_deg"],
        tilt_deg=inputs["tilt_deg"],
        p_module_kw=inputs["p_module_kw"],
        area_module_m2=inputs["area_module_m2"],
        eta_mod_pct=inputs["eta_mod_pct"],
    )

    profil = {
        "producer_label": prod.get("label") or prod.get("techno") or "PV",
        "installed_kw": float(prod.get("puissance_kw", 0.0) or 0.0),
        "monthly_kWh_per_kW": E_per_kw,              # Série pandas (12 mois)
        "annual_kWh_per_kW": float(E_per_kw.sum()),
    }
    return profil, []


def compute_pv_self_consumption_split(
    annual_prod_kwh: float,
    self_consumption_rate: float = 0.20,
    method: str = "fixed_rate",
    annual_load_kwh: Optional[float] = None,
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Retourne (E_self, E_inj, meta).

    method:
      - "fixed_rate": E_self = prod * rate
      - "future_load_based": placeholder pour plus tard (ex: min(prod, load)*factor)
    """
    annual_prod_kwh = float(annual_prod_kwh or 0.0)
    r = float(self_consumption_rate or 0.0)

    if annual_prod_kwh <= 0:
        return 0.0, 0.0, {"method": method, "rate": r}

    if method == "fixed_rate":
        r = max(0.0, min(1.0, r))
        e_self = annual_prod_kwh * r
        e_inj = annual_prod_kwh - e_self
        return e_self, e_inj, {"method": method, "rate": r}

    # placeholder pour plus tard
    # ex: basé sur charge annuelle ou profil horaire
    e_self = annual_prod_kwh * r
    e_inj = annual_prod_kwh - e_self
    return e_self, e_inj, {"method": "future_load_based_stub", "rate": r, "annual_load_kwh": annual_load_kwh}

def compute_ru_pv_added_weighted(
    p_kw: float,
    inclination_deg: Optional[float] = None,
    altitude_m: Optional[float] = None,
    parking: bool = False,
) -> Dict[str, Any]:
    """
    RU PV (OEneR annexe 2.1) – installations AJOUTÉES / ISOLÉES uniquement.
    Pondération par tranches, barèmes valables À PARTIR DU 01.04.2025.

    Taux (ajouté/isolé):
      - 0–30 kW   : 360 CHF/kW
      - 30–100 kW : 300 CHF/kW
      - >100 kW   : 250 CHF/kW

    Contribution de base: 0 CHF (>= 01.04.2025)

    Bonus:
      - inclinaison >= 75° : +200 CHF/kW
      - altitude >= 1500 m : +250 CHF/kW
      - parking            : +250 CHF/kW
    """

    p_kw = float(p_kw or 0.0)
    if p_kw < 2:
        return {
            "ru_total_chf": 0.0,
            "ru_breakdown": [],
            "ru_bonus_chf_kw": 0.0,
            "ru_bonus_total_chf": 0.0,
            "assumptions": {"period": ">= 01.04.2025", "type": "added_or_isolated", "weighted": True},
        }

    # --- taux ajoutés/isolés (>= 01.04.2025) ---
    rate_t1, rate_t2, rate_t3 = 360.0, 300.0, 250.0

    # --- bonus CHF/kW ---
    bonus_chf_kw = 0.0
    if inclination_deg is not None and float(inclination_deg) <= 75.0:
        bonus_chf_kw += 200.0
    if altitude_m is not None and float(altitude_m) >= 1500.0:
        bonus_chf_kw += 250.0
    if parking:
        bonus_chf_kw += 250.0

    breakdown = []

    # tranche 0–30
    t1_kw = min(p_kw, 30.0)
    t1_rate = rate_t1 + bonus_chf_kw
    t1_chf = t1_kw * t1_rate
    breakdown.append({"from_kw": 0.0, "to_kw": 30.0, "kw": t1_kw, "rate_chf_kw": t1_rate, "chf": t1_chf})

    # tranche 30–100
    if p_kw > 30.0:
        t2_kw = min(p_kw - 30.0, 70.0)
        t2_rate = rate_t2 + bonus_chf_kw
        t2_chf = t2_kw * t2_rate
        breakdown.append({"from_kw": 30.0, "to_kw": 100.0, "kw": t2_kw, "rate_chf_kw": t2_rate, "chf": t2_chf})

    # tranche >100
    if p_kw > 100.0:
        t3_kw = p_kw - 100.0
        t3_rate = rate_t3 + bonus_chf_kw
        t3_chf = t3_kw * t3_rate
        breakdown.append({"from_kw": 100.0, "to_kw": None, "kw": t3_kw, "rate_chf_kw": t3_rate, "chf": t3_chf})

    ru_total_chf = sum(x["chf"] for x in breakdown)

    return {
        "ru_total_chf": ru_total_chf,
        "ru_breakdown": breakdown,
        "ru_bonus_chf_kw": bonus_chf_kw,
        "ru_bonus_total_chf": bonus_chf_kw * p_kw,
        "assumptions": {
            "period": ">= 01.04.2025",
            "type": "added_or_isolated",
            "weighted": True,
            "base_contribution_chf": 0.0,
        },
    }

def compute_pv_cashflow_series(
    capex_total_chf: float,
    opex_annual_chf: float,
    e_self_kwh: float,
    e_inj_kwh: float,
    price_buy_chf_kwh: float = 0.25,
    price_sell_chf_kwh: float = 0.07,
    years: int = 25,
    ru_upfront_chf: float = 0.0,   # si tu veux intégrer la RU Pronovo
) -> dict:
    capex_total_chf = float(capex_total_chf or 0.0)
    opex_annual_chf = float(opex_annual_chf or 0.0)

    gain_self = float(e_self_kwh or 0.0) * float(price_buy_chf_kwh or 0.0)
    gain_inj  = float(e_inj_kwh or 0.0)  * float(price_sell_chf_kwh or 0.0)
    cashflow_year = gain_self + gain_inj - opex_annual_chf

    years_list = list(range(0, years + 1))
    annual = [0.0] * (years + 1)
    cumulative = [0.0] * (years + 1)

    # Année 0: investissement + RU upfront (si tu la mets au départ)
    annual[0] = -capex_total_chf + ru_upfront_chf
    cumulative[0] = annual[0]

    for y in range(1, years + 1):
        annual[y] = cashflow_year
        cumulative[y] = cumulative[y-1] + annual[y]

    payback_year = next((y for y in range(0, years + 1) if cumulative[y] >= 0), None)

    return {
        "years": years_list,
        "annual_cashflow_chf": annual,
        "cumulative_cashflow_chf": cumulative,
        "gain_self_chf": gain_self,
        "gain_inj_chf": gain_inj,
        "cashflow_year_chf": cashflow_year,
        "payback_year": payback_year,
    }

def compute_simulated_rooftop_pv_per_building(project):
    """
    Prend le projet et retourne les résultats PV simulés par bâtiment (structure identique à ton bloc actuel)
    """
    results = {"pv_proposed_by_batiment": {}}  # <- exactement comme tu fais déjà
    # 🔁 Ici, colle tout ton bloc EXISTANT de simulation PV (avec for bi, bat in enumerate... etc)
    # À la fin, retourne juste ce que tu veux stocker dans le projet :
    return results["pv_proposed_by_batiment"]

def reconstruct_pv_to_load_proportional_bounded(
    pv_ts,
    load_ts,
    selfc_pct_input,
):
    """
    Reconstruit PV->Load physiquement cohérent :
    - bornage instantané : pv_to_load(t) <= min(pv(t), load(t))
    - cible annuelle : selfc_pct_input * sum(PV)
    - saturation si cible > max physique
    """

    # --- robust fraction 0..1 ---
    pct_raw = 0.0 if selfc_pct_input in (None, "") else float(selfc_pct_input)
    selfc_frac = pct_raw if pct_raw <= 1.0 else pct_raw / 100.0

    pv = pv_ts.astype(float)
    load = load_ts.astype(float)

    # --- physical capacity ---
    capacity = pv.combine(load, min)

    pv_sum = float(pv.sum())
    capacity_sum = float(capacity.sum())

    target_kWh = selfc_frac * pv_sum
    effective_target = min(target_kWh, capacity_sum)

    if capacity_sum <= 0.0 or effective_target <= 0.0:
        pv_to_load = pv * 0.0
    else:
        pv_to_load = capacity * (effective_target / capacity_sum)

    # --- final safety ---
    pv_to_load = pv_to_load.clip(lower=0.0)
    pv_to_load = pv_to_load.where(pv_to_load <= pv, pv)
    pv_to_load = pv_to_load.where(pv_to_load <= load, load)

    meta = {
        "selfc_pct_input": selfc_frac,
        "selfc_pct_real": 0.0 if pv_sum <= 0 else pv_to_load.sum() / pv_sum,
        "max_selfc_pct": 0.0 if pv_sum <= 0 else capacity_sum / pv_sum,
        "saturated": target_kWh > capacity_sum,
    }

    return pv_to_load, meta
