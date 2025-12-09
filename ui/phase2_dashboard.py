# -*- coding: utf-8 -*-
"""
Phase 2 - Présentation et analyse du projet SynergyFlex
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px  # ✅ nécessaire pour show_economics
from core.pv_module import (
    compute_pv_monthly_energy_per_kw,
    _get_param_val,
)



# =====================================================================
# FONCTION PRINCIPALE
# =====================================================================

def render_phase2():
    project = st.session_state.get("project")
    if project is None:
        st.warning("Aucun projet chargé. Retourne d'abord à la Phase 1 pour préparer les données.")
        return

    _inject_css()

    st.title("Phase 2 – Analyse et visualisation")
    st.markdown("## Vue d’ensemble du projet")

    page = st.sidebar.radio(
        "Sections",
        [
            "Vue d’ensemble",
            "Diagramme de Sankey",
            "📊 Économie",          # ✅ virgule ajoutée ici
            "Comparaison standard",
            "Analyse conso / prod",
            "Variantes",
        ],
    )

    if page == "Vue d’ensemble":
        show_overview(project)
    elif page == "Diagramme de Sankey":
        show_sankey(project)
    elif page == "📊 Économie":
        show_economics(project)
    elif page == "Comparaison standard":
        show_comparison(project)
    elif page == "Analyse conso / prod":
        show_phases_analysis(project)
    elif page == "Variantes":
        show_variantes(project)

    if st.sidebar.button("🔙 Revenir à la Phase 1 (modifier les données)"):
        st.session_state["validated"] = False
        try:
            st.rerun()
        except AttributeError:
            st.experimental_rerun()


# =====================================================================
# CSS
# =====================================================================

def _inject_css():
    st.markdown(
        """
        <style>
        .sf-card {
            background-color: #ffffff;
            border-radius: 0.7rem;
            padding: 1rem 1.2rem;
            border: 1px solid rgba(0,0,0,0.07);
            box-shadow: 0 4px 12px rgba(15, 23, 42, 0.05);
            margin-bottom: 0.8rem;
        }
        .sf-card-title {
            font-weight: 600;
            font-size: 0.95rem;
            margin-bottom: 0.15rem;
        }
        .sf-card-subtitle {
            font-size: 0.8rem;
            color: #6b7280;
            margin-bottom: 0.4rem;
        }
        .sf-card-body {
            font-size: 0.8rem;
            color: #374151;
        }
        .sf-section-title {
            font-weight: 600;
            font-size: 1.1rem;
            margin-top: 1.5rem;
            margin-bottom: 0.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =====================================================================
# VUE D’ENSEMBLE
# =====================================================================

def show_overview(project: dict):
    """Vue d’ensemble : résumé + bâtiments + ouvrages + graphiques."""

    batiments = project.get("batiments", [])
    params = project.get("params", {})

    # On récupère les catégories d’ouvrage depuis le projet,
    # sinon depuis la session (plus robuste).
    categories = project.get("categories_ouvrages") or st.session_state.get("categories_ouvrages", [])

    # nom du bâtiment pour affichage
    if batiments:
        nom_bat = batiments[0].get("nom", "Bâtiment principal")
    else:
        nom_bat = params.get("batiment_nom", "Bâtiment principal")

    # --------- Résumé du projet ---------
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Nom du projet", params.get("nom_projet", "—"))
    with col2:
        st.metric("Nombre de bâtiments", max(1, len(batiments)))
    with col3:
        st.metric("Nombre d’ouvrages", len(categories))

    # --------- Bâtiments ---------
    st.markdown("<div class='sf-section-title'>Bâtiments</div>", unsafe_allow_html=True)
    if batiments:
        st.dataframe(pd.DataFrame(batiments), use_container_width=True)
    else:
        df_bat = pd.DataFrame(
            [
                {
                    "nom": nom_bat,
                    "adresse": params.get("batiment_adresse", ""),
                    "npa_ville": params.get("batiment_npa_ville", ""),
                }
            ]
        )
        st.dataframe(df_bat, use_container_width=True)

    # --------- Ouvrages (tableau) ---------
    st.markdown("<div class='sf-section-title'>Ouvrages</div>", unsafe_allow_html=True)
    if categories:
        rows = []
        for cat in categories:
            rows.append(
                {
                    "batiment": nom_bat,
                    "type_ouvrage": cat.get("type"),
                    "nom_ouvrage": cat.get("nom"),
                    "sheet_name": cat.get("sheet_name"),
                    "sre_m2": cat.get("sre"),
                    "surface_enveloppe_m2": cat.get("surface_enveloppe"),
                    "time_col": cat.get("time_col"),
                    "conso_elec_col": cat.get("conso_elec_col"),
                    "conso_th_col": cat.get("conso_th_col"),
                }
            )
        df_ouvrages = pd.DataFrame(rows)
        st.dataframe(df_ouvrages, use_container_width=True)
    else:
        st.info("Aucun ouvrage défini dans le projet.")
        return

    # --------- Graphiques par ouvrage ---------
    st.markdown("<div class='sf-section-title'>Graphiques par ouvrage</div>", unsafe_allow_html=True)

    sheets_dict = st.session_state.get("excel_sheets")
    fallback_df = None
    if not sheets_dict:
        fallback_df = _get_fallback_dataframe_from_project(project)
        
    for idx, cat in enumerate(categories):
        type_ouv = cat.get("type", "Ouvrage")
        nom_ouv = cat.get("nom") or f"Ouvrage {idx+1}"
        sheet_name = cat.get("sheet_name")
        sre = cat.get("sre")
        surf_env = cat.get("surface_enveloppe")
        time_col = cat.get("time_col")
        conso_elec_col = cat.get("conso_elec_col")
        conso_th_col = cat.get("conso_th_col")

        # ---------- Carte texte ----------
        st.markdown(
            f"""
            <div class="sf-card">
              <div class="sf-card-title">{type_ouv} – {nom_ouv}</div>
              <div class="sf-card-subtitle">{nom_bat}</div>
              <div class="sf-card-body">
                <b>SRE :</b> {sre if sre is not None else "—"} m² &nbsp; | &nbsp;
                <b>Enveloppe :</b> {surf_env if surf_env is not None else "—"} m² &nbsp; | &nbsp;
                <b>Feuille Excel :</b> {sheet_name or "n/a"}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ---------- DataFrame correspondant ----------
        df_sheet = None
        if sheets_dict and isinstance(sheets_dict, dict):
            if sheet_name and sheet_name in sheets_dict:
                df_sheet = sheets_dict[sheet_name]
            else:
                first_sheet_name = list(sheets_dict.keys())[0]
                df_sheet = sheets_dict[first_sheet_name]
        elif fallback_df is not None:
            df_sheet = fallback_df

        if df_sheet is None:
            st.info("Données brutes introuvables pour cet ouvrage.")
            st.markdown("---")
            continue

        if not time_col or time_col not in df_sheet.columns:
            st.info("Colonne de temps non définie ou introuvable pour cet ouvrage.")
            st.markmarkdown("---")
            continue

        conso_elec_valid = conso_elec_col if conso_elec_col in df_sheet.columns else None
        conso_th_valid = conso_th_col if conso_th_col in df_sheet.columns else None

        if not conso_elec_valid and not conso_th_valid:
            st.info("Aucune colonne de consommation électrique / thermique valide pour cet ouvrage.")
            st.markdown("---")
            continue

        # ---------- Graphique area ----------
        x_raw = df_sheet[time_col]

        if time_col.lower().startswith("mois") and "Annee" in df_sheet.columns:
            x = pd.to_datetime(
                df_sheet["Annee"].astype(int).astype(str) + "-" +
                df_sheet["Mois"].astype(int).astype(str) + "-01",
                errors="coerce"
            )
        else:
            x = pd.to_datetime(x_raw, errors="ignore")

        fig = go.Figure()

        if conso_elec_valid:
            y = pd.to_numeric(df_sheet[conso_elec_valid], errors="coerce")
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    name=f"Conso élec – {conso_elec_valid}",
                    fill="tozeroy",
                    line=dict(width=1.5),
                )
            )

        if conso_th_valid:
            y = pd.to_numeric(df_sheet[conso_th_valid], errors="coerce")
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    name=f"Conso th – {conso_th_valid}",
                    fill="tozeroy",
                    line=dict(width=1.5),
                )
            )

        fig.update_layout(
            height=320,
            margin=dict(l=40, r=20, t=10, b=40),
            xaxis_title="Temps",
            yaxis_title="Puissance / Énergie",
            hovermode="x unified",
        )

        st.plotly_chart(fig, use_container_width=True)
        st.markdown("---")


# =====================================================================
# SANKey helper
# =====================================================================

def _build_sankey_from_flow_block(block: dict):
    """
    Construit une figure Sankey Plotly à partir d'un flow_block standardisé.
    """
    nodes = block.get("nodes", [])
    links = block.get("links", [])

    if not nodes or not links:
        return None

    # --- mapping id -> index ---
    id_to_index = {node["id"]: i for i, node in enumerate(nodes)}
    labels = [node.get("label", node["id"]) for node in nodes]

    rgb_by_group = {
        "prod_elec":  (255, 193,   7),
        "use_elec":   ( 33, 150, 243),
        "reseau":     (244,  67,  54),
        "stock_elec": (123,  31, 162),
        "prod_th":    (255, 112,  67),
        "demande":    ( 76, 175,  80),
        "final":      ( 96, 125, 139),
    }
    default_rgb = (158, 158, 158)

    def rgba(rgb, alpha):
        r, g, b = rgb
        return f"rgba({r},{g},{b},{alpha})"

    node_colors = []
    for node in nodes:
        group = node.get("group")
        rgb = rgb_by_group.get(group, default_rgb)
        node_colors.append(rgba(rgb, 0.9))
    
    from collections import defaultdict

    # --- Positionnement des nœuds pour limiter les croisements ---
    # On place les groupes sur des "couches" horizontales :
    # prod_elec / prod_th -> gauche
    # reseau               -> milieu gauche
    # demande              -> milieu droite
    # final                -> droite
    group_layer = {
        "prod_elec": 0,
        "prod_th":   0,
        "reseau":    1,
        "demande":   2,
        "final":     3,
    }

    layer_to_indices = defaultdict(list)
    for i, node in enumerate(nodes):
        g = node.get("group")
        layer = group_layer.get(g, 1)  # défaut = reseau
        layer_to_indices[layer].append(i)

    x = [0.0] * len(nodes)
    y = [0.0] * len(nodes)

    for layer, idxs in layer_to_indices.items():
        # position x de la couche (0 -> gauche, 1 -> droite)
        x_base = 0.05 + 0.25 * layer
        # on espace les nœuds verticalement dans la couche
        step = 1.0 / (len(idxs) + 1)
        for rank, idx in enumerate(idxs):
            x[idx] = x_base
            y[idx] = step * (rank + 1)

    
    sources, targets, values, link_colors = [], [], [], []
    for link in links:
        src_id = link.get("source")
        tgt_id = link.get("target")
        val = float(link.get("value", 0) or 0)

        if val <= 0:
            continue
        if src_id not in id_to_index or tgt_id not in id_to_index:
            continue

        s_idx = id_to_index[src_id]
        t_idx = id_to_index[tgt_id]

        sources.append(s_idx)
        targets.append(t_idx)
        values.append(val)

        src_group = next((n.get("group") for n in nodes if n["id"] == src_id), None)
        rgb = rgb_by_group.get(src_group, default_rgb)
        link_colors.append(rgba(rgb, 0.5))

    if not values:
        return None

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                valueformat=".0f",
                valuesuffix=" kWh",
                node=dict(
                    pad=35,
                    thickness=30,
                    line=dict(width=1.5, color="rgba(0,0,0,0.4)"),
                    label=labels,
                    color=node_colors,
                    align="center",
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                    color=link_colors,
                    hovertemplate="%{value:.0f} kWh<extra></extra>",
                ),
            )
        ]
    )

    title = block.get("name") or f"Flux {block.get('type', '')}".strip()
    fig.update_layout(
        title_text=title,
        font=dict(size=14, color="rgba(0,0,0,0.85)"),
        margin=dict(l=80, r=80, t=40, b=40),
        height=500,
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    return fig


def show_sankey(project: dict):
    st.markdown(
        "<div class='sf-section-title'>Diagramme de Sankey – Producteurs / Bilan</div>",
        unsafe_allow_html=True,
    )

    results = project.get("results", {})
    flows = results.get("flows", {})
    bat_flows = flows.get("batiments", {})

    if not bat_flows:
        st.info("Aucun flux n’a été calculé. Retourne en Phase 1 et relance les calculs.")
        return

    options = []
    for bat_id, blocks in bat_flows.items():
        for i, block in enumerate(blocks):
            meta = block.get("meta", {})
            bat_nom = meta.get("batiment_nom", f"Bâtiment {bat_id}")
            ouv_nom = meta.get("ouvrage_nom", "Ouvrage")
            bloc_nom = block.get("name", f"Bloc {i+1}")
            label = f"{bat_nom} – {ouv_nom} – {bloc_nom} ({block.get('type', 'n/a')})"
            options.append((label, block))

    if not options:
        st.info("Aucun flow_block disponible pour construire un Sankey.")
        return

    labels = [lbl for lbl, _ in options]
    mapping = {lbl: blk for lbl, blk in options}

    selected_label = st.selectbox("Sélectionne un bloc de flux :", labels)
    selected_block = mapping[selected_label]

    fig = _build_sankey_from_flow_block(selected_block)
    if fig is None:
        st.info("Impossible de construire le Sankey pour ce bloc (pas de liens valides).")
        return

    st.plotly_chart(fig, use_container_width=True)

    totals = selected_block.get("totals", {})
    if totals:
        st.markdown("#### Bilan énergétique du bloc sélectionné")
        df_tot = pd.DataFrame([totals]).T
        df_tot.columns = ["Valeur"]
        st.dataframe(df_tot, use_container_width=True)


# =====================================================================
# 📊 ÉCONOMIE
# =====================================================================

def show_economics(project: dict):
    st.header("📊 Analyse économique")

    econ_by_bat = project.get("results", {}).get("economics_by_batiment", {})

    if not econ_by_bat:
        st.info("Aucune donnée économique disponible (vérifie run_calculations).")
        return

    rows = []
    detail_rows = []
    batiments = project.get("batiments", [])

    for bat_id, econ in econ_by_bat.items():
        # Compat : si l'ancien format (flat) existe, on le supporte
        if isinstance(econ, dict) and "by_type" in econ:
            totals = econ
            by_type = econ.get("by_type", {}) or {}
        else:
            totals = econ
            by_type = {}

        nom_bat = f"Bâtiment {bat_id}"
        if isinstance(bat_id, int) and bat_id < len(batiments):
            nom_bat = batiments[bat_id].get("nom", nom_bat)

        capex = float(totals.get("capex_total_CHF", 0.0) or 0.0)
        opex = float(totals.get("opex_annual_CHF", 0.0) or 0.0)
        prod = float(totals.get("production_totale_kWh", 0.0) or 0.0)
        lcoe = float(totals.get("lcoe_global_CHF_kWh", 0.0) or 0.0)

        rows.append({
            "Bâtiment": nom_bat,
            "CAPEX total [CHF]": round(capex, 1),
            "OPEX annuel [CHF/an]": round(opex, 1),
            "Production totale [kWh]": round(prod, 1),
            "LCOE global [CHF/kWh]": round(lcoe, 4),
        })

        # Détail par type de machine (pv, pac, boiler, ...)
        for mtype, stats in by_type.items():
            detail_rows.append({
                "Bâtiment": nom_bat,
                "Type de machine": mtype,
                "CAPEX [CHF]": round(stats.get("capex_total_CHF", 0.0) or 0.0, 1),
                "OPEX annuel [CHF/an]": round(stats.get("opex_annual_CHF", 0.0) or 0.0, 1),
                "Production [kWh/an]": round(stats.get("production_machine_kWh", 0.0) or 0.0, 1),
                "LCOE type [CHF/kWh]": round(stats.get("lcoe_machine_CHF_kWh", 0.0) or 0.0, 4),
            })

    df = pd.DataFrame(rows)
    st.subheader("Vue globale par bâtiment")
    st.dataframe(df, use_container_width=True)

    if not df.empty:
        st.subheader("CAPEX par bâtiment")
        fig_capex = px.bar(df, x="Bâtiment", y="CAPEX total [CHF]", text_auto=True)
        st.plotly_chart(fig_capex, use_container_width=True)

        st.subheader("LCOE global par bâtiment")
        fig_lcoe = px.bar(df, x="Bâtiment", y="LCOE global [CHF/kWh]", text_auto=True)
        st.plotly_chart(fig_lcoe, use_container_width=True)

    # ----- Détail par type de machine -----
    if detail_rows:
        st.markdown("---")
        st.subheader("Détail par type de machine")

        df_detail = pd.DataFrame(detail_rows)
        st.dataframe(df_detail, use_container_width=True)

        # Camembert CAPEX par techno pour un bâtiment choisi
        bat_list = df["Bâtiment"].tolist()
        selected_bat = st.selectbox(
            "Bâtiment pour détail CAPEX par technologie",
            options=bat_list,
        )

        df_bat = df_detail[df_detail["Bâtiment"] == selected_bat]
        if not df_bat.empty:
            fig_pie = px.pie(
                df_bat,
                names="Type de machine",
                values="CAPEX [CHF]",
                title=f"Répartition CAPEX par technologie – {selected_bat}",
            )
            st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("Pas encore de détail par type de machine (by_type vide).")



# =====================================================================
# AUTRES SECTIONS (placeholders)
# =====================================================================

# =====================================================================
# 🔍 COMPARAISON PV : MESURÉ vs THÉORIQUE (STANDARD)
# =====================================================================

def _iter_pv_blocks_with_comparison(project: dict):
    """
    Génère (label, flow_block) pour tous les blocs PV qui ont une
    comparaison mensuelle mesuré vs théorique.
    """
    results = project.get("results", {}) or {}
    flows = results.get("flows", {}) or {}
    bat_flows = flows.get("batiments", {}) or {}

    for bat_id, blocks in bat_flows.items():
        for fb in blocks:
            if fb.get("type") != "pv":
                continue

            profiles = fb.get("profiles") or {}
            if "pv_monthly_comparison" not in profiles:
                continue

            meta = fb.get("meta", {}) or {}
            bat_nom = meta.get("batiment_nom", f"Bâtiment {bat_id}")
            ouv_nom = meta.get("ouvrage_nom", "")
            pv_label = meta.get("pv_label", "PV")

            label = f"{bat_nom} – {ouv_nom} – {pv_label}"
            yield label, fb


def render_pv_comparison_standard(project: dict):
    """
    Vue 'Comparaison standard' pour les producteurs PV :
    - Tableau mensuel mesuré vs théorique
    - Graphique barres
    - PR global
    """
    pv_blocks = list(_iter_pv_blocks_with_comparison(project))

    if not pv_blocks:
        st.info(
            "Aucune comparaison PV disponible. Vérifie que :\n"
            "- la station météo est définie en Phase 1,\n"
            "- les paramètres de module PV (Puissance, Surface, Rendement) sont renseignés,\n"
            "- le producteur est bien de type PV avec profil de production."
        )
        return

    labels = [lbl for lbl, _ in pv_blocks]
    st.markdown("### Comparaison production PV – Mesurée vs théorique")

    selected_label = st.selectbox(
        "Sélectionne un producteur PV",
        options=labels,
    )

    # Récupérer le bloc sélectionné
    fb_map = {lbl: blk for lbl, blk in pv_blocks}
    fb = fb_map[selected_label]

    profiles = fb.get("profiles", {})
    comp_records = profiles.get("pv_monthly_comparison", [])

    if not comp_records:
        st.warning("Pas de données de comparaison pour ce producteur PV.")
        return

    comp_df = pd.DataFrame(comp_records)

    # Sécuriser la colonne mois
    if "month" in comp_df.columns:
        comp_df = comp_df.sort_values("month")
        comp_df["month_label"] = comp_df["month"].astype(int).astype(str)
    else:
        comp_df["month"] = range(1, len(comp_df) + 1)
        comp_df["month_label"] = comp_df["month"].astype(int).astype(str)

    # ----- PR global & totaux -----
    totals = fb.get("totals", {}) or {}
    pv_meas_total = totals.get("pv_measured_kWh_total", None)
    pv_theo_total = totals.get("pv_theoretical_kWh_total", None)
    pv_PR_global = totals.get("pv_PR_global", None)

    col1, col2, col3 = st.columns(3)
    with col1:
        if pv_meas_total is not None:
            st.metric(
                "Prod. mesurée annuelle [kWh]",
                f"{pv_meas_total:,.0f}".replace(",", " ")
            )
        else:
            st.metric("Prod. mesurée annuelle [kWh]", "—")
    with col2:
        if pv_theo_total is not None:
            st.metric(
                "Prod. théorique annuelle [kWh]",
                f"{pv_theo_total:,.0f}".replace(",", " ")
            )
        else:
            st.metric("Prod. théorique annuelle [kWh]", "—")
    with col3:
        if pv_PR_global is not None:
            st.metric("PR global [-]", f"{pv_PR_global:.2f}")
        else:
            st.metric("PR global [-]", "—")

    st.markdown("---")

    # ----- Graphique barres Mesuré vs Théorique -----
    fig = go.Figure()
    fig.add_bar(
        x=comp_df["month_label"],
        y=comp_df["pv_measured_kWh"],
        name="Mesuré [kWh/mois]",
    )
    fig.add_bar(
        x=comp_df["month_label"],
        y=comp_df["pv_theoretical_kWh"],
        name="Théorique [kWh/mois]",
        opacity=0.7,
    )

    fig.update_layout(
        barmode="group",
        xaxis_title="Mois",
        yaxis_title="Énergie [kWh/mois]",
        height=400,
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    st.plotly_chart(fig, use_container_width=True)

    # ----- Tableau détaillé -----
    st.markdown("#### Détail mensuel")
    st.dataframe(
        comp_df[[
            "month",
            "pv_measured_kWh",
            "pv_theoretical_kWh",
            "ratio_theoretical_over_measured",
        ]],
        use_container_width=True,
    )


# =====================================================================
# COMPARAISON STANDARD PV
# =====================================================================

def show_comparison(project: dict):
    st.markdown(
        "<div class='sf-section-title'>Comparaison standard PV</div>",
        unsafe_allow_html=True,
    )

    results = project.get("results", {}) or {}
    pv_std = results.get("pv_standard", []) or []

    if not pv_std:
        st.info(
            "Aucune donnée pour la comparaison standard PV. "
            "Ajoute au moins un producteur PV correctement paramétré en Phase 1 "
            "et relance les calculs."
        )
        return

    # -------------------------------------------------------------
    # 1) Tableau récapitulatif des paramètres PV
    # -------------------------------------------------------------
    rows = []
    warnings_msgs = []

    for rec in pv_std:
        rows.append({
            "Bâtiment": rec.get("batiment_nom", ""),
            "Ouvrage": rec.get("ouvrage_nom", ""),
            "Technologie": rec.get("producer_techno", ""),
            "P_installée [kW]": rec.get("installed_kw", 0.0),
            "Orientation [°]": rec.get("orientation_deg"),
            "Inclinaison [°]": rec.get("inclinaison_deg"),
            "P_module [kW]": rec.get("p_module_kw"),
            "Surface module [m²]": rec.get("area_module_m2"),
            "Rendement module [%]": rec.get("eta_mod_pct"),
            "Prod. théorique [kWh/kW/an]": rec.get("theoretical_annual_kWh_per_kW"),
            "Prod. mesurée [kWh/kW/an]": rec.get("measured_annual_kWh_per_kW"),
        })

        missing = rec.get("missing_fields", []) or []
        if missing:
            label = f"{rec.get('batiment_nom','')} – {rec.get('ouvrage_nom','')} – {rec.get('producer_techno','PV')}"
            warnings_msgs.append(
                f"Producteur PV « {label} » : infos manquantes → {', '.join(missing)}"
            )

        if rec.get("calc_error"):
            label = f"{rec.get('batiment_nom','')} – {rec.get('ouvrage_nom','')} – {rec.get('producer_techno','PV')}"
            warnings_msgs.append(
                f"Erreur de calcul pour « {label} » : {rec['calc_error']}"
            )

    df = pd.DataFrame(rows)
    st.subheader("Paramètres PV utilisés pour la comparaison standard")
    st.dataframe(df, use_container_width=True)

    for msg in warnings_msgs:
        st.warning(msg)

    # -------------------------------------------------------------
    # 2) Profil mensuel théorique vs mesuré (on prend le 1er producteur valide)
    # -------------------------------------------------------------
    first_with_profile = None
    for rec in pv_std:
        if rec.get("theoretical_profile_kWh_per_kW") is not None:
            first_with_profile = rec
            break

    if first_with_profile is None:
        st.info("Profil mensuel théorique non disponible (aucun producteur PV avec profil calculé).")
        return

    theo_vals = first_with_profile.get("theoretical_profile_kWh_per_kW")
    meas_vals = first_with_profile.get("measured_profile_kWh_per_kW")

    def _to_list(x):
        if isinstance(x, pd.Series):
            return x.tolist()
        if isinstance(x, (list, tuple)):
            return list(x)
        return None

    theo_vals = _to_list(theo_vals)
    meas_vals = _to_list(meas_vals)

    if theo_vals is None or len(theo_vals) != 12:
        if first_with_profile.get("calc_error"):
            st.warning(f"Erreur de calcul PV : {first_with_profile['calc_error']}")
        else:
            st.info("Profil mensuel théorique non disponible (liste incorrecte).")
        return

    mois_labels = ["Jan", "Fév", "Mar", "Avr", "Mai", "Juin",
                   "Juil", "Août", "Sept", "Oct", "Nov", "Déc"]
    df_plot = pd.DataFrame({"Mois": mois_labels, "kWh/kW_th": theo_vals})

    if meas_vals is not None and len(meas_vals) == 12:
        df_plot["kWh/kW_meas"] = meas_vals

    bat_nom = first_with_profile.get("batiment_nom", "")
    ouv_nom = first_with_profile.get("ouvrage_nom", "")
    techno = first_with_profile.get("producer_techno", "PV")

    st.subheader(f"Profil théorique mensuel – {bat_nom} / {ouv_nom} / {techno}")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_plot["Mois"],
            y=df_plot["kWh/kW_th"],
            mode="lines+markers",
            name="Théorique",
        )
    )

    if "kWh/kW_meas" in df_plot.columns:
        fig.add_trace(
            go.Scatter(
                x=df_plot["Mois"],
                y=df_plot["kWh/kW_meas"],
                mode="lines+markers",
                name="Mesuré",
            )
        )

    fig.update_layout(
        xaxis_title="Mois",
        yaxis_title="Production spécifique [kWh/kW]",
        height=350,
        margin=dict(l=40, r=20, t=40, b=40),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)






def show_phases_analysis(project: dict):
    st.markdown("<div class='sf-section-title'>Analyse des phases de consommation / production</div>", unsafe_allow_html=True)
    st.info("Analyse des profils temporels, heures de pointe, potentiel de flexibilité, etc.")


def show_variantes(project: dict):
    st.markdown("<div class='sf-section-title'>Variantes</div>", unsafe_allow_html=True)
    results = project.get("results", {})
    st.write("Résultats bruts (placeholder) :", results)
    st.info("Section future pour comparer les variantes (coûts, CO₂, autonomie, etc.).")


# =====================================================================
# FALLBACK DF
# =====================================================================

def _get_fallback_dataframe_from_project(project: dict) -> pd.DataFrame | None:
    data_dict = project.get("raw_data_preview") or project.get("raw_data")
    if not data_dict:
        return None
    try:
        return pd.DataFrame(data_dict)
    except Exception:
        return None
