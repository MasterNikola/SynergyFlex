# -*- coding: utf-8 -*-
"""
Phase 2 - Présentation et analyse du projet SynergyFlex
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from core.pv_module import (
    compute_pv_monthly_energy_per_kw,
    _get_param_val,
)
from core.calculations import _build_monthly_series_from_df



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
            st.rerun()



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
        st.metric("Nom du projet", project.get("nom") or project.get("meta", {}).get("nom_projet", "—"))
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
            st.markdown("---")
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
    import streamlit as st
    import pandas as pd

    st.markdown("## Diagramme de Sankey – Flux énergétique réel du bâtiment")
    st.caption(
        "Le diagramme représente l’état physique réel du bâtiment. "
        "S’il existe un stockage, il est automatiquement intégré."
    )

    results = project.get("results", {}) or {}
    flows_by_bat = results.get("flows", {}).get("batiments", {}) or {}

    if not flows_by_bat:
        st.info("Aucun flux calculé. Retourne en Phase 1 et relance les calculs.")
        return

    # --- Sélection bâtiment uniquement
    batiments = project.get("batiments", []) or []
    bat_options = []
    for i, bat in enumerate(batiments):
        bat_id = bat.get("id") or bat.get("batiment_id") or i
        bat_nom = bat.get("nom") or f"Bâtiment {i+1}"
        bat_options.append((bat_id, bat_nom))

    if not bat_options:
        bat_options = [(k, f"Bâtiment {k}") for k in flows_by_bat.keys()]

    labels = [f"{bid} – {bnom}" for bid, bnom in bat_options]
    idx = st.selectbox("Bâtiment", range(len(bat_options)), format_func=lambda i: labels[i])
    bat_id, bat_nom = bat_options[idx]

    blocks = flows_by_bat.get(bat_id) or flows_by_bat.get(str(bat_id)) or []
    if not blocks:
        st.info("Aucun flow_block pour ce bâtiment.")
        return

    # --- Sélection AUTOMATIQUE du bon bloc global
    global_block = None
    for b in blocks:
        if b.get("type") == "energy_global_with_storage":
            global_block = b
            break
    if global_block is None:
        for b in blocks:
            if b.get("type") == "energy_global":
                global_block = b
                break

    if global_block is None:
        st.warning("Aucun bloc global trouvé. Impossible d’afficher le Sankey.")
        return

    # --- Sankey global
    fig = _build_sankey_from_flow_block(global_block)
    if fig is None:
        st.info("Impossible de construire le Sankey (liens nuls ou incomplets).")
        return

    st.plotly_chart(fig, use_container_width=True)

    # --- Totaux
    totals = global_block.get("totals", {}) or {}
    if totals:
        with st.expander("Totaux du bilan énergétique"):
            df = pd.DataFrame([totals]).T
            df.columns = ["Valeur"]
            st.dataframe(df, use_container_width=True)
    
    def _is_proposed_block(b: dict) -> bool:
        """
        Détecte si un flow_block appartient à une variante/proposition ("after").
        On exclut ces blocs dans la vue 'Flux énergétique réel'.
        Heuristique robuste (sans casser l'existant).
        """
        # 1) tag direct
        tag = b.get("tag")
        if isinstance(tag, str) and tag.lower() in ("after", "proposed", "proposal", "variant"):
            return True
    
        # 2) tags list
        tags = b.get("tags")
        if isinstance(tags, (list, tuple, set)):
            low = {str(t).lower() for t in tags}
            if {"after", "proposed", "proposal", "variant"} & low:
                return True
    
        # 3) meta fields
        meta = b.get("meta") or {}
        for k in ("scenario", "scope", "mode", "variant", "state"):
            v = meta.get(k)
            if isinstance(v, str) and v.lower() in ("after", "proposed", "proposal", "variant"):
                return True
    
        # 4) name fallback (souvent le seul indice)
        name = b.get("name")
        if isinstance(name, str) and ("propos" in name.lower() or "after" in name.lower()):
            return True
    
        return False
    
    # --- Détails stockage (EXISTANT / MESURÉ uniquement)
    storage_blocks = [
        b for b in blocks
        if b.get("type") in ("battery_elec", "storage_elec", "storage_th")
        and not _is_proposed_block(b)
    ]
    
    if storage_blocks:
        st.markdown("---")
        st.subheader("Détails stockage")
        st.caption("Sous-systèmes de stockage existants (vue mesurée uniquement).")
    
        for b in storage_blocks:
            fig_b = _build_sankey_from_flow_block(b)
            if fig_b is not None:
                st.plotly_chart(fig_b, use_container_width=True)





# =====================================================================
# 📊 ÉCONOMIE
# =====================================================================

def show_economics(project: dict):
    import streamlit as st
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go

    results = project.get("results", {}) or {}
    econ_by_bat = results.get("economics_by_batiment", {}) or {}
    batt_by_bat = results.get("storage_economics_by_batiment", {}) or {}
    batiments = project.get("batiments", []) or []

    if not econ_by_bat and not batt_by_bat:
        st.info("Aucune donnée économique disponible.")
        return

    # -------------------------
    # Helpers
    # -------------------------
    def _bat_name_from_id(bat_id):
        try:
            if isinstance(bat_id, int) and bat_id < len(batiments):
                return batiments[bat_id].get("nom") or f"Bâtiment {bat_id}"
        except Exception:
            pass
        return f"Bâtiment {bat_id}"

    def _cashflow_bar(years, cum, height=380):
        df_cf = pd.DataFrame({"Année": years, "Cumul [CHF]": cum})
        vals = df_cf["Cumul [CHF]"].tolist()
        colors = ["rgba(220,53,69,0.85)" if v < 0 else "rgba(40,167,69,0.85)" for v in vals]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df_cf["Année"], y=df_cf["Cumul [CHF]"], marker_color=colors))
        fig.update_layout(xaxis_title="Années", yaxis_title="CHF", height=height, margin=dict(l=10, r=10, t=20, b=10))
        return fig

    # ==========================================================
    # Onglets
    # ==========================================================
    tab1, tab2 = st.tabs(["⚙️ Machines / LCOE", "🔋 Batteries / Cashflow"])

    # ==========================================================
    # TAB 1 — MACHINES / LCOE
    # ==========================================================
    with tab1:
        rows = []
        detail_rows = []

        for bat_id, econ in econ_by_bat.items():
            # compat ancien format
            if isinstance(econ, dict) and "by_type" in econ:
                totals = econ
                by_type = econ.get("by_type", {}) or {}
            else:
                totals = econ
                by_type = {}

            nom_bat = _bat_name_from_id(bat_id)

            capex = float(totals.get("capex_total_CHF", 0.0) or 0.0)
            opex = float(totals.get("opex_annual_CHF", 0.0) or 0.0)
            prod = float(totals.get("production_totale_kWh", 0.0) or 0.0)
            lcoe = float(totals.get("lcoe_global_CHF_kWh", 0.0) or 0.0)

            rows.append({
                "Bâtiment": nom_bat,
                "CAPEX machines [CHF]": capex,
                "OPEX machines [CHF/an]": opex,
                "Production [kWh/an]": prod,
                "LCOE global [CHF/kWh]": lcoe,
            })

            for mtype, stats in by_type.items():
                detail_rows.append({
                    "Bâtiment": nom_bat,
                    "Type": mtype,
                    "CAPEX [CHF]": float(stats.get("capex_total_CHF", 0.0) or 0.0),
                    "OPEX [CHF/an]": float(stats.get("opex_annual_CHF", 0.0) or 0.0),
                    "Production [kWh/an]": float(stats.get("production_machine_kWh", 0.0) or 0.0),
                    "LCOE [CHF/kWh]": float(stats.get("lcoe_machine_CHF_kWh", 0.0) or 0.0),
                })

        df = pd.DataFrame(rows)

        # KPIs globaux
        capex_tot = float(df["CAPEX machines [CHF]"].sum()) if not df.empty else 0.0
        opex_tot = float(df["OPEX machines [CHF/an]"].sum()) if not df.empty else 0.0
        prod_tot = float(df["Production [kWh/an]"].sum()) if not df.empty else 0.0
        # lcoe “indicatif” global (pondéré énergie)
        lcoe_ind = (capex_tot + opex_tot) / prod_tot if prod_tot > 0 else 0.0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("CAPEX total (machines) [CHF]", f"{capex_tot:,.0f}")
        c2.metric("OPEX annuel (machines) [CHF/an]", f"{opex_tot:,.0f}")
        c3.metric("Production totale [kWh/an]", f"{prod_tot:,.0f}")
        c4.metric("LCOE indicatif [CHF/kWh]", f"{lcoe_ind:.3f}" if lcoe_ind > 0 else "—")

        st.markdown("#### Vue globale (machines / LCOE)")
        st.dataframe(df, use_container_width=True, hide_index=True)

        if not df.empty:
            g1, g2 = st.columns(2)
            with g1:
                fig_capex = px.bar(df, x="Bâtiment", y="CAPEX machines [CHF]", text_auto=True)
                fig_capex.update_layout(title="CAPEX machines par bâtiment", height=360, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig_capex, use_container_width=True)

            with g2:
                fig_lcoe = px.bar(df, x="Bâtiment", y="LCOE global [CHF/kWh]", text_auto=True)
                fig_lcoe.update_layout(title="LCOE global par bâtiment", height=360, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig_lcoe, use_container_width=True)

        # Détail par type (si dispo)
        if detail_rows:
            st.markdown("---")
            st.markdown("#### Détail par type de machine")
            df_detail = pd.DataFrame(detail_rows)
            st.dataframe(df_detail, use_container_width=True, hide_index=True)

            bat_list = df["Bâtiment"].tolist()
            selected_bat = st.selectbox("Bâtiment (détail CAPEX par techno)", options=bat_list, key="econ_bat_detail")
            df_bat = df_detail[df_detail["Bâtiment"] == selected_bat]
            if not df_bat.empty:
                fig_pie = px.pie(df_bat, names="Type", values="CAPEX [CHF]", title=f"Répartition CAPEX – {selected_bat}")
                fig_pie.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
                st.plotly_chart(fig_pie, use_container_width=True)

    # ==========================================================
    # TAB 2 — BATTERIES / CASHFLOW
    # ==========================================================
    with tab2:
        # Flatten batteries
        batt_rows = []
        for bat_id, lst in (batt_by_bat or {}).items():
            for b in (lst or []):
                econ = (b.get("economics") or {})
                years = econ.get("years") or []
                cum = econ.get("cashflow_cum_CHF") or []
                payback = econ.get("payback_year")
                benefit = econ.get("benefit_horizon_CHF") if econ.get("benefit_horizon_CHF") is not None else econ.get("benefit_25_CHF")
                capex0 = econ.get("capex_total_CHF", None)  # si tu l’as mis dans compute_battery_cashflow_series

                batt_rows.append({
                    "batiment_id": bat_id,
                    "Bâtiment": _bat_name_from_id(int(bat_id)) if str(bat_id).isdigit() else f"Bâtiment {bat_id}",
                    "Ouvrage": b.get("ouvrage_nom"),
                    "Techno": b.get("techno"),
                    "Mode": b.get("mode"),
                    "Capacité [kWh]": b.get("capacity_kwh"),
                    "CAPEX total [CHF]": capex0,
                    "Payback [an]": payback,
                    "Bénéfice horizon [CHF]": benefit,
                    "_years": years,
                    "_cum": cum,
                })

        if not batt_rows:
            st.info("Aucune économie batterie disponible.")
            return

        df_batt = pd.DataFrame(batt_rows)

        # tableau compact (visible)
        st.markdown("#### Batteries (cashflow)")
        visible_cols = ["Bâtiment", "Ouvrage", "Techno", "Mode", "Capacité [kWh]", "CAPEX total [CHF]", "Payback [an]", "Bénéfice horizon [CHF]"]
        st.dataframe(df_batt[visible_cols], use_container_width=True, hide_index=True)

        # Sélecteur “1 batterie”
        labels = [
            f"{r['Bâtiment']} | {r['Ouvrage']} | {r['Techno']} | {r['Capacité [kWh]']} kWh"
            for _, r in df_batt.iterrows()
        ]
        sel = st.selectbox("Choisir une batterie", options=list(range(len(labels))), format_func=lambda i: labels[i], key="batt_select")

        row = df_batt.iloc[int(sel)]
        years = row["_years"]
        cum = row["_cum"]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Payback", f"Année {int(row['Payback [an]'])}" if pd.notna(row["Payback [an]"]) else "—")
        m2.metric("Bénéfice horizon [CHF]", f"{float(row['Bénéfice horizon [CHF]']):,.0f}" if pd.notna(row["Bénéfice horizon [CHF]"]) else "—")
        m3.metric("Capacité [kWh]", f"{float(row['Capacité [kWh]']):.1f}" if pd.notna(row["Capacité [kWh]"]) else "—")
        m4.metric("CAPEX total [CHF]", f"{float(row['CAPEX total [CHF]']):,.0f}" if pd.notna(row["CAPEX total [CHF]"]) else "—")

        if isinstance(years, list) and isinstance(cum, list) and len(years) == len(cum) and len(years) > 0:
            st.plotly_chart(_cashflow_bar(years, cum, height=420), use_container_width=True)
        else:
            st.info("Cashflow batterie incomplet (years / cashflow_cum_CHF).")





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
    st.markdown(
        "<div class='sf-section-title'>Analyse temporelle : consommation vs production PV vs autoconsommation</div>",
        unsafe_allow_html=True,
    )
    st.info(
        "Données temporelles détaillées (point par point) + vues trimestrielles."
    )

       # ------------------------------------------------------------------
    # 4) Graphique temporel détaillé
    # ------------------------------------------------------------------
    st.subheader("Courbes temporelles détaillées")
    st.caption(
    "ℹ️ La courbe d’autoconsommation PV (rouge) représente l’énergie photovoltaïque "
    "réellement consommée par le bâtiment à chaque pas de temps. "
    "Lorsqu’elle n’est pas mesurée, elle est reconstruite dans la Phase 1 "
    "par une méthode proportionnelle physiquement bornée "
    "(autoconsommation ≤ production PV et ≤ consommation électrique)."
)

    
    # ------------------------------------------------------------------
    # OVERRIDE df_full : on trace UNIQUEMENT depuis results (source de vérité)
    # ------------------------------------------------------------------
    results = project.get("results", {}) or {}
    ts_by_bat = results.get("timeseries_by_batiment", {}) or {}
    
    if not ts_by_bat:
        st.warning("Aucune série temporelle calculée dans results. Relancer la Phase 1.")
        return
    
    agg_load = None
    agg_pv = None
    agg_pv_auto = None
    
    base_index = None
    
    for _bat_id, bat_data in ts_by_bat.items():
        measured = (bat_data.get("measured") or {})
        idx_raw = bat_data.get("index") or measured.get("index")
        if not idx_raw:
            continue
    
        dt_idx = pd.to_datetime(idx_raw, errors="coerce")
        # si parsing impossible -> fallback index positionnel
        if dt_idx.isna().all():
            dt_idx = pd.RangeIndex(len(idx_raw))
    
        load_list = measured.get("load_kWh")
        pv_list = measured.get("pv_prod_kWh")
        auto_list = measured.get("pv_to_load_kWh")
    
        if load_list is None and pv_list is None:
            continue
    
        if base_index is None:
            base_index = dt_idx
    
        # aligne sur base_index (même longueur), padding/troncature minimale
        def _to_series(v):
            n = len(base_index)
            if v is None:
                return pd.Series([0.0] * n, index=base_index)
        
            # force list-like
            arr = list(v)
            if len(arr) < n:
                arr = arr + [0.0] * (n - len(arr))
            elif len(arr) > n:
                arr = arr[:n]
        
            return pd.Series(arr, index=base_index, dtype=float)


        s_load = _to_series(load_list)
        s_pv = _to_series(pv_list)
        s_auto = _to_series(auto_list)
    
        agg_load = s_load if agg_load is None else agg_load.add(s_load, fill_value=0.0)
        agg_pv = s_pv if agg_pv is None else agg_pv.add(s_pv, fill_value=0.0)
        agg_pv_auto = s_auto if agg_pv_auto is None else agg_pv_auto.add(s_auto, fill_value=0.0)
    
    if agg_load is None and agg_pv is None:
        st.warning("Séries timeseries_by_batiment présentes mais inexploitables.")
        return
    
    df_full = pd.DataFrame({
        "Consommation [kWh]": agg_load if agg_load is not None else pd.Series(dtype=float),
        "Production PV [kWh]": agg_pv if agg_pv is not None else pd.Series(dtype=float),
        "Autocons. PV [kWh]": agg_pv_auto if agg_pv_auto is not None else pd.Series(dtype=float),
    }).sort_index()
    
    index_type = "datetime" if isinstance(df_full.index, pd.DatetimeIndex) else "int"

    fig_full = px.line(
        df_full,
        x=df_full.index,
        y=df_full.columns,
        title="Conso vs PV vs Autoconsommation – détail temporel",
    )
    # Conso en avant-plan, PV plus discret
    for tr in fig_full.data:
        if "Consommation" in tr.name:
            tr.update(line=dict(width=3))              # trait épais
        elif "Production PV" in tr.name:
            tr.update(line=dict(width=1, dash="dash"))  # PV en pointillé fin
        else:
            tr.update(line=dict(width=1))               # autoconsommation normal

    fig_full.update_layout(
        xaxis_title="Temps" if index_type == "datetime" else "Index (pas de temps)",
        yaxis_title="Énergie [kWh]",
        legend_title="Flux",
    )
    
    # --- PV self-consumption saturation warning ---
    for bat_id, bat_data in ts_by_bat.items():
        meta = bat_data.get("measured", {}).get("pv_selfc_meta")
        if meta and meta.get("saturated", False):
            st.warning(
                f"⚠️ Autoconsommation demandée = {meta['selfc_pct_input']*100:.1f} % "
                f"→ maximum physique atteignable = {meta['max_selfc_pct']*100:.1f} %. "
                f"Résultat obtenu = {meta['selfc_pct_real']*100:.1f} %."
            )
            break

    st.plotly_chart(fig_full, use_container_width=True)

    # ------------------------------------------------------------------
    # 5) Vue trimestrielle (avec fallback si pas de dates)
    # ------------------------------------------------------------------
    st.subheader("Synthèse par périodes de 3 mois")

    def compute_quarterly_from_datetime(df):
        df_q = df.copy()
        df_q["month"] = df_q.index.month

        groups = {
            "Jan–Mar": [1, 2, 3],
            "Avr–Jun": [4, 5, 6],
            "Jul–Sep": [7, 8, 9],
            "Oct–Déc": [10, 11, 12],
        }

        rows = []
        for label, months in groups.items():
            subset = df_q[df_q["month"].isin(months)]
            rows.append(
                {
                    "Période": label,
                    "Consommation [kWh]": subset["Consommation [kWh]"].sum(),
                    "Production PV [kWh]": subset["Production PV [kWh]"].sum(),
                    "Autocons. PV [kWh]": subset["Autocons. PV [kWh]"].sum(),
                }
            )
        return pd.DataFrame(rows)

    def compute_quarterly_fallback(df):
        # Découpe en 4 blocs égaux (si pas de dates disponibles)
        n = len(df)
        if n == 0:
            return None

        q = n // 4
        slices = [
            ("Q1", df.iloc[0:q]),
            ("Q2", df.iloc[q:2*q]),
            ("Q3", df.iloc[2*q:3*q]),
            ("Q4", df.iloc[3*q:]),
        ]

        rows = []
        for label, part in slices:
            rows.append(
                {
                    "Période": label,
                    "Consommation [kWh]": part["Consommation [kWh]"].sum(),
                    "Production PV [kWh]": part["Production PV [kWh]"].sum(),
                    "Autocons. PV [kWh]": part["Autocons. PV [kWh]"].sum(),
                }
            )
        return pd.DataFrame(rows)

    # Cas 1 : VRAIES dates -> grouping par mois
    if index_type == "datetime":
        df_quarters = compute_quarterly_from_datetime(df_full)

    else:
        # Cas 2 : fallback -> découpage en 4 blocs temporels équivalents
        df_quarters = compute_quarterly_fallback(df_full)

    if df_quarters is not None:
        df_quarters_long = df_quarters.melt(
            id_vars="Période",
            var_name="Type",
            value_name="Énergie [kWh]",
        )

        fig_quarters = px.bar(
            df_quarters_long,
            x="Période",
            y="Énergie [kWh]",
            color="Type",
            barmode="group",
        )
        fig_quarters.update_layout(
            xaxis_title="Période",
            yaxis_title="Énergie [kWh]",
            legend_title="Flux",
        )
        st.plotly_chart(fig_quarters, use_container_width=True)

def show_variantes(project: dict):
    import streamlit as st
    import pandas as pd
    import plotly.graph_objects as go

    def _sum_block_totals(blocks, btype: str, key: str) -> float:
        s = 0.0
        for b in (blocks or []):
            if b.get("type") == btype:
                t = b.get("totals", {}) or {}
                s += float(t.get(key, 0.0) or 0.0)
        return float(s)

    def _make_sankey(title: str, nodes: list, links: list, height: int = 520):
        """
        nodes: [{"id":..., "label":...}, ...]
        links: [{"source":node_id, "target":node_id, "value":float}, ...]
        """
        id_to_idx = {n["id"]: i for i, n in enumerate(nodes)}
        labels = [n["label"] for n in nodes]

        # Couleurs type "comme avant" (PV orange, réseau bleu, conso vert)
        def node_color(label: str) -> str:
            l = (label or "").lower()
            if "pv" in l:
                return "rgba(255,193,7,0.90)"     # orange
            if "réseau" in l or "reseau" in l or "grid" in l:
                return "rgba(33,150,243,0.60)"    # bleu
            if "conso" in l or "consommation" in l or "load" in l:
                return "rgba(76,175,80,0.60)"     # vert
            return "rgba(160,160,160,0.45)"       # gris

        node_colors = [node_color(n["label"]) for n in nodes]

        sources, targets, values, link_colors = [], [], [], []
        for lk in links:
            v = float(lk.get("value", 0.0) or 0.0)
            if v <= 0:
                continue
            s_id, t_id = lk.get("source"), lk.get("target")
            if s_id not in id_to_idx or t_id not in id_to_idx:
                continue

            s_lab = nodes[id_to_idx[s_id]]["label"]
            # Liens colorés selon la source
            if "pv" in (s_lab or "").lower():
                link_c = "rgba(255,193,7,0.45)"
            elif "réseau" in (s_lab or "").lower() or "reseau" in (s_lab or "").lower() or "grid" in (s_lab or "").lower():
                link_c = "rgba(33,150,243,0.35)"
            else:
                link_c = "rgba(120,120,120,0.25)"

            sources.append(id_to_idx[s_id])
            targets.append(id_to_idx[t_id])
            values.append(v)
            link_colors.append(link_c)

        fig = go.Figure(
            data=[
                go.Sankey(
                    node=dict(label=labels, pad=15, thickness=18, color=node_colors),
                    link=dict(source=sources, target=targets, value=values, color=link_colors),
                )
            ]
        )
        fig.update_layout(title=title, height=height)
        return fig
    def _get_batt_proposed_summary(results: dict, bat_id):
        bp = (results.get("battery_proposed_by_batiment") or {})
        return bp.get(bat_id) or bp.get(str(bat_id))
    
    def _get_ts_payload(results: dict, bat_id):
        ts_root = (results.get("timeseries_by_batiment") or {})
        bat_data = ts_root.get(bat_id) or ts_root.get(str(bat_id))
        if not isinstance(bat_data, dict):
            return None
    
        # Nouveau format (recommandé): bat_data["battery_proposed"]
        if isinstance(bat_data.get("battery_proposed"), dict):
            return bat_data["battery_proposed"]
    
        # Ancien format (fallback): bat_data contient directement before/after/index
        if "before" in bat_data or "after" in bat_data:
            return bat_data
    
        return None

    
    def _quarters_from_index(idx: pd.DatetimeIndex) -> pd.Series:
        # returns PeriodIndex-like labels: 'Jan–Mar 2024' etc.
        q = idx.to_period("Q")
        # label friendly
        return q.astype(str)
    
    def _agg_by_quarter(idx: pd.DatetimeIndex, series: pd.Series) -> pd.DataFrame:
        df = pd.DataFrame({"v": series.values}, index=idx)
        q = idx.to_period("Q")
        g = df.groupby(q)["v"].sum()
        out = g.reset_index()
        out.columns = ["Quarter", "kWh"]
        out["Quarter"] = out["Quarter"].astype(str)
        return out
    
    def _bar_cum_cashflow(years: list, cashflow_cum: list, title: str):
        df_cf = pd.DataFrame({"Année": years, "Cumul [CHF]": cashflow_cum})
        vals = df_cf["Cumul [CHF]"].tolist()
        colors = ["rgba(220,53,69,0.85)" if v < 0 else "rgba(40,167,69,0.85)" for v in vals]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df_cf["Année"], y=df_cf["Cumul [CHF]"], marker_color=colors))
        fig.update_layout(title=title, xaxis_title="Années", yaxis_title="CHF", height=360)
        return fig

    st.markdown("## Variantes")

    results = project.get("results", {}) or {}
    flows = (results.get("flows", {}) or {}).get("batiments", {}) or {}

    # PV simulé calculé dans calculations.py (clé = bat_id idéalement)
    pv_sim_all = (project.get("pv_simulated") or results.get("pv_proposed_by_batiment") or {})
    if not isinstance(pv_sim_all, dict):
        pv_sim_all = {}


    batiments = project.get("batiments", []) or []
    if not batiments:
        st.warning("Aucun bâtiment dans le projet.")
        return

    # ---- Sélecteur bâtiment ----
    opts = []
    for bi, bat in enumerate(batiments):
        bat_id = bat.get("id") or bat.get("batiment_id") or bi
        bat_nom = bat.get("nom") or f"Bâtiment {bi+1}"
        opts.append((bat_id, bat_nom))

    labels = [f"{bid} – {bnom}" for bid, bnom in opts]
    idx = st.selectbox("Bâtiment", list(range(len(opts))), format_func=lambda i: labels[i])
    bat_id, bat_nom = opts[idx]

    blocks = flows.get(bat_id) or flows.get(str(bat_id)) or []
    batt_summary = _get_batt_proposed_summary(results, bat_id)
    ts_payload = _get_ts_payload(results, bat_id)

    # ---- Totaux DEMANDE & PV EXISTANT ----
    demand_annual = _sum_block_totals(blocks, "demand_elec", "demand_elec_kWh")
    pv_exist_prod = _sum_block_totals(blocks, "pv", "pv_prod_kWh")
    pv_exist_auto = _sum_block_totals(blocks, "pv", "pv_auto_kWh")
    pv_exist_inj  = _sum_block_totals(blocks, "pv", "pv_inj_kWh")

    has_demand = demand_annual > 0
    has_pv_exist = pv_exist_prod > 0

    # ---- PV SIMULÉ ----
    sim = pv_sim_all.get(bat_id) or pv_sim_all.get(str(bat_id))
    has_pv_sim = isinstance(sim, dict) and bool(sim)
    
    if not has_pv_sim:
        st.info("Aucun PV proposé (toiture) n'est disponible pour ce bâtiment.")
    else:
        sim_annual = float(sim.get("annual_kwh", 0.0) or 0.0)
        if sim_annual <= 0:
            st.info("PV proposé présent mais production nulle (toits non éligibles / filtres).")
            has_pv_sim = False
    
    if has_pv_sim:
        s_usable = float(sim.get("surface_usable_total_m2", 0.0) or 0.0)
        p_tot_kw = float(sim.get("p_tot_kw", 0.0) or 0.0)
    
        if "selfc_pct" not in sim:
            st.error("PV proposé: selfc_pct manquant dans results. Relancer Phase 1 / calculations.")
            has_pv_sim = False
        else:
            selfc_pct_param = float(sim["selfc_pct"])
            sim_selfc_param_kwh = float(sim.get("selfc_kwh", sim_annual * selfc_pct_param) or 0.0)
            sim_inj_param_kwh = float(sim.get("inj_kwh", max(sim_annual - sim_selfc_param_kwh, 0.0)) or 0.0)
    
            if has_demand:
                sim_auto_used_kwh = min(sim_selfc_param_kwh, demand_annual)
                effective_selfc_pct = sim_auto_used_kwh / sim_annual if sim_annual > 0 else 0.0
                grid_to_load_kwh = max(demand_annual - sim_auto_used_kwh, 0.0)
                pv_to_grid_kwh = max(sim_annual - sim_auto_used_kwh, 0.0)
            else:
                sim_auto_used_kwh = 0.0
                effective_selfc_pct = 0.0
                grid_to_load_kwh = 0.0
                pv_to_grid_kwh = sim_annual

        # ==========================================================
        # SECTION 1 — PV PROPOSÉ (toiture)
        # ==========================================================
        # (ton code existant Section 1-4 ici, indenté)

        # ==========================================================
        # SECTION 1 — PV PROPOSÉ (toiture)
        # ==========================================================
        st.markdown("### 🟠 PV proposé (toiture)")
    
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Surf. toiture utilisée [m²]", f"{s_usable:,.1f}")
        c2.metric("Puissance proposée [kW]", f"{p_tot_kw:,.2f}")
        c3.metric("Production annuelle [kWh/an]", f"{sim_annual:,.0f}")
        # Affiche un truc intelligent: 0% si pas de conso
        c4.metric("Autoconsommation [%]", f"{selfc_pct_param*100:,.0f}%")
    
        # ------------------ Sankey PV proposé ------------------
        st.markdown("#### Flux PV proposé (hypothèse autoconsommation fixe)")
    
        # IDs
        N_PV_SIM = "PV_SIM"
        N_SELF = "SELF"        # autoconsommation (hypothèse fixe)
        N_GRID = "GRID"
        N_LOAD = "LOAD"        # demande (si dispo)
        
        # Hypothèse fixe (toujours visible)
        selfc_pct_param = float(sim.get("selfc_pct", 0.20) or 0.20)
        selfc_kwh_hyp = float(sim.get("selfc_kwh", sim_annual * selfc_pct_param) or 0.0)
        inj_kwh_hyp = float(sim.get("inj_kwh", max(sim_annual - selfc_kwh_hyp, 0.0)) or 0.0)
        
        # Sécurité
        selfc_kwh_hyp = max(selfc_kwh_hyp, 0.0)
        inj_kwh_hyp = max(inj_kwh_hyp, 0.0)
        
        # Recalage si incohérence
        if abs((selfc_kwh_hyp + inj_kwh_hyp) - sim_annual) > 1e-6:
            inj_kwh_hyp = max(sim_annual - selfc_kwh_hyp, 0.0)
        
        # Si on a une demande, on ajoute la ligne Réseau->Conso + Self->Conso
        # (tout en gardant la division 20% visible)
        has_demand = demand_annual > 0
        
        if has_demand:
            # On limite l’autoconsommation par la demande (réel utilisé)
            self_used = min(selfc_kwh_hyp, demand_annual)
            grid_to_load = max(demand_annual - self_used, 0.0)
            # La part "self" non utilisée est renvoyée au réseau pour fermer le bilan
            self_unused_to_grid = max(selfc_kwh_hyp - self_used, 0.0)
        
            nodes_sim = [
                {"id": N_PV_SIM, "label": "PV proposé"},
                {"id": N_SELF, "label": f"Autoconsommation ({int(selfc_pct_param*100)}%)"},
                {"id": N_GRID, "label": "Réseau"},
                {"id": N_LOAD, "label": "Consommation élec"},
            ]
            links_sim = [
                # division hypothèse visible
                {"source": N_PV_SIM, "target": N_SELF, "value": selfc_kwh_hyp},
                {"source": N_PV_SIM, "target": N_GRID, "value": inj_kwh_hyp},
                # utilisation réelle sur la conso
                {"source": N_SELF, "target": N_LOAD, "value": self_used},
                {"source": N_GRID, "target": N_LOAD, "value": grid_to_load},
            ]
            # fermeture du bilan si self > demande
            if self_unused_to_grid > 0:
                links_sim.append({"source": N_SELF, "target": N_GRID, "value": self_unused_to_grid})
        
        else:
            # Pas de demande : on garde uniquement la division 20% vs 80%
            nodes_sim = [
                {"id": N_PV_SIM, "label": "PV proposé"},
                {"id": N_SELF, "label": f"Autoconsommation ({int(selfc_pct_param*100)}%)"},
                {"id": N_GRID, "label": "Réseau"},
            ]
            links_sim = [
                {"source": N_PV_SIM, "target": N_SELF, "value": selfc_kwh_hyp},
                {"source": N_PV_SIM, "target": N_GRID, "value": inj_kwh_hyp},
            ]
        
        st.plotly_chart(_make_sankey(f"{bat_nom} — PV proposé", nodes_sim, links_sim), use_container_width=True)
    
        # ------------------ Détails des pans retenus ------------------
        with st.expander("Détails des pans retenus", expanded=False):
            eligible = sim.get("eligible_roofs", []) or []
            if not eligible:
                st.info("Aucun pan détaillé disponible.")
            else:
                df = pd.DataFrame(eligible)
                preferred_cols = []
                for col in [
                    "orientation_deg", "orientation_code",
                    "inclinaison_deg",
                    "surface_usable_m2", "surface_utilisable_m2", "surface utilisiable",
                    "p_kw",
                ]:
                    if col in df.columns:
                        preferred_cols.append(col)
                st.dataframe(df[preferred_cols] if preferred_cols else df, use_container_width=True)
    
        # ==========================================================
        # SECTION 2 — Indicateurs économiques (PV proposé)
        # ==========================================================
        st.markdown("#### Indicateurs économiques (PV proposé)")
        econ = (sim.get("economics") or {})
    
        capex = float(econ.get("capex_total_CHF", 0.0) or 0.0)
        opex = float(econ.get("opex_annual_CHF", 0.0) or 0.0)
    
        # LCOE : on tente plusieurs clés + fallback calcul simple si absent
        lcoe = (
            econ.get("lcoe_chf_kwh")
            or econ.get("LCOE_CHF_kWh")
            or econ.get("lcoe")
            or econ.get("LCOE")
        )
        
        try:
            lcoe = float(lcoe) if lcoe not in (None, "") else None
        except Exception:
            lcoe = None
        
        # fallback simple si pas calculé:
        # LCOE ≈ (CAPEX + OPEX*H) / (Prod_annuelle*H)
        if lcoe is None:
            try:
                annual = float(sim_annual or 0.0)
                p = project.get("params") or {}
                horizon = int(p.get("horizon_analyse_ans") or 0)
                if horizon <= 0:
                    st.error("Paramètre manquant: params.horizon_analyse_ans (>0). Retour Phase 1.")
                    return
    
                if annual > 0:
                    lcoe = (capex + opex * horizon) / (annual * horizon)
            except Exception:
                lcoe = None
    
        e1, e2, e3 = st.columns(3)
        e1.metric("CAPEX total [CHF]", f"{capex:,.0f}")
        e2.metric("OPEX annuel [CHF/an]", f"{opex:,.0f}")
        e3.metric("LCOE [CHF/kWh]", f"{lcoe:.3f}" if lcoe is not None else "—")
    
        # ==========================================================
        # SECTION 3 — Rentabilité PV (cumul) avec barres rouge/vert
        # ==========================================================
        st.markdown("#### Rentabilité PV (cumul)")
    
        years = econ.get("years")
        cashflow_cum = econ.get("cashflow_cum_CHF")
    
        if isinstance(years, list) and isinstance(cashflow_cum, list) and len(years) == len(cashflow_cum):
            df_cf = pd.DataFrame({"Année": years, "Cumul [CHF]": cashflow_cum})
    
            vals = df_cf["Cumul [CHF]"].tolist()
            colors = ["rgba(220,53,69,0.85)" if v < 0 else "rgba(40,167,69,0.85)" for v in vals]
    
            fig = go.Figure()
            fig.add_trace(go.Bar(x=df_cf["Année"], y=df_cf["Cumul [CHF]"], marker_color=colors))
            fig.update_layout(xaxis_title="Années", yaxis_title="CHF", height=380)
            st.plotly_chart(fig, use_container_width=True)
    
            payback_year = econ.get("payback_year", None)
            benefit_25 = float(econ.get("benefit_25_CHF", 0.0) or 0.0)
    
            ru_amount = econ.get("ru_amount_CHF", None)
            
            try:
                ru_amount = float(ru_amount) if ru_amount not in (None, "") else None
            except Exception:
                ru_amount = None
            
            r1, r2, r3 = st.columns(3)
            
            # RU
            if ru_amount is not None:
                r1.metric("RU après 1 an [CHF]", f"{ru_amount:,.0f}")
                r1.caption("Période estimée : avril 2025 → avril 2026")
            else:
                r1.metric("RU après 1 an [CHF]", "—")
            
            # Rentabilité
            r2.metric(
                "Rentabilité",
                f"Année {payback_year}" if payback_year is not None else "—"
            )
            
            # Bénéfice à 25 ans
            r3.metric(
                "Bénéfice à 25 ans [CHF]",
                f"{benefit_25:,.0f}"
            )
        else:
            st.info("Pas de cashflow cumulé disponible dans economics (years / cashflow_cum_CHF).")
    
        # ==========================================================
        # SECTION 4 — COMBINÉ automatique (si PV existant)
        # ==========================================================
        if has_pv_exist:
            st.markdown("---")
            st.markdown("### 🔵 PV existant + PV proposé (combiné)")
    
            N_PV_EXIST = "PV_EXIST"
    
            nodes_comb = [
                {"id": N_PV_EXIST, "label": "PV existant"},
                {"id": N_PV_SIM, "label": "PV proposé"},
                {"id": N_GRID, "label": "Réseau"},
                {"id": N_LOAD, "label": "Consommation élec"},
            ]
    
            if has_demand:
                # on sert la demande: PV existant -> PV proposé -> réseau
                exist_auto_used = min(pv_exist_auto, demand_annual)
                remaining = max(demand_annual - exist_auto_used, 0.0)
    
                sim_auto_used = min(sim_selfc_param_kwh, remaining)
                remaining2 = max(remaining - sim_auto_used, 0.0)
    
                grid_to_load = remaining2
    
                pv_exist_to_grid = max(pv_exist_prod - exist_auto_used, 0.0)
                pv_sim_to_grid = max(sim_annual - sim_auto_used, 0.0)
    
                links_comb = [
                    {"source": N_PV_EXIST, "target": N_LOAD, "value": exist_auto_used},
                    {"source": N_PV_SIM, "target": N_LOAD, "value": sim_auto_used},
                    {"source": N_GRID, "target": N_LOAD, "value": grid_to_load},
                    {"source": N_PV_EXIST, "target": N_GRID, "value": pv_exist_to_grid},
                    {"source": N_PV_SIM, "target": N_GRID, "value": pv_sim_to_grid},
                ]
            else:
                links_comb = [
                    {"source": N_PV_EXIST, "target": N_GRID, "value": pv_exist_prod},
                    {"source": N_PV_SIM, "target": N_GRID, "value": sim_annual},
                ]
    
            st.plotly_chart(_make_sankey(f"{bat_nom} — Combiné", nodes_comb, links_comb), use_container_width=True)
    
            comb_prod = pv_exist_prod + sim_annual
            comb_auto = pv_exist_auto + sim_selfc_param_kwh
            comb_inj = pv_exist_inj + sim_inj_param_kwh
    
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Demande annuelle [kWh]", f"{demand_annual:,.0f}")
            k2.metric("Prod PV combinée [kWh]", f"{comb_prod:,.0f}")
            k3.metric("Auto combinée [kWh]", f"{comb_auto:,.0f}")
            k4.metric("Injection combinée [kWh]", f"{comb_inj:,.0f}")
    
            st.caption("Affichage automatique : PV existant détecté dans results['flows']['batiments'][bat_id].")
        
    # ==========================================================
    # SECTION 5 — BATTERIE PROPOSÉE (AVANT / APRÈS)
    # ==========================================================
    if batt_summary:
        st.markdown("---")
        st.markdown("## 🔋 Batterie proposée — Comparaison avant / après")
        
        # ---------- Résumé dimensionnement batterie ----------
        st.markdown("### Résultat concret — Dimensionnement batterie")
        
        cap_total = float(batt_summary.get("capacity_kwh", 0.0) or 0.0)
        pack_caps = batt_summary.get("pack_capacities_kwh") or []
        p_target_kw = batt_summary.get("p_target_kw")
        hours_target = batt_summary.get("hours_target")
        eta_global = batt_summary.get("eta_global")
        
        # Normalisation packs
        if not isinstance(pack_caps, list):
            pack_caps = []
        pack_caps_clean = []
        for x in pack_caps:
            try:
                pack_caps_clean.append(float(x))
            except Exception:
                pass
        
        n_packs = len(pack_caps_clean) if pack_caps_clean else (1 if cap_total > 0 else 0)
        cap_pack_avg = (cap_total / n_packs) if n_packs > 0 else 0.0
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Capacité totale [kWh]", f"{cap_total:,.1f}")
        c2.metric("Nombre de packs [-]", f"{n_packs:d}")
        c3.metric("Capacité moyenne / pack [kWh]", f"{cap_pack_avg:,.1f}" if n_packs > 0 else "—")
        c4.metric("Rendement global η [-]", f"{float(eta_global):.3f}" if eta_global is not None else "—")
        
        d1, d2 = st.columns(2)
        d1.metric("P cible (charge/décharge) [kW]", f"{float(p_target_kw):,.1f}" if p_target_kw is not None else "—")
        d2.metric("Heures cible [h]", f"{float(hours_target):.1f}" if hours_target is not None else "—")
        
        # Détail packs (si plusieurs)
        if pack_caps_clean:
            with st.expander("Détail des packs (capacités individuelles)", expanded=False):
                import pandas as pd
                df_packs = pd.DataFrame({
                    "Pack": list(range(1, len(pack_caps_clean) + 1)),
                    "Capacité [kWh]": pack_caps_clean,
                })
                st.dataframe(df_packs, use_container_width=True)

    
        # ---------- Sankey AVANT / APRÈS ----------
        st.markdown("### Sankey — Avant vs Après batterie")
    
        b_before = batt_summary.get("totals_before") or {}
        b_after = batt_summary.get("totals_after") or {}
    
        # Nodes
        N_PV = "PV"
        N_GRID = "GRID"
        N_LOAD = "LOAD"
        N_BATT = "BATT"
        N_LOSS = "LOSS"
    
        nodes_base = [
            {"id": N_PV, "label": "PV"},
            {"id": N_GRID, "label": "Réseau"},
            {"id": N_LOAD, "label": "Consommation élec"},
        ]
    
        links_before = [
            {"source": N_PV, "target": N_LOAD, "value": float(b_before.get("pv_to_load_kwh", 0.0) or 0.0)},
            {"source": N_GRID, "target": N_LOAD, "value": float(b_before.get("grid_to_load_kwh", 0.0) or 0.0)},
            {"source": N_PV, "target": N_GRID, "value": float(b_before.get("pv_to_grid_kwh", 0.0) or 0.0)},
        ]
    
        nodes_after = [
            {"id": N_PV, "label": "PV"},
            {"id": N_BATT, "label": "Batterie"},
            {"id": N_LOSS, "label": "Pertes"},
            {"id": N_GRID, "label": "Réseau"},
            {"id": N_LOAD, "label": "Consommation élec"},
        ]
        links_after = [
            {"source": N_PV, "target": N_LOAD, "value": float(b_after.get("pv_to_load_kwh", 0.0) or 0.0)},
            {"source": N_PV, "target": N_BATT, "value": float(b_after.get("pv_to_batt_kwh", 0.0) or 0.0)},
            {"source": N_BATT, "target": N_LOAD, "value": float(b_after.get("batt_to_load_kwh", 0.0) or 0.0)},
            {"source": N_GRID, "target": N_LOAD, "value": float(b_after.get("grid_to_load_kwh", 0.0) or 0.0)},
            {"source": N_PV, "target": N_GRID, "value": float(b_after.get("pv_to_grid_kwh", 0.0) or 0.0)},
            {"source": N_BATT, "target": N_LOSS, "value": float(b_after.get("losses_kwh", 0.0) or 0.0)},
        ]
    
        cL, cR = st.columns(2)
        with cL:
            st.plotly_chart(_make_sankey(f"{bat_nom} — Avant batterie", nodes_base, links_before, height=420),
                            use_container_width=True)
        with cR:
            st.plotly_chart(_make_sankey(f"{bat_nom} — Après batterie", nodes_after, links_after, height=420),
                            use_container_width=True)
    
        # KPIs simples avant/après
        demand_kwh = float(b_after.get("demand_kwh", b_before.get("demand_kwh", 0.0)) or 0.0)
        pv_prod_kwh = float(b_after.get("pv_prod_kwh", b_before.get("pv_prod_kwh", 0.0)) or 0.0)
    
        pv_to_load_before = float(b_before.get("pv_to_load_kwh", 0.0) or 0.0)
        pv_to_load_after = float(b_after.get("pv_to_load_kwh", 0.0) or 0.0)
        batt_to_load = float(b_after.get("batt_to_load_kwh", 0.0) or 0.0)
    
        sc_before = (pv_to_load_before / pv_prod_kwh) if pv_prod_kwh > 0 else 0.0
        sc_after = ((pv_to_load_after + batt_to_load) / pv_prod_kwh) if pv_prod_kwh > 0 else 0.0
    
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Demande annuelle [kWh]", f"{demand_kwh:,.0f}")
        k2.metric("Production PV [kWh]", f"{pv_prod_kwh:,.0f}")
        k3.metric("Autoconsommation avant [%]", f"{sc_before*100:,.1f}%")
        k4.metric("Autoconsommation après [%]", f"{sc_after*100:,.1f}%")
    
        # ---------- Courbes temporelles ----------
        st.markdown("### Courbes temporelles détaillées")
    
        if not ts_payload:
            st.info("Timeseries batterie non disponibles dans results['timeseries_by_batiment']. Relancer les calculs.")
        else:
            idx_str = ts_payload.get("index") or []
            if not idx_str:
                st.info("Timeseries: index vide.")
            else:
                idx = pd.to_datetime(idx_str)
                idx = pd.DatetimeIndex(idx)

    
                before = (ts_payload.get("before") or {})
                after = (ts_payload.get("after") or {})
    
                load_kwh = pd.Series(before.get("load_kWh") or [], index=idx)
                pv_kwh = pd.Series(before.get("pv_prod_kWh") or [], index=idx)
                pv_to_load = pd.Series(before.get("pv_to_load_kWh") or [], index=idx)
    
                # après (peut être partiellement None selon ton module)
                pv_to_batt = after.get("pv_to_batt_kWh")
                batt_to_load_ts = after.get("batt_to_load_kWh")
                soc_ts = after.get("soc_kWh")
    
                # sécurités
                if len(load_kwh) == 0 or len(pv_kwh) == 0:
                    st.info("Timeseries: séries load/pv vides.")
                else:
                    # AVANT: conso, PV, auto
                    fig1 = go.Figure()
                    fig1.add_trace(go.Scatter(x=idx, y=load_kwh, mode="lines", name="Consommation [kWh]"))
                    fig1.add_trace(go.Scatter(x=idx, y=pv_kwh, mode="lines", name="Production PV [kWh]"))
                    fig1.add_trace(go.Scatter(x=idx, y=pv_to_load, mode="lines", name="Autocons. PV [kWh]"))
                    fig1.update_layout(
                        title=f"{bat_nom} — Avant batterie (détail temporel)",
                        xaxis_title="Temps",
                        yaxis_title="Énergie [kWh]",
                        height=380,
                        legend_title="Flux",
                    )
                    st.plotly_chart(fig1, use_container_width=True)
    
                    # APRÈS: ajoute PV->Batt, Batt->Load, SOC si dispo
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(x=idx, y=load_kwh, mode="lines", name="Consommation [kWh]"))
                    fig2.add_trace(go.Scatter(x=idx, y=pv_kwh, mode="lines", name="Production PV [kWh]"))
                    fig2.add_trace(go.Scatter(x=idx, y=pv_to_load, mode="lines", name="Autocons. PV directe [kWh]"))
    
                    if isinstance(pv_to_batt, list) and len(pv_to_batt) == len(idx):
                        fig2.add_trace(go.Scatter(x=idx, y=pv_to_batt, mode="lines", name="PV → Batterie [kWh]"))
                    if isinstance(batt_to_load_ts, list) and len(batt_to_load_ts) == len(idx):
                        fig2.add_trace(go.Scatter(x=idx, y=batt_to_load_ts, mode="lines", name="Batterie → Load [kWh]"))
    
                    fig2.update_layout(
                        title=f"{bat_nom} — Après batterie (détail temporel)",
                        xaxis_title="Temps",
                        yaxis_title="Énergie [kWh]",
                        height=380,
                        legend_title="Flux",
                    )
                    st.plotly_chart(fig2, use_container_width=True)
    
                    # SOC séparé (optionnel)
                    if isinstance(soc_ts, list) and len(soc_ts) == len(idx):
                        fig_soc = go.Figure()
                        fig_soc.add_trace(go.Scatter(x=idx, y=soc_ts, mode="lines", name="SOC [kWh]"))
                        fig_soc.update_layout(
                            title=f"{bat_nom} — SOC batterie",
                            xaxis_title="Temps",
                            yaxis_title="SOC [kWh]",
                            height=280,
                        )
                        st.plotly_chart(fig_soc, use_container_width=True)
    
                    # ---------- Synthèse trimestrielle (3 mois) ----------
                    st.markdown("### Synthèse par périodes de 3 mois")
    
                    # Agrégations
                    df_load_q = _agg_by_quarter(idx, load_kwh)
                    df_pv_q = _agg_by_quarter(idx, pv_kwh)
                    df_auto_q = _agg_by_quarter(idx, pv_to_load)
    
                    # après: on ajoute batt_to_load à l'autoconsommation totale si dispo
                    if isinstance(batt_to_load_ts, list) and len(batt_to_load_ts) == len(idx):
                        auto_total = pv_to_load + pd.Series(batt_to_load_ts, index=idx)
                    else:
                        auto_total = pv_to_load
    
                    df_auto_tot_q = _agg_by_quarter(idx, auto_total)
    
                    # merge sur Quarter
                    dfQ = df_load_q.rename(columns={"kWh": "Consommation [kWh]"}).merge(
                        df_pv_q.rename(columns={"kWh": "Production PV [kWh]"}), on="Quarter", how="outer"
                    ).merge(
                        df_auto_tot_q.rename(columns={"kWh": "Autocons. totale [kWh]"}), on="Quarter", how="outer"
                    ).fillna(0.0)
    
                    figQ = go.Figure()
                    figQ.add_trace(go.Bar(x=dfQ["Quarter"], y=dfQ["Consommation [kWh]"], name="Consommation [kWh]"))
                    figQ.add_trace(go.Bar(x=dfQ["Quarter"], y=dfQ["Production PV [kWh]"], name="Production PV [kWh]"))
                    figQ.add_trace(go.Bar(x=dfQ["Quarter"], y=dfQ["Autocons. totale [kWh]"], name="Autocons. totale [kWh]"))
                    figQ.update_layout(
                        barmode="group",
                        height=380,
                        xaxis_title="Période",
                        yaxis_title="Énergie [kWh]",
                        legend_title="Flux",
                    )
                    st.plotly_chart(figQ, use_container_width=True)
    
        # ---------- Économie batterie (cashflow barres) ----------
        st.markdown("### Rentabilité batterie — Cashflow cumulé")
    
        stor_econ = (results.get("storage_economics_by_batiment") or {}).get(bat_id) \
                    or (results.get("storage_economics_by_batiment") or {}).get(str(bat_id)) \
                    or []
        # on filtre "proposed"
        econ_b = None
        for it in stor_econ:
            if (it.get("mode") == "proposed") and (it.get("engine") == "bat_li_ion"):
                econ_b = it.get("economics") or {}
                break
    
        if not econ_b:
            st.info("Aucune économie batterie proposée disponible. Relancer les calculs.")
        else:
            years = econ_b.get("years")
            cashflow_cum = econ_b.get("cashflow_cum_CHF")
            if isinstance(years, list) and isinstance(cashflow_cum, list) and len(years) == len(cashflow_cum):
                st.plotly_chart(_bar_cum_cashflow(years, cashflow_cum, f"{bat_nom} — Batterie (cumul)"),
                                use_container_width=True)
    
                payback_year = econ_b.get("payback_year", None)
                benefit_h = float(econ_b.get("benefit_horizon_CHF", econ_b.get("benefit_25_CHF", 0.0)) or 0.0)
    
                capex_b = econ_b.get("capex_total_CHF", None)
                try:
                    capex_b = float(capex_b) if capex_b not in (None, "") else None
                except Exception:
                    capex_b = None
    
                m1, m2, m3 = st.columns(3)
                m1.metric("CAPEX [CHF]", f"{capex_b:,.0f}" if capex_b is not None else "—")
                m2.metric("Rentabilité", f"Année {payback_year}" if payback_year is not None else "—")
                m3.metric("Bénéfice horizon [CHF]", f"{benefit_h:,.0f}")
            else:
                st.info("Économie batterie : years/cashflow_cum_CHF manquants.")
    else:
        st.caption("Aucune batterie proposée détectée dans results pour ce bâtiment.")





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
