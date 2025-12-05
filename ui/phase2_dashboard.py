# -*- coding: utf-8 -*-
"""
Phase 2 - Présentation et analyse du projet SynergyFlex
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go



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
            "Comparaison standard",
            "Analyse conso / prod",
            "Variantes",
        ],
    )

    if page == "Vue d’ensemble":
        show_overview(project)
    elif page == "Diagramme de Sankey":
        show_sankey(project)
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
        # On affiche quand même un “pseudo-bâtiment”
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
                # fallback : première feuille du fichier
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

        # ---------- Graphique area non empilé ----------
        x_raw = df_sheet[time_col]
        
        # 🔹 NEW : gestion propre du temps
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

def show_pv_sankey_from_df(df: pd.DataFrame, titre: str = "Diagramme de Sankey – Panneaux solaires"):
    st.markdown(f"### {titre}")

    if df is None or df.empty:
        st.info("Aucune donnée disponible pour construire le Sankey PV.")
        return

    # On ne propose que les colonnes numériques
    cols_num = df.select_dtypes(include=["number"]).columns.tolist()
    if not cols_num:
        st.info("Aucune colonne numérique trouvée dans les données.")
        return

    # --- Sélection des colonnes / hypothèses par l'utilisateur ---

    col_pv_prod = st.selectbox(
        "Colonne de production PV [kWh]",
        options=cols_num,
        key="pv_prod_col",
    )

    has_pv_auto = st.checkbox(
        "J'ai une colonne pour l'autoconsommation PV",
        value=any("auto" in c.lower() for c in cols_num),
        key="has_pv_auto",
    )
    col_pv_auto = None
    default_selfc = None
    if has_pv_auto:
        col_pv_auto = st.selectbox(
            "Colonne PV autoconsommé [kWh]",
            options=cols_num,
            key="pv_auto_col",
        )
    else:
        default_selfc = st.slider(
            "Taux d'autoconsommation PV estimé [%]",
            min_value=0,
            max_value=100,
            value=55,
            step=1,
            key="pv_selfc_ratio",
        )

    has_pv_inj = st.checkbox(
        "J'ai une colonne pour l'injection PV sur le réseau",
        value=any("inj" in c.lower() or "réseau" in c.lower() for c in cols_num),
        key="has_pv_inj",
    )
    col_pv_inj = None
    if has_pv_inj:
        col_pv_inj = st.selectbox(
            "Colonne PV injecté sur le réseau [kWh]",
            options=cols_num,
            key="pv_inj_col",
        )

    has_conso_elec = st.checkbox(
        "J'ai une colonne pour la consommation électrique du bâtiment (hors PAC)",
        value=any("conso" in c.lower() and "elec" in c.lower() for c in df.columns),
        key="has_conso_elec",
    )
    col_conso_elec = None
    if has_conso_elec:
        col_conso_elec = st.selectbox(
            "Colonne de consommation électrique [kWh]",
            options=cols_num,
            key="conso_elec_col",
        )

    has_import_reseau = st.checkbox(
        "J'ai une colonne pour l'import réseau",
        value=any("import" in c.lower() for c in df.columns),
        key="has_import_reseau",
    )
    col_import_reseau = None
    if has_import_reseau:
        col_import_reseau = st.selectbox(
            "Colonne d'import réseau [kWh]",
            options=cols_num,
            key="import_reseau_col",
        )

    if not col_pv_prod:
        st.warning("Sélectionne au minimum une colonne de production PV.")
        return

    if st.button("Construire le Sankey PV", key="build_sankey_pv"):
        # --- Agrégation des données ---
        prod_pv = float(df[col_pv_prod].astype(float).sum())

        if prod_pv <= 0:
            st.warning("La production PV totale est nulle ou négative, impossible de construire le Sankey.")
            return

        # Autoconsommation
        if col_pv_auto:
            pv_auto = float(df[col_pv_auto].astype(float).sum())
        else:
            pv_auto = prod_pv * (default_selfc / 100.0)

        # Injection
        if col_pv_inj:
            pv_inj = float(df[col_pv_inj].astype(float).sum())
        else:
            pv_inj = max(prod_pv - pv_auto, 0.0)

        # Conso / Import réseau
        conso_elec = float(df[col_conso_elec].astype(float).sum()) if col_conso_elec else 0.0

        if col_import_reseau:
            grid_to_uses = float(df[col_import_reseau].astype(float).sum())
        else:
            # Approximation : ce que la conso ne reçoit pas du PV autoconsommé
            grid_to_uses = max(conso_elec - pv_auto, 0.0)

        # --- Définition des nœuds et flux ---

        labels = [
            "PV",               # 0
            "Usages élec.",     # 1
            "Réseau",           # 2
            "Injection réseau", # 3
        ]

        sources = []
        targets = []
        values = []

        # PV -> Usages
        if pv_auto > 0:
            sources.append(0)
            targets.append(1)
            values.append(pv_auto)

        # PV -> Injection
        if pv_inj > 0:
            sources.append(0)
            targets.append(3)
            values.append(pv_inj)

        # Réseau -> Usages
        if grid_to_uses > 0:
            sources.append(2)
            targets.append(1)
            values.append(grid_to_uses)

        if not values:
            st.info("Les flux PV calculés sont nuls, Sankey non pertinent.")
            return

        fig = go.Figure(
            data=[
                go.Sankey(
                    node=dict(
                        pad=20,
                        thickness=20,
                        line=dict(width=0.5),
                        label=labels,
                    ),
                    link=dict(
                        source=sources,
                        target=targets,
                        value=values,
                    ),
                )
            ]
        )

        fig.update_layout(
            title_text="Flux électriques liés aux panneaux solaires",
            font_size=12,
            margin=dict(l=20, r=20, t=40, b=20),
        )

        st.plotly_chart(fig, use_container_width=True)

# =====================================================================
# AUTRES SECTIONS (placeholders)
# =====================================================================

def _build_sankey_from_flow_block(block: dict):
    """
    Construit une figure Sankey Plotly à partir d'un flow_block standardisé.
    """
    import plotly.graph_objects as go

    nodes = block.get("nodes", [])
    links = block.get("links", [])

    if not nodes or not links:
        return None

    # --- mapping id -> index ---
    id_to_index = {node["id"]: i for i, node in enumerate(nodes)}
    labels = [node.get("label", node["id"]) for node in nodes]

    # --- palette de couleurs par groupe (type énergie) ---
    rgb_by_group = {
        "prod_elec":  (255, 193,   7),   # jaune solaire
        "use_elec":   ( 33, 150, 243),   # bleu usages
        "reseau":     (244,  67,  54),   # rouge réseau
        "stock_elec": (123,  31, 162),   # violet stockage (pour plus tard)
    }
    default_rgb = (158, 158, 158)

    def rgba(rgb, alpha):
        r, g, b = rgb
        return f"rgba({r},{g},{b},{alpha})"

    # Couleurs des nœuds (blocs)
    node_colors = []
    for node in nodes:
        group = node.get("group")
        rgb = rgb_by_group.get(group, default_rgb)
        node_colors.append(rgba(rgb, 0.9))

    # Liens
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
                    pad=35,                     # distance entre blocs
                    thickness=300,               # blocs bien plus larges
                    line=dict(width=1.5, color="rgba(0,0,0,0.4)"),
                    label=labels,
                    color=node_colors,
                    align="center",             # centre les nœuds horizontalement
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
        # 👉 police globale (y compris labels des nœuds)
        font=dict(
            size=40,
            color="rgba(0,0,0,0.85)",
        ),
        margin=dict(l=80, r=80, t=40, b=40),
        height=500,                 # diagramme plus haut
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    return fig







def show_sankey(project: dict):
    st.markdown(
        "<div class='sf-section-title'>Diagramme de Sankey – Producteurs</div>",
        unsafe_allow_html=True,
    )

    results = project.get("results", {})
    flows = results.get("flows", {})
    bat_flows = flows.get("batiments", {})

    if not bat_flows:
        st.info("Aucun flux n’a été calculé. Retourne en Phase 1 et relance les calculs.")
        return

    # --- liste à plat de tous les flow_blocks avec un label lisible ---
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
    selected_label = st.selectbox("Sélectionne un bloc de flux :", labels)
    selected_block = dict(options)[selected_label]

    fig = _build_sankey_from_flow_block(selected_block)
    if fig is None:
        st.info("Impossible de construire le Sankey pour ce bloc (pas de liens valides).")
        return

    st.plotly_chart(fig, use_container_width=True)

    # --- petit tableau récapitulatif des totaux ---
    totals = selected_block.get("totals", {})
    if totals:
        st.markdown("#### Bilan énergétique du bloc sélectionné")
        df_tot = pd.DataFrame([totals]).T
        df_tot.columns = ["Valeur"]
        st.dataframe(df_tot, use_container_width=True)




def show_comparison(project: dict):
    st.markdown("<div class='sf-section-title'>Comparaison standard</div>", unsafe_allow_html=True)
    st.info("Section prévue pour comparer différents scénarios / variantes.")


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
