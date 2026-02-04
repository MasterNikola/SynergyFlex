# -*- coding: utf-8 -*-
"""
Phase 2 - Project presentation and analysis (SynergyFlex)
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
from core.utils.timebase import dt_hours_series
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


# =====================================================================
# MAIN FUNCTION
# =====================================================================

def render_phase2():
    project = st.session_state.get("project")
    if project is None:
        st.warning("No project loaded. Go back to Phase 1 to prepare the data.")
        return

    _inject_css()

    st.title("Phase 2 — Analysis & visualization")
    st.markdown("## Project overview")

    PAGES = {
        'overview': 'Overview',
        'sankey': 'Sankey diagram',
        'economics': '📊 Economics',
        'co2': '🌿 CO₂ (Scope 2)',
        'exergy': '🔥 Exergy (Heating)',
        'pv_standard': 'PV standard comparison',
        'cons_prod': 'Consumption / production',
        'variants': 'Variants',
        # "decision": "🧠 Decision / Comparison",
    }



    page_id = st.sidebar.radio('Sections', list(PAGES.keys()), format_func=lambda k: PAGES[k])

    if page_id == 'overview':
        show_overview(project)
    elif page_id == 'sankey':
        show_sankey(project)
    elif page_id == 'economics':
        show_economics(project)
    elif page_id == 'co2':
        show_co2(project)
    elif page_id == 'exergy':
        show_exergy(project)
    elif page_id == 'pv_standard':
        show_comparison(project)
    elif page_id == 'cons_prod':
        show_phases_analysis(project)
    elif page_id == 'variants':
        show_variantes(project)
    # elif page_id == "decision":
    #     show_decision(project)
    if st.sidebar.button("🔙 Back to Phase 1 (edit data)"):
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
# OVERVIEW
# =====================================================================
def render_energy_label_arrow(title: str, label: str) -> None:
    """
    CECB-like pyramid A..G with left-pointing black marker (Matplotlib only).
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    label = (label or "").strip().upper()
    classes = ["A", "B", "C", "D", "E", "F", "G"]
    if label not in classes:
        st.info(f"{title}: no label available.")
        return

    colors = {
        "A": "#00A651",
        "B": "#6CC04A",
        "C": "#C7D300",
        "D": "#FFE600",
        "E": "#F8B133",
        "F": "#F36F21",
        "G": "#E30613",
    }

    # Geometry
    base_width = 4.5
    width_step = 0.9      # pyramidal growth
    arrow_h = 0.9
    tip = 0.9
    gap = 0.18

    widths = {
        k: base_width + i * width_step
        for i, k in enumerate(classes)
    }

    fig_h = len(classes) * (arrow_h + gap) + 1.2
    fig, ax = plt.subplots(figsize=(5.2, fig_h), dpi=120)

    ax.axis("off")

    # Title INSIDE figure (avoids wrapping issues)
    ax.text(0.0, fig_h - 0.6, title, fontsize=14, fontweight="bold", va="top")

    y0 = fig_h - 1.8

    for i, k in enumerate(classes):
        w = widths[k]
        y = y0 - i * (arrow_h + gap)

        # Right-pointing pyramid arrow
        pts = [
            (0.0, y),
            (w - tip, y),
            (w, y + arrow_h / 2),
            (w - tip, y + arrow_h),
            (0.0, y + arrow_h),
        ]
        poly = Polygon(pts, closed=True, facecolor=colors[k], edgecolor="none", alpha=0.95)
        ax.add_patch(poly)

        ax.text((w - tip) / 2, y + arrow_h / 2, k,
                color="white", fontsize=13, fontweight="bold",
                ha="center", va="center")

        if k == label:
            # Outline
            ax.add_patch(Polygon(pts, closed=True, fill=False,
                                 edgecolor="#111", linewidth=2))

            # LEFT-pointing black marker
            mx = w + 0.6
            mw = 1.8
            mpts = [
                (mx + mw, y + 0.15),
                (mx + 0.4, y + 0.15),
                (mx, y + arrow_h / 2),
                (mx + 0.4, y + arrow_h - 0.15),
                (mx + mw, y + arrow_h - 0.15),
            ]
            mpoly = Polygon(mpts, closed=True, facecolor="#111", edgecolor="none")
            ax.add_patch(mpoly)

            ax.text(mx + mw * 0.65, y + arrow_h / 2, k,
                    color="white", fontsize=12, fontweight="bold",
                    ha="center", va="center")

    ax.set_xlim(0, max(widths.values()) + 3.2)
    ax.set_ylim(0, fig_h)

    fig.tight_layout(pad=0.4)
    st.pyplot(fig, clear_figure=True, use_container_width=True)



def show_overview(project: dict):
    """Project overview: summary + buildings + units + charts."""

    batiments = project.get("batiments", [])
    params = project.get("params", {})

    # On récupère les catégories d’ouvrage depuis le projet,
    # sinon depuis la session (plus robuste).
    categories = project.get("categories_ouvrages") or st.session_state.get("categories_ouvrages", [])

    # nom du bâtiment pour affichage
    if batiments:
        nom_bat = batiments[0].get("nom", "Main building")
    else:
        nom_bat = params.get("batiment_nom", "Main building")

    # --------- Résumé du projet ---------
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Project name", project.get("nom") or project.get("meta", {}).get("Project_name", "—"))
    with col2:
        st.metric("Number of buildings", max(1, len(batiments)))
    with col3:
        st.metric("Number of units", len(categories))

    # --------- Buildings ---------
    st.markdown("<div class='sf-section-title'>Buildings</div>", unsafe_allow_html=True)
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

    # --------- Units (tableau) ---------
    st.markdown("<div class='sf-section-title'>Units</div>", unsafe_allow_html=True)
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
        st.info("No unit defined in the project.")
        return

    # --------- Graphiques par ouvrage ---------
    st.markdown("<div class='sf-section-title'>Charts by unit</div>", unsafe_allow_html=True)

    sheets_dict = st.session_state.get("excel_sheets")
    fallback_df = None
    if not sheets_dict:
        fallback_df = _get_fallback_dataframe_from_project(project)
        
    for idx, cat in enumerate(categories):
        type_ouv = cat.get("type", "Unit")
        nom_ouv = cat.get("nom") or f"Unit {idx+1}"
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
                <b>Excel sheet :</b> {sheet_name or "n/a"}
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
            st.info("Raw data not found for this unit.")
            st.markdown("---")
            continue

        if not time_col or time_col not in df_sheet.columns:
            st.info("Time column not defined or not found for this unit.")
            st.markdown("---")
            continue

        conso_elec_valid = conso_elec_col if conso_elec_col in df_sheet.columns else None
        conso_th_valid = conso_th_col if conso_th_col in df_sheet.columns else None

        if not conso_elec_valid and not conso_th_valid:
            st.info("No valid electric/thermal consumption column found for this unit.")
            st.markdown("---")
            continue

        # ---------- Graphique area ----------
        x_raw = df_sheet[time_col]

        if time_col.lower().startswith("mois") and "Annee" in df_sheet.columns:
            x = pd.to_datetime(
                df_sheet["Annee"].astype(int).astype(str) + "-" +
                df_sheet["Month"].astype(int).astype(str) + "-01",
                errors="coerce"
            )
        else:
            try:
                x = pd.to_datetime(x_raw)
            except Exception:
                x = x_raw


        fig = go.Figure()

        if conso_elec_valid:
            y = pd.to_numeric(df_sheet[conso_elec_valid], errors="coerce")
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    name=f"Electric consumption — {conso_elec_valid}",
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
                    name=f"Thermal consumption — {conso_th_valid}",
                    fill="tozeroy",
                    line=dict(width=1.5),
                )
            )

        fig.update_layout(
            height=320,
            margin=dict(l=40, r=20, t=10, b=40),
            xaxis_title="Time",
            yaxis_title="Power / Energy",
            hovermode="x unified",
        )

        st.plotly_chart(fig, use_container_width=True)
        st.markdown("---")
        
        # -------------------------------------------------------------
        # SIA / CECB-like thermal envelope label (baseline measured)
        # Pure display: strict read from results["sia"]
        # -------------------------------------------------------------
        sia = (project.get("results") or {}).get("sia") or {}
        thermal = (sia.get("thermal") or {})
        by_bat = (thermal.get("by_batiment") or {})
        
        st.markdown("### Building envelope efficiency (thermal)")
        
        if not by_bat:
            st.info("No SIA thermal label found in results (results['sia']['thermal']['by_batiment'] missing).")
        else:
            # Helper: try to map bat_id -> building name (best-effort, no hardcoding)
            batiments = project.get("batiments", []) or []
        
            def _bat_name_from_id(bid: str) -> str:
                # If bat_id is an int-like string and matches batiments list index -> name
                try:
                    i = int(str(bid))
                    if 0 <= i < len(batiments):
                        return str(batiments[i].get("nom") or f"Building {i}")
                except Exception:
                    pass
                return f"Building {bid}"
        
            # Render one arrow per batiment_id + key KPIs (defendable)
            for bat_id, rec in by_bat.items():
                bat_name = _bat_name_from_id(bat_id)
        
                if not isinstance(rec, dict):
                    st.warning(f"{bat_name}: invalid SIA record format.")
                    continue
        
                ok = bool(rec.get("ok", False))
        
                left, right = st.columns([1.05, 1.2])
        
                with left:
                    # Title includes building name; arrow expects A..G
                    label = rec.get("class", None)
                    render_energy_label_arrow(f"{bat_name}", label)

        
                with right:
                    if not ok:
                        err = rec.get("error") or "SIA label could not be computed."
                        st.warning(f"{bat_name}: {err}")
                    else:
                        r_pct = rec.get("R_pct", None)
                        q_spec = rec.get("Qh_measured_specific_kWh_m2", None)
                        q_lim = rec.get("Qh_li_kWh_m2", None)
                        ratio = rec.get("ath_over_sre", None)
                        sre_used = rec.get("sre_m2_used", None)
                        q_total = None
                        if q_spec is not None and sre_used not in (None, 0):
                            q_total = float(q_spec) * float(sre_used)

        
                        c1, c2 = st.columns(2)
                        with c1:
                            if r_pct is not None:
                                st.metric("R (measured / limit)", f"{float(r_pct):.1f} %")
                            if q_spec is not None:
                                st.metric("Measured heat demand", f"{float(q_spec):.1f} kWh/m²·a")
                            if q_total is not None:
                                st.metric("Measured heat demand (total)", f"{q_total:,.0f} kWh/y")
                        with c2:
                            if q_lim is not None:
                                st.metric("Limit Qh,li", f"{float(q_lim):.1f} kWh/m²·a")
                            if ratio is not None:
                                st.metric("Ath / SRE", f"{float(ratio):.2f} [-]")
        
                st.markdown("---")
                
        # -------------------------------------------------------------
        # SIA-like ELECTRICITY label (baseline measured)
        # Pure display: strict read from results["sia"]
        # -------------------------------------------------------------
        elec = (sia.get("electricity") or {})
        by_bat_el = (elec.get("by_batiment") or {})

        st.markdown("### Electricity efficiency (SIA-like, baseline measured)")

        if not by_bat_el:
            st.info("No SIA electricity label found in results (results['sia']['electricity']['by_batiment'] missing).")
        else:
            for bat_id, rec in by_bat_el.items():
                bat_name = _bat_name_from_id(bat_id)

                if not isinstance(rec, dict):
                    st.warning(f"{bat_name}: invalid SIA electricity record format.")
                    continue

                ok = bool(rec.get("ok", False))

                left, right = st.columns([1.05, 1.2])

                with left:
                    label = rec.get("class", None)
                    render_energy_label_arrow(f"{bat_name}", label)

                with right:
                    if not ok:
                        err = rec.get("error") or "Electricity label could not be computed."
                        st.warning(f"{bat_name}: {err}")
                    else:
                        r_pct = rec.get("R_pct", None)
                        e_spec = rec.get("E_el_measured_specific_kWh_m2a", None)
                        e_lim = rec.get("E_el_limit_kWh_m2a", None)
                        e_total = rec.get("E_el_measured_kWh_y", None)

                        c1, c2 = st.columns(2)
                        with c1:
                            if r_pct is not None:
                                st.metric("R (measured / limit)", f"{float(r_pct):.1f} %")
                            if e_spec is not None:
                                st.metric("Measured electricity", f"{float(e_spec):.1f} kWh/m²·a")
                            if e_total is not None:
                                st.metric("Measured electricity (total)", f"{float(e_total):,.0f} kWh/a")

                        with c2:
                            if e_lim is not None:
                                st.metric("Reference electricity demand", f"{float(e_lim):.1f} kWh/m²·a")
                            ref = rec.get("ref") or {}
                            mj = ref.get("E_el_ref_MJ_m2a")
                            if mj is not None:
                                st.caption(f"Ref source: {float(mj):.0f} MJ/m²·a (→ /3.6)")

                st.markdown("---")

        st.markdown("### Direct CO₂ emissions (CECB-like)")

        co2 = (project.get("results") or {}).get("co2") or {}
        by_bat = (co2.get("by_batiment") or {})
        
        if not by_bat:
            st.info("No CO₂ results found in results['co2']['by_batiment'].")
        else:
            # mêmes helpers bat_id -> nom que tu utilises déjà
            for bat_id, rec in by_bat.items():
                scope1 = (rec or {}).get("scope1") or {}
                lab = (scope1.get("direct_cecb_label") or {})
        
                left, right = st.columns([1.05, 1.2])
                with left:
                    render_energy_label_arrow(f"Building {bat_id}", lab.get("class"))
        
                with right:
                    if not lab.get("ok"):
                        st.warning(lab.get("note") or "CO₂ label not available.")
                    else:
                        v = lab.get("direct_kgCO2e_per_m2a", None)
                        sre_used = lab.get("sre_m2_used", None)
                        cls = lab.get("class", None)
                        thr = lab.get("thresholds_kgCO2_per_m2a", None) or {}
                
                        # total kgCO2e/y (display-only)
                        total_co2 = None
                        if v is not None and sre_used not in (None, 0):
                            total_co2 = float(v) * float(sre_used)
                
                        # R% measured / class upper limit (display-only)
                        r_pct = None
                        lim_max = None
                        if cls in thr:
                            bounds = thr.get(cls)
                            if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
                                lim_max = bounds[1]  # None for G
                        if v is not None and lim_max not in (None, 0):
                            r_pct = 100.0 * float(v) / float(lim_max)
                
                        c1, c2 = st.columns(2)
                        with c1:
                            if v is not None:
                                st.metric("Direct CO₂ (specific)", f"{float(v):.2f} kgCO₂/m²·a")
                            if total_co2 is not None:
                                st.metric("Direct CO₂ (total)", f"{total_co2:,.0f} kgCO₂e/y")
                                
                
                        with c2:
                            fcor = lab.get("f_cor", None)
                            if fcor is not None:
                                st.metric("f_cor", f"{float(fcor):.4f} [-]")
                
                            th = lab.get("theta_e_avg_C")
                            if th is not None:
                                st.caption(f"θe,avg = {float(th):.2f} °C")

        
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

    title = block.get("name") or f"Flows {block.get('type', '')}".strip()
    fig.update_layout(
        title_text=title,
        font=dict(size=14, color="rgba(0,0,0,0.85)"),
        margin=dict(l=80, r=80, t=40, b=40),
        height=500,
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    return fig

def _is_proposed_block(b: dict) -> bool:
    """
    Detect if a flow_block belongs to a proposed/variant ("after") scenario.
    Used to exclude proposed blocks from measured-only views.
    """
    # 1) direct tag
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

    # 4) name fallback
    name = b.get("name")
    if isinstance(name, str) and ("propos" in name.lower() or "after" in name.lower()):
        return True

    return False


def show_sankey(project: dict):
    import streamlit as st
    import pandas as pd

    st.markdown("## Sankey diagram — Real building energy flows")
    st.caption(
        "The diagram represents the building’s real physical state. "
        "If storage exists, it is automatically included."
    )

    results = project.get("results", {}) or {}
    flows_by_bat = results.get("flows", {}).get("batiments", {}) or {}

    if not flows_by_bat:
        st.info("No computed flows. Go back to Phase 1 and re-run the calculations.")
        return

    # --- Sélection bâtiment uniquement
    batiments = project.get("batiments", []) or []
    bat_options = []
    for i, bat in enumerate(batiments):
        bat_id = bat.get("id") or bat.get("batiment_id") or i
        bat_nom = bat.get("nom") or f"Building {i+1}"
        bat_options.append((bat_id, bat_nom))

    if not bat_options:
        bat_options = [(k, f"Building {k}") for k in flows_by_bat.keys()]

    labels = [f"{bid} – {bnom}" for bid, bnom in bat_options]
    idx = st.selectbox("Building", range(len(bat_options)), format_func=lambda i: labels[i])
    bat_id, bat_nom = bat_options[idx]

    blocks = flows_by_bat.get(bat_id) or flows_by_bat.get(str(bat_id)) or []
    if not blocks:
        st.info("No flow_block for this building.")
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
        st.warning("No global block found. Unable to display the Sankey diagram.")
        return

    # --- Option: include thermal branch in the global Sankey (overview)
    include_thermal_in_global = st.checkbox(
        "Include thermal branch in global Sankey",
        value=True,
        help="Adds thermal flow blocks (e.g., oil boiler) to the global diagram. No recalculation: results-only."
    )
    
    # Identify thermal blocks (measured/real only)
    thermal_blocks_measured = [
        b for b in blocks
        if not _is_proposed_block(b)
        and isinstance(b.get("type"), str)
        and (b.get("type").startswith("thermal_") or b.get("type") == "demand_th")
    ]
    
    # Build a combined Sankey block (electric global + thermal blocks) for overview
    block_to_plot = global_block
    if include_thermal_in_global and thermal_blocks_measured:
        combined_nodes = []
        combined_links = []
        seen_nodes = set()
    
        # Start with electric global block
        for n in (global_block.get("nodes") or []):
            nid = n.get("id")
            if nid is None or nid in seen_nodes:
                continue
            combined_nodes.append(n)
            seen_nodes.add(nid)
        combined_links.extend(global_block.get("links") or [])
    
        # Append thermal blocks (nodes+links)
        for tb in thermal_blocks_measured:
            for n in (tb.get("nodes") or []):
                nid = n.get("id")
                if nid is None or nid in seen_nodes:
                    continue
                combined_nodes.append(n)
                seen_nodes.add(nid)
            combined_links.extend(tb.get("links") or [])
    
        block_to_plot = {
            "name": f"Energy balance (electric + thermal) — {bat_nom}",
            "type": "energy_global_overview",
            "meta": {
                "batiment_id": bat_id,
                "batiment_nom": bat_nom,
                "includes": ["electric_global", "thermal_blocks_measured"],
            },
            "nodes": combined_nodes,
            "links": combined_links,
            # totals handled separately in expander below (no recomputation here)
            "totals": global_block.get("totals", {}) or {},
        }
    
    fig = _build_sankey_from_flow_block(block_to_plot)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)

    # --- Totals (electric + thermal)
    elec_totals = global_block.get("totals", {}) or {}

    # Identify thermal blocks (measured/real only)
    thermal_blocks_measured = [
        b for b in blocks
        if not _is_proposed_block(b)
        and isinstance(b.get("type"), str)
        and (b.get("type").startswith("thermal_") or b.get("type") == "demand_th")
    ]

    # Build a thermal balance summary from existing totals (NO recomputation)
    demand_block = next((b for b in thermal_blocks_measured if b.get("type") == "demand_th"), None)
    supply_blocks = [b for b in thermal_blocks_measured if isinstance(b.get("type"), str) and b.get("type").startswith("thermal_")]

    demand_totals = (demand_block.get("totals", {}) or {}) if demand_block else {}
    demand_th_kWh = demand_totals.get("demand_th_kWh", None)
    residual_th_kWh = demand_totals.get("residual_th_kWh", None)

    def _sum_totals(blocks_list, key: str) -> float:
        s = 0.0
        for bb in blocks_list:
            tt = bb.get("totals", {}) or {}
            v = tt.get(key, 0.0)
            try:
                s += float(v)
            except Exception:
                pass
        return float(s)

    thermal_summary = {}
    if demand_th_kWh is not None:
        thermal_summary["thermal_demand_th_kWh"] = demand_th_kWh
    if residual_th_kWh is not None:
        thermal_summary["thermal_residual_th_kWh"] = residual_th_kWh

    # Supply-side totals (sum over thermal_* blocks)
    if supply_blocks:
        thermal_summary["thermal_heat_out_th_kWh"] = _sum_totals(supply_blocks, "heat_out_th_kWh")
        thermal_summary["thermal_fuel_in_kWh"] = _sum_totals(supply_blocks, "fuel_in_kWh")
        thermal_summary["thermal_losses_kWh"] = _sum_totals(supply_blocks, "losses_kWh")
        thermal_summary["thermal_unmet_th_kWh"] = _sum_totals(supply_blocks, "unmet_th_kWh")

        thermal_summary["thermal_fuel_cost_CHF"] = _sum_totals(supply_blocks, "fuel_cost_CHF")
        thermal_summary["thermal_scope1_kgCO2e"] = _sum_totals(supply_blocks, "fuel_scope1_kgCO2e")

        # Standardized economics (if present in totals)
        thermal_summary["thermal_capex_total_CHF"] = _sum_totals(supply_blocks, "capex_total_CHF")
        thermal_summary["thermal_opex_annual_CHF"] = _sum_totals(supply_blocks, "opex_annual_CHF")

    # Show totals
    if elec_totals or thermal_summary:
        with st.expander("Energy balance totals"):
            c1, c2 = st.columns(2)

            with c1:
                st.markdown("**Electric balance**")
                if elec_totals:
                    df_e = pd.DataFrame([elec_totals]).T
                    df_e.columns = ["Value"]
                    st.dataframe(df_e, use_container_width=True)
                else:
                    st.info("No electric totals available for this building.")

            with c2:
                st.markdown("**Thermal balance**")
                if thermal_summary:
                    df_t = pd.DataFrame([thermal_summary]).T
                    df_t.columns = ["Value"]
                    st.dataframe(df_t, use_container_width=True)
                else:
                    st.info("No thermal totals available for this building.")

    
       
    # --- Storage details (EXISTANT / MESURÉ uniquement)
    storage_blocks = [
        b for b in blocks
        if b.get("type") in ("battery_elec", "storage_elec", "storage_th")
        and not _is_proposed_block(b)
    ]
    
    if storage_blocks:
        st.markdown("---")
        st.subheader("Storage details")
        st.caption("Existing storage subsystems (measured view only).")
    
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
        st.info("No economic data available.")
        return

    # -------------------------
    # Helpers
    # -------------------------
    def _bat_name_from_id(bat_id):
        try:
            if isinstance(bat_id, int) and bat_id < len(batiments):
                return batiments[bat_id].get("nom") or f"Building {bat_id}"
        except Exception:
            pass
        return f"Building {bat_id}"

    def _cashflow_bar(years, cum, height=380):
        df_cf = pd.DataFrame({"Year": years, "Cumulative [CHF]": cum})
        vals = df_cf["Cumulative [CHF]"].tolist()
        colors = ["rgba(220,53,69,0.85)" if v < 0 else "rgba(40,167,69,0.85)" for v in vals]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df_cf["Year"], y=df_cf["Cumulative [CHF]"], marker_color=colors))
        fig.update_layout(xaxis_title="Years", yaxis_title="CHF", height=height, margin=dict(l=10, r=10, t=20, b=10))
        return fig
    
    def _pretty_machine_type(bat_id, mtype: str) -> str:
        """Return a user-friendly label for a machine type using flow_blocks metadata (results-only)."""
        try:
            flows_by_bat = (results.get("flows", {}) or {}).get("batiments", {}) or {}
            blocks = flows_by_bat.get(bat_id, []) or flows_by_bat.get(str(bat_id), []) or []
            for b in blocks:
                if b.get("type") == mtype:
                    meta = b.get("meta") or {}
                    techno = meta.get("techno")
                    if isinstance(techno, str) and techno.strip():
                        return techno.strip()
                    name = b.get("name")
                    if isinstance(name, str) and name.strip():
                        return name.strip()
                    break
        except Exception:
            pass
    
        # Fallback: keep the raw id if nothing better is available
        return str(mtype)

    
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
                "Building": nom_bat,
                "CAPEX machines [CHF]": capex,
                "OPEX machines [CHF/an]": opex,
                "Production [kWh/yr]": prod,
                "LCOE global [CHF/kWh]": lcoe,
            })

            for mtype, stats in by_type.items():
                mtype_str = str(mtype)

                # Carrier classification (Phase 2 only, based on type id)
                if mtype_str.startswith("thermal_") or mtype_str == "demand_th":
                    carrier = "Thermal"
                elif mtype_str.startswith("pv") or mtype_str == "demand_elec" or mtype_str.startswith("battery"):
                    carrier = "Electric"
                else:
                    carrier = "Other"

                detail_rows.append({
                    "Building": nom_bat,
                    "Type": mtype_str,
                    "Carrier": carrier,
                    "CAPEX [CHF]": float(stats.get("capex_total_CHF", 0.0) or 0.0),
                    "OPEX [CHF/an]": float(stats.get("opex_annual_CHF", 0.0) or 0.0),
                    "Production [kWh/yr]": float(stats.get("production_machine_kWh", 0.0) or 0.0),
                    "LCOE [CHF/kWh]": float(stats.get("lcoe_machine_CHF_kWh", 0.0) or 0.0),
                })


        df = pd.DataFrame(rows)

        if not df.empty:
            g1, g2 = st.columns(2)

            # If we have detail_rows, we can do stacked CAPEX and carrier LCOE.
            if detail_rows:
                df_detail = pd.DataFrame(detail_rows).copy()

                # Ensure numeric
                for col in ["CAPEX [CHF]", "OPEX [CHF/an]", "Production [kWh/yr]"]:
                    if col in df_detail.columns:
                        df_detail[col] = pd.to_numeric(df_detail[col], errors="coerce").fillna(0.0)

                # ---------- (1) CAPEX stacked by technology ----------
                capex_stack = (
                    df_detail.groupby(["Building", "Type"], as_index=False)["CAPEX [CHF]"]
                    .sum()
                )

                with g1:
                    fig_capex = px.bar(
                        capex_stack,
                        x="Building",
                        y="CAPEX [CHF]",
                        color="Type",
                        barmode="stack",
                        text_auto=True,
                    )
                    fig_capex.update_layout(
                        title="Machine CAPEX by building (stacked by technology)",
                        height=360,
                        margin=dict(l=10, r=10, t=40, b=10),
                        legend_title="Technology",
                    )
                    st.plotly_chart(fig_capex, use_container_width=True)

                # ---------- (2) LCOE split: Electric vs Thermal ----------
                # LCOE_carrier = (sum CAPEX + sum OPEX) / sum Production  (per building, per carrier)
                lcoe_rows = []
                lcoe_rows = []
                for (bld, car), grp in df_detail.groupby(["Building", "Carrier"]):
                    # Ensure numeric
                    prod = pd.to_numeric(grp["Production [kWh/yr]"], errors="coerce").fillna(0.0)
                    lcoe = pd.to_numeric(grp["LCOE [CHF/kWh]"], errors="coerce").fillna(0.0)
                
                    prod_sum = float(prod.sum())
                    if prod_sum > 0:
                        lcoe_val = float((lcoe * prod).sum() / prod_sum)
                    else:
                        lcoe_val = 0.0
                
                    lcoe_rows.append({
                        "Building": bld,
                        "Carrier": car,
                        "LCOE [CHF/kWh]": lcoe_val,
                    })


                df_lcoe_split = pd.DataFrame(lcoe_rows)
                # keep only Electric/Thermal (drop Other to avoid confusing MVP)
                df_lcoe_split = df_lcoe_split[df_lcoe_split["Carrier"].isin(["Electric", "Thermal"])]

                with g2:
                    fig_lcoe = px.bar(
                        df_lcoe_split,
                        x="Building",
                        y="LCOE [CHF/kWh]",
                        color="Carrier",
                        barmode="group",
                        text_auto=True,
                    )
                    fig_lcoe.update_layout(
                        title="LCOE by building (Electric vs Thermal)",
                        height=360,
                        margin=dict(l=10, r=10, t=40, b=10),
                        legend_title="Carrier",
                    )
                    st.plotly_chart(fig_lcoe, use_container_width=True)

            else:
                # Fallback: old behavior
                with g1:
                    fig_capex = px.bar(df, x="Building", y="CAPEX machines [CHF]", text_auto=True)
                    fig_capex.update_layout(title="Machine CAPEX by building", height=360, margin=dict(l=10, r=10, t=40, b=10))
                    st.plotly_chart(fig_capex, use_container_width=True)

                with g2:
                    fig_lcoe = px.bar(df, x="Building", y="LCOE global [CHF/kWh]", text_auto=True)
                    fig_lcoe.update_layout(title="Global LCOE by building", height=360, margin=dict(l=10, r=10, t=40, b=10))
                    st.plotly_chart(fig_lcoe, use_container_width=True)


        # Détail par type (si dispo)
        if detail_rows:
            st.markdown("---")
            st.markdown("#### Breakdown by machine type")
            df_detail = pd.DataFrame(detail_rows)
            st.dataframe(df_detail, use_container_width=True, hide_index=True)
        
            # Building selector must be defined from existing data (no assumptions)
            bat_names = sorted(df_detail["Building"].dropna().unique().tolist())
        
            if not bat_names:
                st.info("No building available for technology breakdown.")
            else:
                selected_bat = st.selectbox(
                    "Building (breakdown by technology)",
                    bat_names,
                    key="econ_bat_detail",
                )
                df_bat = df_detail[df_detail["Building"] == selected_bat].copy()
        
                if not df_bat.empty:
                    # Ensure numeric (robust)
                    for col in ["CAPEX [CHF]", "OPEX [CHF/an]", "Production [kWh/yr]", "LCOE [CHF/kWh]"]:
                        if col in df_bat.columns:
                            df_bat[col] = pd.to_numeric(df_bat[col], errors="coerce").fillna(0.0)
        
                    st.subheader("Breakdown by technology")
        
                    c1, c2, c3 = st.columns(3)
        
                    with c1:
                        fig_capex = px.pie(
                            df_bat, names="Type", values="CAPEX [CHF]",
                            title=f"CAPEX split — {selected_bat}"
                        )
                        fig_capex.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10))
                        st.plotly_chart(fig_capex, use_container_width=True)
        
                    with c2:
                        fig_opex = px.pie(
                            df_bat, names="Type", values="OPEX [CHF/an]",
                            title=f"OPEX split — {selected_bat}"
                        )
                        fig_opex.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10))
                        st.plotly_chart(fig_opex, use_container_width=True)
        
                    with c3:
                        fig_prod = px.pie(
                            df_bat, names="Type", values="Production [kWh/yr]",
                            title=f"Production split — {selected_bat}"
                        )
                        fig_prod.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10))
                        st.plotly_chart(fig_prod, use_container_width=True)
        
                    

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
                    "Building": _bat_name_from_id(int(bat_id)) if str(bat_id).isdigit() else f"Building {bat_id}",
                    "Unit": b.get("ouvrage_nom"),
                    "Techno": b.get("techno"),
                    "Mode": b.get("mode"),
                    "Capacity [kWh]": b.get("capacity_kwh"),
                    "Total CAPEX [CHF]": capex0,
                    "Payback [yr]": payback,
                    "Horizon benefit [CHF]": benefit,
                    "_years": years,
                    "_cum": cum,
                })

        if not batt_rows:
            st.info("No battery economics available.")
            return

        df_batt = pd.DataFrame(batt_rows)

        # tableau compact (visible)
        st.markdown("#### Batteries (cashflow)")
        visible_cols = ["Building", "Unit", "Techno", "Mode", "Capacity [kWh]", "Total CAPEX [CHF]", "Payback [yr]", "Horizon benefit [CHF]"]
        st.dataframe(df_batt[visible_cols], use_container_width=True, hide_index=True)

        # Sélecteur “1 batterie”
        labels = [
            f"{r['Building']} | {r['Unit']} | {r['Techno']} | {r['Capacity [kWh]']} kWh"
            for _, r in df_batt.iterrows()
        ]
        sel = st.selectbox("Select a battery", options=list(range(len(labels))), format_func=lambda i: labels[i], key="batt_select")

        row = df_batt.iloc[int(sel)]
        years = row["_years"]
        cum = row["_cum"]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric('Payback', f"Year {int(row['Payback [yr]'])}" if pd.notna(row['Payback [yr]']) else '—')
        m2.metric("Horizon benefit [CHF]", f"{float(row['Horizon benefit [CHF]']):,.0f}" if pd.notna(row["Horizon benefit [CHF]"]) else "—")
        m3.metric("Capacity [kWh]", f"{float(row['Capacity [kWh]']):.1f}" if pd.notna(row["Capacity [kWh]"]) else "—")
        m4.metric("Total CAPEX [CHF]", f"{float(row['Total CAPEX [CHF]']):,.0f}" if pd.notna(row["Total CAPEX [CHF]"]) else "—")

        if isinstance(years, list) and isinstance(cum, list) and len(years) == len(cum) and len(years) > 0:
            st.plotly_chart(_cashflow_bar(years, cum, height=420), use_container_width=True)
        else:
            st.info("Incomplete battery cashflow (years / cashflow_cum_CHF).")





def show_co2(project: dict):
    import streamlit as st
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go

    results = project.get("results", {}) or {}
    co2 = results.get("co2") or {}

    st.markdown("## CO₂ — Results")

    if not co2:
        st.warning("CO₂ results not available. Run calculations in Phase 1.")
        return

    # -----------------------
    # Meta / traceability (read-only)
    # -----------------------
    meta = co2.get("meta") or {}
    factors = meta.get("factors") or {}
    grid_factor = factors.get("grid_import_factor_kg_per_kWh")

    by_bat = co2.get("by_batiment") or {}
    ts_root = co2.get("timeseries_by_batiment") or {}

    tab_g, tab_th, tab_el = st.tabs(["🌍 Global", "🔥 Thermal (detailed)", "⚡ Electric (detailed)"])

    # =====================================================================
    # TAB 1 — GLOBAL
    # =====================================================================
    with tab_g:
        st.markdown("### Global CO₂ footprint (available scopes)")

        with st.expander("Assumptions / method (traceability)", expanded=True):
            st.write({
                "method": meta.get("method"),
                "export_credit": meta.get("export_credit"),
                "grid_factor_kgCO2e_per_kWh": grid_factor,
                "factor_source": factors.get("source"),
                "notes": meta.get("notes"),
            })
            st.caption(
                "This tab aggregates what is available in results['co2'] (no recomputation). "
                "Thermal Scope 1 is shown only if it exists in results."
            )

        # Aggregate global KPIs from existing core results (no recompute)
        glob_s2 = ((co2.get("global") or {}).get("scope2") or {}).get("totals") or {}
        net_kg_global = glob_s2.get("net_kgCO2e")
        imp_kWh_global = glob_s2.get("grid_import_kWh")
        exp_kWh_global = glob_s2.get("grid_export_kWh")

        # Sum system_total_kgCO2e across buildings when available (more meaningful than net_kg only)
        sys_total_sum = 0.0
        sys_total_has = False
        comp_sum = {}

        for bid, bdata in (by_bat or {}).items():
            s2 = (bdata.get("scope2") or {})
            totals = s2.get("totals") or {}

            v = totals.get("system_total_kgCO2e")
            if v is not None:
                try:
                    sys_total_sum += float(v)
                    sys_total_has = True
                except Exception:
                    pass

            comp = totals.get("components_kgCO2e") or (s2.get("components_kgCO2e") or {})
            if isinstance(comp, dict):
                for k, vv in comp.items():
                    try:
                        comp_sum[k] = float(comp_sum.get(k, 0.0)) + float(vv)
                    except Exception:
                        pass
        
        # ---- Scope 1 global (thermal combustion) ----
        scope1_global_totals = ((co2.get("global") or {}).get("scope1") or {}).get("totals") or {}
        scope1_global_kg = scope1_global_totals.get("fuel_scope1_kgCO2e")
        
        # Fallback if global totals not present: sum by building scope1
        if scope1_global_kg is None:
            s = 0.0
            found = False
            for bid, bdata in (by_bat or {}).items():
                s1 = (bdata.get("scope1") or {})
                t1 = (s1.get("totals") or {})
                v = t1.get("fuel_scope1_kgCO2e")
                if v is not None:
                    try:
                        s += float(v)
                        found = True
                    except Exception:
                        pass
            scope1_global_kg = float(s) if found else None
        
        has_scope1 = scope1_global_kg is not None

        
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            # "Global" = sum of available scopes (no recompute)
            scope2_val = float(sys_total_sum) if sys_total_has else (float(net_kg_global) if net_kg_global is not None else None)
        
            if scope2_val is None and not has_scope1:
                st.metric("Total CO₂ (available scopes)", "—")
            else:
                total_available = (float(scope1_global_kg) if has_scope1 else 0.0) + (float(scope2_val) if scope2_val is not None else 0.0)
                st.metric("Total CO₂ (available scopes)", f"{total_available/1000:.2f} tCO₂e/year")

        with c2:
            st.metric("Annual grid import (all buildings)", "—" if imp_kWh_global is None else f"{imp_kWh_global:,.0f} kWh")
        with c3:
            st.metric("Annual PV export (all buildings)", "—" if exp_kWh_global is None else f"{exp_kWh_global:,.0f} kWh")
        with c4:
            st.metric("Grid factor", "—" if grid_factor is None else f"{float(grid_factor):.3f} kgCO₂e/kWh")

        st.markdown("---")

        # Scope view (future-proof)
        st.markdown("#### Breakdown by scope (available)")
        scope2_val = float(sys_total_sum) if sys_total_has else (float(net_kg_global) if net_kg_global is not None else 0.0)

        scope_rows = [
            {
                "Scope": "Scope 1 (thermal combustion)",
                "kgCO2e/year": float(scope1_global_kg) if has_scope1 else 0.0,
                "Status": "Available" if has_scope1 else "Not available yet",
            },
            {
                "Scope": "Scope 2 (electricity system)",
                "kgCO2e/year": float(scope2_val),
                "Status": "Available",
            },
            {
                "Scope": "Scope 3 (embodied)",
                "kgCO2e/year": 0.0,
                "Status": "Included in components (current MVP)",
            },
        ]

        df_scope = pd.DataFrame(scope_rows)
        st.dataframe(df_scope, use_container_width=True, hide_index=True)

        st.markdown("---")

        # Component breakdown aggregated across buildings (signed values are already in components_kgCO2e)
        st.markdown("#### System components (aggregated)")
        if comp_sum:
            dfc = pd.DataFrame([{"Component": k, "kgCO2e": v} for k, v in comp_sum.items()])
            dfc = dfc.sort_values("kgCO2e", ascending=False)

            fig_c = px.bar(
                dfc,
                x="Component",
                y="kgCO2e",
                text=dfc["kgCO2e"].map(lambda x: f"{x:,.0f}"),
            )
            fig_c.update_traces(textposition="outside")
            fig_c.update_layout(
                showlegend=False,
                yaxis_title="kgCO₂e per year (signed)",
                xaxis_title="",
            )
            st.plotly_chart(fig_c, use_container_width=True)
            st.caption("Signed convention comes from core results: benefits are negative, costs are positive.")
        else:
            st.info("No component breakdown available at global level.")
        
        if has_scope1:
            comp_sum["fuel_combustion_scope1_kgCO2e"] = float(comp_sum.get("fuel_combustion_scope1_kgCO2e", 0.0)) + float(scope1_global_kg)

    # =====================================================================
    # TAB 2 — THERMAL (DETAILED)
    # =====================================================================
    with tab_th:
        st.markdown("### Thermal CO₂ (Scope 1) — detailed view")

        # MVP guard: thermal scope not integrated yet in co2 results
        has_scope1 = False
        for bid, bdata in (by_bat or {}).items():
            if isinstance(bdata, dict) and ("scope1" in bdata):
                has_scope1 = True
                break

        if not has_scope1:
            st.info(
                "Thermal CO₂ is not available yet in results['co2'].\n\n"
                "Current MVP computes electricity Scope 2 (grid/PV/battery). "
                "Thermal Scope 1 (fuel combustion) will be added next."
            )
        else:
            # Future-proof placeholder structure: show totals/components if scope1 exists
            bat_ids = list(by_bat.keys())
            bat_id = st.selectbox("Select building", bat_ids, index=0, key="co2_th_bat")
            b = (by_bat.get(bat_id) or {}).get("scope1") or {}
            totals = b.get("totals") or {}
            comp = totals.get("components_kgCO2e") or (b.get("components_kgCO2e") or {})

            st.markdown("#### Thermal totals (Scope 1)")
            st.write(totals)

            st.markdown("#### Thermal components (Scope 1)")
            if isinstance(comp, dict) and comp:
                dfc = pd.DataFrame([{"Component": k, "kgCO2e": v} for k, v in comp.items()])
                fig = px.bar(dfc, x="Component", y="kgCO2e", text=dfc["kgCO2e"].map(lambda x: f"{x:,.0f}"))
                fig.update_traces(textposition="outside")
                fig.update_layout(showlegend=False, yaxis_title="kgCO₂e per year", xaxis_title="")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.caption("No thermal components available.")

    # =====================================================================
    # TAB 3 — ELECTRIC (DETAILED)  (your current content, reorganized)
    # =====================================================================
    with tab_el:
        st.markdown("### Electricity CO₂ — Scope 2 + PV/battery factors")

        if not by_bat:
            st.info("No building-level CO₂ breakdown available.")
            return

        bat_ids = list(by_bat.keys())
        bat_id = st.selectbox("Select building", bat_ids, index=0, key="co2_el_bat")
        b = (by_bat.get(bat_id) or {}).get("scope2") or {}
        totals = b.get("totals") or {}
        comp = totals.get("components_kgCO2e") or (b.get("components_kgCO2e") or {})

        # Read computed values (NO recompute)
        pv_factor = totals.get("pv_factor_kg_per_kWh", None)
        pv_self_kWh = totals.get("pv_self_kWh", totals.get("pv_self_consumed_kWh", 0.0)) or 0.0
        pv_export_kWh = totals.get("pv_export_kWh", totals.get("grid_export_kWh", 0.0)) or 0.0

        pv_self_grid_avoided_kg = totals.get("pv_self_grid_avoided_kgCO2e", 0.0) or 0.0
        pv_self_pv_cost_kg = totals.get("pv_self_pv_cost_kgCO2e", 0.0) or 0.0
        pv_export_pv_cost_kg = totals.get("pv_export_pv_cost_kgCO2e", 0.0) or 0.0

        avoided_pv_for_consumption_kg = totals.get("avoided_by_pv_for_consumption_kgCO2e", totals.get("avoided_by_pv_kgCO2e", 0.0)) or 0.0
        diff_total_vs_no_pv_kg = totals.get("diff_total_vs_no_pv_kgCO2e", 0.0) or 0.0

        avoided_storage_kg = totals.get("avoided_by_storage_kgCO2e", 0.0) or 0.0
        batt_emb_kg = totals.get("battery_embodied_annualized_kgCO2e", 0.0) or 0.0
        batt_payback_years = totals.get("battery_payback_years", None)

        net_grid_kg = totals.get("net_kgCO2e", None)
        system_total_kg = totals.get("system_total_kgCO2e", None)

        # -----------------------
        # KPI row (building)
        # -----------------------
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("System CO₂ (annual)", "—" if system_total_kg is None else f"{float(system_total_kg)/1000:.2f} tCO₂e")
        with k2:
            st.metric("Net grid CO₂ (annual)", "—" if net_grid_kg is None else f"{float(net_grid_kg)/1000:.2f} tCO₂e")
        with k3:
            st.metric("PV factor", "—" if pv_factor is None else f"{float(pv_factor):.3f} kgCO₂e/kWh")
        with k4:
            st.metric("Grid factor", "—" if grid_factor is None else f"{float(grid_factor):.3f} kgCO₂e/kWh")

        st.markdown("---")

        # -----------------------
        # Storytelling
        # -----------------------
        st.markdown("#### CO₂ impact — existing assets (storytelling)")

        s_cols = st.columns([1.2, 1])
        with s_cols[0]:
            st.write(
                f"**PV self-consumption**: {pv_self_kWh:,.0f} kWh/year\n\n"
                f"- Grid avoided (gross): **{pv_self_grid_avoided_kg:,.0f} kgCO₂e/year**\n"
                f"- PV CO₂ cost (tech factor): **{pv_self_pv_cost_kg:,.0f} kgCO₂e/year**\n"
                f"- Net benefit on self-consumption: **{avoided_pv_for_consumption_kg:,.0f} kgCO₂e/year**"
            )

            st.write(
                f"**PV export**: {pv_export_kWh:,.0f} kWh/year\n\n"
                f"- PV CO₂ cost on exported energy (no credit): **{pv_export_pv_cost_kg:,.0f} kgCO₂e/year**"
            )

            st.success(
                f"**Total PV net benefit vs no PV** (your definition): "
                f"**{diff_total_vs_no_pv_kg:,.0f} kgCO₂e/year**"
            )

        with s_cols[1]:
            has_battery = float(totals.get("battery_capacity_kWh", 0.0) or 0.0) > 0.0
            if has_battery:
                st.metric("Battery avoided (annual)", f"{float(avoided_storage_kg)/1000:.2f} tCO₂e")
                st.metric("Battery embodied (annualized)", f"{float(batt_emb_kg)/1000:.2f} tCO₂e")
                if batt_payback_years is not None:
                    st.metric("Battery CO₂ payback", f"{float(batt_payback_years):.1f} years")
                else:
                    st.caption("Battery CO₂ payback: not available.")
            else:
                st.caption("No battery detected in measured results for this building.")

        st.markdown("---")

        # -----------------------
        # Breakdown (from components_kgCO2e)
        # -----------------------
        st.markdown("#### Annual CO₂ breakdown (read from core results)")

        if not isinstance(comp, dict) or not comp:
            st.info("No CO₂ components breakdown available for this building.")
        else:
            df_bar = pd.DataFrame([{"Component": k, "Signed kgCO2e": float(v)} for k, v in comp.items()])
            df_bar = df_bar.sort_values("Signed kgCO2e", ascending=False)

            fig_bar = px.bar(
                df_bar,
                x="Component",
                y="Signed kgCO2e",
                text=df_bar["Signed kgCO2e"].map(lambda x: f"{x:,.0f}"),
            )
            fig_bar.update_traces(textposition="outside")
            fig_bar.update_layout(showlegend=False, yaxis_title="kgCO₂e per year (signed)", xaxis_title="")
            st.plotly_chart(fig_bar, use_container_width=True)

            st.caption("Sign convention: benefits are negative, costs are positive (as provided by core).")

        st.markdown("---")

        # -----------------------
        # Timeseries (monthly)
        # -----------------------
        st.markdown("#### Monthly CO₂ balance on consumption (Scope 2, measured)")

        ts_b = ts_root.get(bat_id) or ts_root.get(str(bat_id)) or {}
        idx = ts_b.get("index") or []
        s2ts = (ts_b.get("scope2") or {})

        baseline_no_pv_ts = s2ts.get("baseline_no_pv_kgCO2e")
        net_ts = s2ts.get("net_kgCO2e")
        delta_vs_no_pv_ts = s2ts.get("delta_vs_no_pv_kgCO2e")

        if not (isinstance(idx, list) and len(idx) > 0
                and isinstance(baseline_no_pv_ts, list) and isinstance(net_ts, list)
                and len(baseline_no_pv_ts) == len(idx) and len(net_ts) == len(idx)):
            st.info("Monthly CO₂ chart not available (missing measured timeseries in results).")
        else:
            dt = pd.to_datetime(pd.Series(idx), errors="coerce")
            mask = dt.notna()

            df = pd.DataFrame({
                "baseline_no_pv": pd.Series(baseline_no_pv_ts),
                "net_with_pv": pd.Series(net_ts),
                "avoided": pd.Series(delta_vs_no_pv_ts) if isinstance(delta_vs_no_pv_ts, list) and len(delta_vs_no_pv_ts) == len(idx)
                           else (pd.Series(baseline_no_pv_ts) - pd.Series(net_ts)),
            })

            df = df.loc[mask.values].copy()
            df.index = dt.loc[mask].values
            m = df.resample("MS").sum(numeric_only=True)

            fig_m = go.Figure()
            fig_m.add_trace(go.Bar(x=m.index, y=m["baseline_no_pv"], name="Baseline (no PV)"))
            fig_m.add_trace(go.Bar(x=m.index, y=m["net_with_pv"], name="Net CO₂ (with PV)"))
            fig_m.add_trace(go.Scatter(
                x=m.index, y=m["avoided"],
                mode="lines+markers",
                name="Avoided vs baseline (PV benefit)",
                yaxis="y2",
            ))

            fig_m.update_layout(
                barmode="group",
                xaxis_title="Month",
                yaxis_title="kgCO₂e per month",
                yaxis2=dict(
                    title="kgCO₂e avoided per month",
                    overlaying="y",
                    side="right",
                    showgrid=False,
                ),
                legend_title="",
            )
            st.plotly_chart(fig_m, use_container_width=True)

        st.markdown("---")

        # -----------------------
        # Battery payback (if available)
        # -----------------------
        st.markdown("#### Battery CO₂ payback (if available)")

        years = totals.get("battery_payback_curve_years")
        cum = totals.get("battery_payback_curve_cum_kgCO2e")

        if isinstance(years, list) and isinstance(cum, list) and len(years) == len(cum) and len(years) > 1:
            if batt_emb_kg and avoided_storage_kg:
                st.write(
                    f"The battery required **{float(batt_emb_kg):,.0f} kgCO₂e** to manufacture "
                    f"and avoids **{float(avoided_storage_kg):,.0f} kgCO₂e per year** (grid factor)."
                )
            if batt_payback_years is not None:
                st.success(f"Estimated CO₂ payback time: **{float(batt_payback_years):.1f} years**.")

            df_pay = pd.DataFrame({"Year": years, "Cumulative CO₂ balance [kgCO₂e]": cum})
            fig_pay = px.bar(df_pay, x="Year", y="Cumulative CO₂ balance [kgCO₂e]")
            st.plotly_chart(fig_pay, use_container_width=True)
        else:
            st.caption("Battery payback curve not available for this building.")

        # -----------------------
        # Raw factors table (optional)
        # -----------------------
        with st.expander("Techno CO₂ factors (raw, imported from Phase 1 / Excel)", expanded=False):
            st.write(co2.get("tech_factors") or {})




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
            bat_nom = meta.get("batiment_nom", f"Building {bat_id}")
            ouv_nom = meta.get("ouvrage_nom", "")
            pv_label = meta.get("pv_label", "PV")

            label = f"{bat_nom} – {ouv_nom} – {pv_label}"
            yield label, fb


def render_pv_comparison_standard(project: dict):
    """
    Standard comparison view for PV producers:
    - Monthly table: measured vs theoretical
    - Bar chart
    - Global PR
    """
    pv_blocks = list(_iter_pv_blocks_with_comparison(project))

    if not pv_blocks:
        st.info(
            "No PV comparison available. Please check that:\n"
            "- a weather station is set in Phase 1,\n"
            "- PV module parameters (power, area, efficiency) are provided,\n"
            "- the producer is of type PV with a production profile."
        )
        return

    labels = [lbl for lbl, _ in pv_blocks]
    st.markdown("### PV production comparison — measured vs theoretical")

    selected_label = st.selectbox(
        "Select a PV producer",
        options=labels,
    )

    # Récupérer le bloc sélectionné
    fb_map = {lbl: blk for lbl, blk in pv_blocks}
    fb = fb_map[selected_label]

    profiles = fb.get("profiles", {})
    comp_records = profiles.get("pv_monthly_comparison", [])

    if not comp_records:
        st.warning("No comparison data for this PV producer.")
        return

    comp_df = pd.DataFrame(comp_records)

    # Sécuriser la colonne mois
    if "month" in comp_df.columns:
        comp_df = comp_df.sort_values("month")
        comp_df["month_label"] = comp_df["month"].astype(int).astype(str)
    else:
        comp_df["month"] = range(1, len(comp_df) + 1)
        comp_df["month_label"] = comp_df["month"].astype(int).astype(str)

    # ----- Global PR & totaux -----
    totals = fb.get("totals", {}) or {}
    pv_meas_total = totals.get("pv_measured_kWh_total", None)
    pv_theo_total = totals.get("pv_theoretical_kWh_total", None)
    pv_PR_global = totals.get("pv_PR_global", None)

    col1, col2, col3 = st.columns(3)
    with col1:
        if pv_meas_total is not None:
            st.metric(
                "Annual measured production [kWh]",
                f"{pv_meas_total:,.0f}".replace(",", " ")
            )
        else:
            st.metric("Annual measured production [kWh]", "—")
    with col2:
        if pv_theo_total is not None:
            st.metric(
                "Annual theoretical production [kWh]",
                f"{pv_theo_total:,.0f}".replace(",", " ")
            )
        else:
            st.metric("Annual theoretical production [kWh]", "—")
    with col3:
        if pv_PR_global is not None:
            st.metric("PR global [-]", f"{pv_PR_global:.2f}")
        else:
            st.metric("PR global [-]", "—")

    st.markdown("---")

    # ----- Bar chart Measured vs Theoretical -----
    fig = go.Figure()
    fig.add_bar(
        x=comp_df["month_label"],
        y=comp_df["pv_measured_kWh"],
        name="Measured [kWh/month]",
    )
    fig.add_bar(
        x=comp_df["month_label"],
        y=comp_df["pv_theoretical_kWh"],
        name="Theoretical [kWh/month]",
        opacity=0.7,
    )

    fig.update_layout(
        barmode="group",
        xaxis_title="Month",
        yaxis_title="Energy [kWh/month]",
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
    st.markdown("#### Monthly details")
    st.dataframe(
        comp_df[[
            "month",
            "pv_measured_kWh",
            "pv_theoretical_kWh",
            "ratio_theoretical_over_measured",
        ]],
        use_container_width=True,
    )

def show_exergy(project: dict):
    st.header("🔥 Exergy — Heating (efficiency-based)")

    results = project.get("results") or {}
    ex = results.get("exergy")

    if not isinstance(ex, dict):
        st.warning("No exergy results found. Run calculations in Phase 1.")
        return

    meta = ex.get("meta") or {}
    by_bat = ex.get("by_batiment") or {}
    global_ex = ex.get("global") or {}

    # -------------------------
    # Helpers
    # -------------------------
    def _fmt(x, fmt: str):
        try:
            if x is None:
                return "—"
            return format(float(x), fmt)
        except Exception:
            return "—"

    def _pct(x):
        return "—" if x is None else f"{100.0*float(x):.1f} %"

    # -------------------------
    # Scope selector
    # -------------------------
    options = ["Global"] + sorted(list(by_bat.keys()))
    sel = st.selectbox("Scope", options=options, index=0)

    if sel == "Global":
        title = "Global (weighted by delivered heat)"
        eta_m = global_ex.get("eta_ex_machine_wavg")
        eta_d = global_ex.get("eta_ex_distribution_wavg")
        eta_g = global_ex.get("eta_ex_global_wavg")
        temps = None
        details = None
        eta_th = None
        sources = None
        Q = global_ex.get("heat_out_th_kWh")
    else:
        title = f"Building {sel}"
        h = (by_bat.get(sel) or {}).get("heating") or {}
        eta_m = h.get("eta_ex_machine")
        eta_d = h.get("eta_ex_distribution")
        eta_g = h.get("eta_ex_global")
        temps = h.get("temps_C") or {}
        details = h.get("details") or {}
        eta_th = h.get("eta_th")
        sources = h.get("sources") or {}
        Q = h.get("heat_out_th_kWh")

    st.markdown(
        """
**Goal:** compare heating variants with a defendable and consistent metric.  
We display **only exergy efficiencies** (no kWh_ex), split into:
- **Machine adequacy** (max temperature capability vs used distribution level) + thermal efficiency
- **Distribution adequacy** (distribution temperature vs room temperature)
- **Global** = product
        """
    )

    if meta.get("T0_C_optional", None) is not None:
        st.caption(f"Optional climate info (not used in η_ex,global): annual mean outdoor temperature ≈ {_fmt(meta.get('T0_C_optional'), '.2f')} °C")

    st.subheader(title)

    # -------------------------
    # KPI cards (ONLY efficiencies)
    # -------------------------
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("η_ex, machine", _pct(eta_m), help=(
            "Definition:\n"
            "η_ex,machine = η_th × (1 − T_dist,K / T_max,K)\n\n"
            "Interpretation:\n"
            "- Penalizes oversizing of temperature level (e.g., boiler able to do 90°C but used at 30–35°C)\n"
            "- Includes the thermal efficiency η_th from the simulation\n"
        ))
    with c2:
        st.metric("η_ex, distribution", _pct(eta_d), help=(
            "Definition:\n"
            "η_ex,distribution = (T_room,K / T_dist,K)\n\n"
            "Interpretation:\n"
            "- Penalizes high distribution temperatures for a given indoor need\n"
            "- No distribution losses are assumed here (temperature-level adequacy only)\n"
        ))
    with c3:
        st.metric("η_ex, global", _pct(eta_g), help=(
            "Definition:\n"
            "η_ex,global = η_ex,machine × η_ex,distribution\n\n"
            "This gives a single defendable number for comparing variants.\n"
        ))
    with c4:
        st.metric("Delivered heat", f"{_fmt(Q, ',.0f')} kWh_th/y")

    # -------------------------
    # Temperatures table (defendable anchor)
    # -------------------------
    if sel != "Global" and isinstance(temps, dict) and temps:
        st.subheader("Temperature levels used")
        df_temp = pd.DataFrame({
            "Parameter": [
                "T_room",
                "T_supply",
                "T_return",
                "T_dist = (T_supply + T_return)/2",
                "T_max (boiler capability)",
            ],
            "Value [°C]": [
                temps.get("room"),
                temps.get("supply"),
                temps.get("return"),
                temps.get("dist"),
                temps.get("tmax_boiler"),
            ],
        })
        st.dataframe(df_temp, hide_index=True, use_container_width=True)

    # -------------------------
    # Detailed calculation (the “demo” inside the UI)
    # -------------------------
    if sel != "Global" and details is not None and temps is not None:
        with st.expander("Calculation details (exact numbers)"):
            st.markdown("**Formulas (Kelvin ratios)**")
            st.code(
                "η_ex,machine = η_th × (1 − T_dist,K / T_max,K)\n"
                "η_ex,distribution = (1 − T_room,K / T_dist,K)\n"
                "η_ex,global = η_ex,machine × η_ex,distribution",
                language="text",
            )

            st.markdown("**Inputs**")
            st.write({
                "η_th (from simulation)": eta_th,
                "T_room [°C]": temps.get("room"),
                "T_dist [°C]": temps.get("dist"),
                "T_max [°C]": temps.get("tmax_boiler"),
                "T_room [K]": details.get("Troom_K"),
                "T_dist [K]": details.get("Tdist_K"),
                "T_max [K]": details.get("Tmax_K"),
                "(1 − T_dist/T_max)": details.get("term_machine_temp"),
                "(1 − T_room/T_dist)": details.get("term_dist_temp"),
            })

            if sources:
                st.markdown("**Sources (traceability)**")
                st.write(sources)

    st.info(
        "Reading tip: a low η_ex does not mean the boiler is 'bad' energetically. "
        "It means the system uses a high-quality energy resource and/or high temperature capability "
        "to satisfy a low-temperature demand — a thermodynamic mismatch."
    )
    

# =====================================================================
# COMPARAISON STANDARD PV
# =====================================================================

def show_comparison(project: dict):
    st.markdown(
        "<div class='sf-section-title'>PV standard comparison</div>",
        unsafe_allow_html=True,
    )

    results = project.get("results", {}) or {}
    pv_std = results.get("pv_standard", []) or []

    if not pv_std:
        st.info(
            "No data for PV standard comparison. "
            "Add at least one properly configured PV producer in Phase 1 "
            "and re-run the calculations."
        )
        return

    # -------------------------------------------------------------
    # 1) Tableau récapitulatif des paramètres PV
    # -------------------------------------------------------------
    rows = []
    warnings_msgs = []

    for rec in pv_std:
        rows.append({
            "Building": rec.get("batiment_nom", ""),
            "Unit": rec.get("ouvrage_nom", ""),
            "Technologie": rec.get("producer_techno", ""),
            "Installed P [kW]": rec.get("installed_kw", 0.0),
            "Orientation [°]": rec.get("orientation_deg"),
            "Inclinaison [°]": rec.get("inclinaison_deg"),
            "P_module [kW]": rec.get("p_module_kw"),
            "Surface module [m²]": rec.get("area_module_m2"),
            "Rendement module [%]": rec.get("eta_mod_pct"),
            "Theoretical production [kWh/kW/yr]": rec.get("theoretical_annual_kWh_per_kW"),
            "Measured production [kWh/kW/yr]": rec.get("measured_annual_kWh_per_kW"),
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
    st.subheader("PV parameters used for the standard comparison")
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
        st.info("Theoretical monthly profile not available (no PV producer with computed profile).")
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
            st.warning(f"PV calculation error: {first_with_profile['calc_error']}")
        else:
            st.info("Theoretical monthly profile not available (invalid list).")
        return

    mois_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    df_plot = pd.DataFrame({"Month": mois_labels, "kWh/kW_th": theo_vals})

    if meas_vals is not None and len(meas_vals) == 12:
        df_plot["kWh/kW_meas"] = meas_vals

    bat_nom = first_with_profile.get("batiment_nom", "")
    ouv_nom = first_with_profile.get("ouvrage_nom", "")
    techno = first_with_profile.get("producer_techno", "PV")

    st.subheader(f"Theoretical monthly profile — {bat_nom} / {ouv_nom} / {techno}")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_plot["Month"],
            y=df_plot["kWh/kW_th"],
            mode="lines+markers",
            name="Theoretical",
        )
    )

    if "kWh/kW_meas" in df_plot.columns:
        fig.add_trace(
            go.Scatter(
                x=df_plot["Month"],
                y=df_plot["kWh/kW_meas"],
                mode="lines+markers",
                name="Measured",
            )
        )

    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Specific production [kWh/kW]",
        height=350,
        margin=dict(l=40, r=20, t=40, b=40),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------------------------
    # 3) Ranked time-step view (sorted by PV production, aligned metrics)
    #    We keep only the zoomed view (PV > 0), as it is more readable.
    # -------------------------------------------------------------
    st.subheader("Ranked timesteps — PV production > 0 only (baseline, aligned metrics)")
    st.caption(
        "Each point is one timestep. The x-axis is a ranking (highest PV production first). "
        "Consumption and PV self-consumption remain the values from the same timestep as the PV production."
    )
    
    # --- source de vérité : results ---
    ts_by_bat = (results.get("timeseries_by_batiment") or {})
    if not ts_by_bat:
        st.info("No time series available in results (timeseries_by_batiment missing).")
        return
    
    agg_load = None
    agg_pv = None
    agg_auto = None
    base_index = None
    
    for _bat_id, bat_data in ts_by_bat.items():
        measured = (bat_data.get("measured") or {})
        idx_raw = bat_data.get("index") or measured.get("index")
        if not idx_raw:
            continue
    
        dt_idx = pd.to_datetime(idx_raw, errors="coerce")
        if dt_idx.isna().all():
            dt_idx = pd.RangeIndex(len(idx_raw))
    
        load_list = measured.get("load_kWh")
        pv_list = measured.get("pv_prod_kWh")
        auto_list = measured.get("pv_to_load_kWh")
    
        if load_list is None and pv_list is None and auto_list is None:
            continue
    
        if base_index is None:
            base_index = dt_idx
    
        def _to_series(v):
            n = len(base_index)
            if v is None:
                return pd.Series([0.0] * n, index=base_index, dtype=float)
    
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
        agg_auto = s_auto if agg_auto is None else agg_auto.add(s_auto, fill_value=0.0)
    
    if agg_pv is None:
        st.info("Time series exist but PV production is missing everywhere (pv_prod_kWh).")
        return
    
    df_rank = pd.DataFrame(
        {
            "timestep": agg_pv.index,
            "Electric consumption [kWh]": agg_load if agg_load is not None else 0.0,
            "PV production [kWh]": agg_pv,
            "PV self-consumption [kWh]": agg_auto if agg_auto is not None else 0.0,
        }
    )
    
   
    # -------------------------
    # Zoomed ranked plot: keep only PV > 0
    # -------------------------
    df_zoom = df_rank[df_rank["PV production [kWh]"] > 0.0].copy()
    if df_zoom.empty:
        st.info("No timestep with PV production > 0 in baseline measured time series.")
    else:
        # tri décroissant sur la PV (les autres colonnes suivent car même ligne = même timestep)
        df_zoom = df_zoom.sort_values("PV production [kWh]", ascending=False).reset_index(drop=True)
        df_zoom["rank"] = np.arange(1, len(df_zoom) + 1)
    
        fig_rank_zoom = go.Figure()
        fig_rank_zoom.add_trace(go.Scatter(
            x=df_zoom["rank"],
            y=df_zoom["Electric consumption [kWh]"],
            mode="lines",
            name="Electric consumption",
            customdata=df_zoom[["timestep"]],
            hovertemplate="Rank=%{x}<br>Consumption=%{y:.3f} kWh<br>Timestep=%{customdata[0]}<extra></extra>",
        ))
        fig_rank_zoom.add_trace(go.Scatter(
            x=df_zoom["rank"],
            y=df_zoom["PV production [kWh]"],
            mode="lines",
            name="PV production",
            customdata=df_zoom[["timestep"]],
            hovertemplate="Rank=%{x}<br>PV=%{y:.3f} kWh<br>Timestep=%{customdata[0]}<extra></extra>",
        ))
        fig_rank_zoom.add_trace(go.Scatter(
            x=df_zoom["rank"],
            y=df_zoom["PV self-consumption [kWh]"],
            mode="lines",
            name="PV self-consumption",
            customdata=df_zoom[["timestep"]],
            hovertemplate="Rank=%{x}<br>Self-cons=%{y:.3f} kWh<br>Timestep=%{customdata[0]}<extra></extra>",
        ))
    
        fig_rank_zoom.update_layout(
            height=380,
            margin=dict(l=40, r=20, t=40, b=40),
            xaxis_title="Rank (PV timesteps only, highest PV first)",
            yaxis_title="Energy per timestep [kWh]",
            hovermode="x unified",
        )
    
        st.plotly_chart(fig_rank_zoom, use_container_width=True)

    
    # -------------------------
    # KPI helper: step duration
    # -------------------------
    if isinstance(base_index, pd.DatetimeIndex):
        step_hours = dt_hours_series(base_index)
    else:
        step_hours = None
        st.warning("Index is not datetime → timestep duration cannot be inferred (KPIs shown as number of timesteps).")
    
    # -------------------------
    # KPIs: PV hours (2-column layout)
    # -------------------------
    col1, col2 = st.columns(2)
    
    with col1:
        if agg_load is not None:
            mask_pv_gt_load = (agg_pv > agg_load)
            if step_hours is not None:
                hours_pv_gt_load = float(step_hours[mask_pv_gt_load].sum())
            else:
                hours_pv_gt_load = float(mask_pv_gt_load.sum())
            st.metric(
                "Hours where PV production > electric consumption",
                f"{hours_pv_gt_load:.1f} h",
            )
    
    with col2:
        mask_pv_pos = (agg_pv > 0.0)
        if step_hours is not None:
            hours_pv_pos = float(step_hours[mask_pv_pos].sum())
        else:
            hours_pv_pos = float(mask_pv_pos.sum())
        st.metric(
            "Hours with PV production > 0",
            f"{hours_pv_pos:.1f} h",
        )

    
    show_rank_by_load = st.checkbox("Show ranked view sorted by electric consumption (baseline)", value=False)

    if show_rank_by_load:
        # Keep only timesteps with consumption > 0 (zoom)
        df_load = df_rank[df_rank["Electric consumption [kWh]"] > 0.0].copy()
    
        if df_load.empty:
            st.info("No timestep with electric consumption > 0 in baseline measured time series.")
        else:
            df_load = df_load.sort_values("Electric consumption [kWh]", ascending=False).reset_index(drop=True)
            df_load["rank"] = np.arange(1, len(df_load) + 1)
    
            st.subheader("Ranked timesteps — sorted by electric consumption (baseline, aligned metrics)")
            st.caption(
                "Each point is one timestep. The x-axis is a ranking (highest consumption first). "
                "PV production and self-consumption remain aligned to the same timestep."
            )
    
            fig_rank_load = go.Figure()

            # Consumption stays as a line (main driver, readable)
            fig_rank_load.add_trace(go.Scatter(
                x=df_load["rank"],
                y=df_load["Electric consumption [kWh]"],
                mode="lines",
                name="Electric consumption",
                customdata=df_load[["timestep"]],
                hovertemplate="Rank=%{x}<br>Consumption=%{y:.3f} kWh<br>Timestep=%{customdata[0]}<extra></extra>",
            ))
            
            # PV production as point cloud
            fig_rank_load.add_trace(go.Scatter(
                x=df_load["rank"],
                y=df_load["PV production [kWh]"],
                mode="markers",
                name="PV production",
                marker=dict(size=4, opacity=0.35),
                customdata=df_load[["timestep"]],
                hovertemplate="Rank=%{x}<br>PV=%{y:.3f} kWh<br>Timestep=%{customdata[0]}<extra></extra>",
            ))
            
            # PV self-consumption as point cloud
            fig_rank_load.add_trace(go.Scatter(
                x=df_load["rank"],
                y=df_load["PV self-consumption [kWh]"],
                mode="markers",
                name="PV self-consumption",
                marker=dict(size=4, opacity=0.45),
                customdata=df_load[["timestep"]],
                hovertemplate="Rank=%{x}<br>Self-cons=%{y:.3f} kWh<br>Timestep=%{customdata[0]}<extra></extra>",
            ))
            
            fig_rank_load.update_layout(
                height=380,
                margin=dict(l=40, r=20, t=40, b=40),
                xaxis_title="Rank (sorted by consumption, highest first)",
                yaxis_title="Energy per timestep [kWh]",
                hovermode="x unified",
            )
            
            st.plotly_chart(fig_rank_load, use_container_width=True)

    
def show_phases_analysis(project: dict):
    st.markdown(
        "<div class='sf-section-title'>Time-series analysis: consumption vs PV production vs self-consumption</div>",
        unsafe_allow_html=True,
    )
    st.info(
        "Detailed time-series data (point-by-point) + quarterly views."
    )

   # ------------------------------------------------------------------
    # 4) Graphique temporel détaillé
    # ------------------------------------------------------------------
    st.subheader("Detailed time-series curves")
    st.caption(
    "ℹ️ The PV self-consumption curve (red) represents the photovoltaic energy "
    "actually consumed by the building at each timestep. "
    "When it is not measured, it is reconstructed in Phase 1 "
    "using a proportion-based, physically bounded method "
    "(self-consumption ≤ PV production and ≤ electric consumption)."
)

    
    # ------------------------------------------------------------------
    # OVERRIDE df_full : on trace UNIQUEMENT depuis results (source de vérité)
    # ------------------------------------------------------------------
    results = project.get("results", {}) or {}
    ts_by_bat = results.get("timeseries_by_batiment", {}) or {}
    
    if not ts_by_bat:
        st.warning("No time series computed in results. Re-run Phase 1.")
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
        st.warning("timeseries_by_batiment exists but cannot be used.")
        return
    
    df_full = pd.DataFrame({
        "Consumptionmmation [kWh]": agg_load if agg_load is not None else pd.Series(dtype=float),
        "PV production [kWh]": agg_pv if agg_pv is not None else pd.Series(dtype=float),
        "PV self-consumption [kWh]": agg_pv_auto if agg_pv_auto is not None else pd.Series(dtype=float),
    }).sort_index()
    
    index_type = "datetime" if isinstance(df_full.index, pd.DatetimeIndex) else "int"

    fig_full = px.line(
        df_full,
        x=df_full.index,
        y=df_full.columns,
        title="Consumption vs PV vs self-consumption — time detail",
    )
    # Consumption en avant-plan, PV plus discret
    for tr in fig_full.data:
        if "Consumptionmmation" in tr.name:
            tr.update(line=dict(width=3))              # trait épais
        elif "PV production" in tr.name:
            tr.update(line=dict(width=1, dash="dash"))  # PV en pointillé fin
        else:
            tr.update(line=dict(width=1))               # autoconsommation normal

    fig_full.update_layout(
        xaxis_title="Time" if index_type == "datetime" else "Index (timesteps)",
        yaxis_title="Energy [kWh]",
        legend_title="Flows",
    )
    
    # --- PV self-consumption saturation warning ---
    for bat_id, bat_data in ts_by_bat.items():
        meta = bat_data.get("measured", {}).get("pv_selfc_meta")
        if meta and meta.get("saturated", False):
            st.warning(
                f"⚠️ Requested self-consumption = {meta['selfc_pct_input']*100:.1f} % "
                f"→ maximum physically achievable = {meta['max_selfc_pct']*100:.1f} %. "
                f"Achieved result = {meta['selfc_pct_real']*100:.1f} %."
            )
            break

    st.plotly_chart(fig_full, use_container_width=True)

    # ------------------------------------------------------------------
    # 5) Vue trimestrielle (avec fallback si pas de dates)
    # ------------------------------------------------------------------
    st.subheader("Summary by 3-month periods")

    def compute_quarterly_from_datetime(df):
        df_q = df.copy()
        df_q["month"] = df_q.index.month

        groups = {
            "Jan–Mar": [1, 2, 3],
            "Apr–Jun": [4, 5, 6],
            "Jul–Sep": [7, 8, 9],
            "Oct–Dec": [10, 11, 12],
        }

        rows = []
        for label, months in groups.items():
            subset = df_q[df_q["month"].isin(months)]
            rows.append(
                {
                    "Period": label,
                    "Consumptionmmation [kWh]": subset["Consumptionmmation [kWh]"].sum(),
                    "PV production [kWh]": subset["PV production [kWh]"].sum(),
                    "PV self-consumption [kWh]": subset["PV self-consumption [kWh]"].sum(),
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
                    "Period": label,
                    "Consumptionmmation [kWh]": part["Consumptionmmation [kWh]"].sum(),
                    "PV production [kWh]": part["PV production [kWh]"].sum(),
                    "PV self-consumption [kWh]": part["PV self-consumption [kWh]"].sum(),
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
            id_vars="Period",
            var_name="Type",
            value_name="Energy [kWh]",
        )

        fig_quarters = px.bar(
            df_quarters_long,
            x="Period",
            y="Energy [kWh]",
            color="Type",
            barmode="group",
        )
        fig_quarters.update_layout(
            xaxis_title="Period",
            yaxis_title="Energy [kWh]",
            legend_title="Flows",
        )
        st.plotly_chart(fig_quarters, use_container_width=True)

def show_variantes(project: dict):
    import streamlit as st
    import pandas as pd
    import plotly.graph_objects as go

    # ==========================================================
    # Helpers (local only)
    # ==========================================================
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

    def _bar_cum_cashflow(years: list, cashflow_cum: list, title: str):
        df_cf = pd.DataFrame({"Year": years, "Cumulative [CHF]": cashflow_cum})
        vals = df_cf["Cumulative [CHF]"].tolist()
        colors = ["rgba(220,53,69,0.85)" if v < 0 else "rgba(40,167,69,0.85)" for v in vals]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df_cf["Year"], y=df_cf["Cumulative [CHF]"], marker_color=colors))
        fig.update_layout(title=title, xaxis_title="Years", yaxis_title="CHF", height=360)
        return fig

    def _get_batt_proposed_summary(results: dict, bat_id):
        bp = (results.get("battery_proposed_by_batiment") or {})
        return bp.get(bat_id) or bp.get(str(bat_id))

    def _get_ts_payload(results: dict, bat_id):
        ts_root = (results.get("timeseries_by_batiment") or {})
        bat_data = ts_root.get(bat_id) or ts_root.get(str(bat_id))
        if not isinstance(bat_data, dict):
            return None
        if isinstance(bat_data.get("battery_proposed"), dict):
            return bat_data["battery_proposed"]
        if "before" in bat_data or "after" in bat_data:
            return bat_data
        return None

    def _agg_by_quarter(idx: pd.DatetimeIndex, series: pd.Series) -> pd.DataFrame:
        df = pd.DataFrame({"v": series.values}, index=idx)
        q = idx.to_period("Q")
        g = df.groupby(q)["v"].sum()
        out = g.reset_index()
        out.columns = ["Quarter", "kWh"]
        out["Quarter"] = out["Quarter"].astype(str)
        return out
    
    def _show_variants_co2_scope2(results: dict, bat_id, bat_nom: str, pv_sim_all: dict | None = None):
        import streamlit as st
        import pandas as pd
        import plotly.graph_objects as go
    
        co2 = (results.get("co2") or {})
        by_bat = (co2.get("by_batiment") or {})
        b = by_bat.get(bat_id) or by_bat.get(str(bat_id)) or {}
        scope2 = (b.get("scope2") or {})
    
        totals = (scope2.get("totals") or {})
        totals_after = (scope2.get("totals_after") or {})
        
        # Fallback: si totals_after n'est pas fourni par le core, on agrège la timeseries CO2 "after"
        if not totals_after:
            ts_root = (co2.get("timeseries_by_batiment") or {})
            ts_b = ts_root.get(bat_id) or ts_root.get(str(bat_id)) or {}
            s2 = (ts_b.get("scope2") or {})
            net_after_ts = s2.get("net_kgCO2e_after")
        
            if isinstance(net_after_ts, list) and len(net_after_ts) > 0:
                totals_after = dict(totals)  # copie pour garder la structure
                # IMPORTANT: le reste du code lit net_kgCO2e_after (pas net_kgCO2e)
                totals_after["net_kgCO2e_after"] = float(sum(net_after_ts))


    
        if not totals:
            st.info("No CO₂ data available for this building (scope2.totals missing).")
            return
        
        battery_included = False
        try:
            battery_included = (
                totals_after
                .get("components_kgCO2e_after", {})
                .get("battery_embodied", 0.0) > 0
            )
        except Exception:
            battery_included = False

    
        # -------------------------
        # KPIs
        # -------------------------
        net_before = float(totals.get("net_kgCO2e", 0.0) or 0.0)
        net_after = float(totals_after.get("net_kgCO2e_after", net_before) or net_before)
    
        delta = net_after - net_before
    
        # Prefer proposed payback if available
        payback_years = totals_after.get("battery_payback_years_after") or totals.get("battery_payback_years", None)
        if payback_years is None:
            payback_years = totals.get("battery_payback_years", None)
        
        # Battery embodied annualized (prefer proposed if available)
        batt_embodied_annual = totals_after.get("battery_embodied_annualized_kgCO2e_after", None)
        if batt_embodied_annual is None:
            batt_embodied_annual = totals.get("battery_embodied_annualized_kgCO2e", None)

    
        # -------------------------
        # KPIs
        # -------------------------
        net_before = float(totals.get("net_kgCO2e", 0.0) or 0.0)
        net_after = float(totals_after.get("net_kgCO2e_after", net_before) or net_before)
        delta_grid = net_after - net_before
        
        # NEW: system totals (from core, no recompute)
        sys_before = float(totals.get("system_total_kgCO2e", 0.0) or 0.0)
        sys_after = float(totals_after.get("system_total_kgCO2e_after", sys_before) or sys_before)
        delta_sys = sys_after - sys_before
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Grid CO₂ (Scope 2) — Before [kgCO₂e/year]", f"{net_before:,.0f}")
        c2.metric("Grid CO₂ (Scope 2) — Proposed [kgCO₂e/year]", f"{net_after:,.0f}")
        c3.metric("Δ Grid CO₂ [kgCO₂e/year]", f"{delta_grid:,.0f}")
        
        c4, c5, c6 = st.columns(3)
        c4.metric("System CO₂ — Before [kgCO₂e/year]", f"{sys_before:,.0f}")
        c5.metric("System CO₂ — Proposed [kgCO₂e/year]", f"{sys_after:,.0f}")
        c6.metric("Δ System CO₂ [kgCO₂e/year]", f"{delta_sys:,.0f}")


    
        # -------------------------
        # Components bars (annual)
        # -------------------------
        comp_before = (totals.get("components_kgCO2e") or {})
        comp_after = (totals_after.get("components_kgCO2e_after") or {})
    
        # Si pas de after → on affiche quand même la barre "Before" seule
        labels = ["Before", "Proposed"]
    
        # Ordre stable et “storytelling”
        comp_order = [
            ("grid_import", "Grid import"),
            ("pv_net_benefit_self_consumption", "PV benefit (self-consumption)"),
            ("pv_export_pv_cost", "PV export (PV cost)"),
            ("battery_embodied", "Battery embodied (annualized)"),
        ]

    
        fig = go.Figure()
        for key, name in comp_order:
            v0 = float(comp_before.get(key, 0.0) or 0.0)
            v1 = float(comp_after.get(key, v0) or v0)
            fig.add_trace(go.Bar(name=name, x=labels, y=[v0, v1]))
    
        fig.update_layout(
            barmode="stack",
            height=420,
            margin=dict(l=10, r=10, t=30, b=10),
            yaxis_title="kgCO₂e/year (benefits may be negative)",
            title="Components — Before vs Proposed (annual)",
            legend_title="Components",
        )
        st.plotly_chart(fig, use_container_width=True)
    
        years_curve = totals_after.get("battery_payback_curve_years") or totals.get("battery_payback_curve_years")
        cum_curve   = totals_after.get("battery_payback_curve_cum_kgCO2e") or totals.get("battery_payback_curve_cum_kgCO2e")

        st.markdown("### Variant includes")

        # Read-only: CO2 per-component annual contributions (after)
        co2_all = results.get("co2") or {}
        co2_bb = (co2_all.get("by_batiment") or {})
        co2_b = co2_bb.get(bat_id) or co2_bb.get(str(bat_id)) or {}
        scope2 = (co2_b.get("scope2") or {})
        totals_after = (scope2.get("totals_after") or {})
        comp_after = (totals_after.get("components_kgCO2e_after") or {})
        
        pv_net = float(comp_after.get("pv_net_benefit_self_consumption", 0.0) or 0.0)
        pv_export_cost = float(comp_after.get("pv_export_pv_cost", 0.0) or 0.0)
        
        # "CO2 saved" display value (positive means improvement)
        # PV CO2 saved — TOTAL system (not only Scope 2)
        sys_before = float(totals.get("system_total_kgCO2e", 0.0) or 0.0)
        sys_after = float(totals_after.get("system_total_kgCO2e_after", sys_before) or sys_before)
        
        pv_saved_display = (sys_before - sys_after)  # positive = improvement

        

        # Battery: show NET impact (avoided - embodied annualized) so it matches System CO₂ logic
        bat_avoided = float(totals_after.get("battery_avoided_kgCO2e_after", 0.0) or 0.0)
        bat_emb_annual = float(totals_after.get("battery_embodied_annualized_kgCO2e_after", 0.0) or 0.0)
        bat_net_saved = bat_avoided - bat_emb_annual  # positive = improvement, negative = worse


    

        rows = []
        
        # PV (from pv_sim_all passed by Variants page)
        pv_sim_all = pv_sim_all or {}
        pv = pv_sim_all.get(bat_id) or pv_sim_all.get(str(bat_id)) or {}
        has_pv = isinstance(pv, dict) and float(pv.get("annual_kwh", 0.0) or 0.0) > 0
        
        rows.append({
            "Component": "PV (proposed)",
            "Included": "Yes" if has_pv else "No",
            "Key values": (
                f"P={float(pv.get('p_tot_kw', 0.0) or 0.0):.1f} kW, "
                f"E={float(pv.get('annual_kwh', 0.0) or 0.0):,.0f} kWh/y"
            ) if has_pv else "—",
            "CO₂ saved [kgCO₂e/year]": f"{pv_saved_display:,.0f}" if has_pv else "—",
        })
        
        # Battery (from results)
        bp_all = results.get("battery_proposed_by_batiment") or {}
        bp = bp_all.get(bat_id) or bp_all.get(str(bat_id)) or {}
        has_batt = isinstance(bp, dict) and float(bp.get("capacity_kwh", 0.0) or 0.0) > 0
        
        rows.append({
            "Component": "Battery (proposed)",
            "Included": "Yes" if has_batt else "No",
            "Key values": (
                f"C={float(bp.get('capacity_kwh', 0.0) or 0.0):.1f} kWh, "
                f"Life={float(bp.get('batt_lifetime_years', 0.0) or 0.0):.0f} y"
            ) if has_batt else "—",
            "CO₂ saved [kgCO₂e/year]": f"{bat_net_saved:,.0f}" if has_batt else "—",
        })

        
        df_inc = pd.DataFrame(rows)
        st.dataframe(df_inc, use_container_width=True, hide_index=True)

        # -------------------------
        # CO₂ payback / impact curves (generic, Proposed)
        # -------------------------
        st.markdown("### CO₂ payback & impact curves (Proposed)")

        # Horizon for all "over time" curves (prefer simulation params, fallback 25)
        params = (results.get("params") or results.get("simu_params") or {})
        try:
            horizon_years = int(params.get("analysis_horizon_years", 25) or 25)
        except Exception:
            horizon_years = 25
        horizon_years = max(1, min(horizon_years, 60))  # UI safety cap
        
        paybacks = (totals_after.get("payback_curves_after") or {}) if isinstance(totals_after, dict) else {}
        if not isinstance(paybacks, dict):
            paybacks = {}
        
        rows_pb = []
        series = []  # list of (label, years_list, cum_list)
        
        def _safe_int(v, default=None):
            try:
                if v in (None, ""):
                    return default
                return int(float(v))
            except Exception:
                return default
        
        def _extend_to_horizon(years_list, cum_list, horizon):
            """
            Extend an existing curve to the analysis horizon for display consistency.
            We do NOT invent extra benefit; we keep the last cumulative value flat.
            """
            if not isinstance(years_list, list) or not isinstance(cum_list, list):
                return None, None
            if len(years_list) != len(cum_list) or len(years_list) < 2:
                return None, None
        
            try:
                y_last = int(years_list[-1])
            except Exception:
                return None, None
        
            if y_last >= horizon:
                return years_list, cum_list
        
            y_ext = list(years_list)
            c_ext = list(cum_list)
            last_val = float(c_ext[-1] or 0.0)
        
            for y in range(y_last + 1, horizon + 1):
                y_ext.append(y)
                c_ext.append(last_val)
        
            return y_ext, c_ext
        
        def _is_pv_component(comp_key, pb_dict):
            try:
                k = str(comp_key).strip().lower()
            except Exception:
                k = ""
            lbl = ""
            if isinstance(pb_dict, dict):
                try:
                    lbl = str(pb_dict.get("label", "")).strip().lower()
                except Exception:
                    lbl = ""
            return (k == "pv") or (lbl == "pv")
        
        def _curve_has_data(pb_dict):
            if not isinstance(pb_dict, dict):
                return False
            curve = pb_dict.get("curve") or {}
            if not isinstance(curve, dict):
                return False
            ys = curve.get("years")
            cs = curve.get("cumulative_kgCO2e")
            return isinstance(ys, list) and isinstance(cs, list) and len(ys) >= 2 and len(ys) == len(cs)
        
        # -------------------------
        # 0) Filter paybacks by included components (avoid showing components not in variant)
        # -------------------------
        included = set()
        if has_pv:
            included.add("pv")
        if has_batt:
            included.add("battery")
        
        paybacks = {k: v for k, v in paybacks.items() if str(k).strip().lower() in included}


        
        # -------------------------
        # 1) PV: ensure a single PV entry, with annual benefit = System CO₂ saved (UI convention)
        #    and if curve missing -> build "impact" curve up to horizon
        # -------------------------
        pv_sim_all = pv_sim_all or {}
        pv = pv_sim_all.get(bat_id) or pv_sim_all.get(str(bat_id)) or {}
        has_pv = bool(has_pv)
        
        sys_before = float(totals.get("system_total_kgCO2e", 0.0) or 0.0)
        sys_after = float(totals_after.get("system_total_kgCO2e_after", sys_before) or sys_before)
        
        # Convention UI: positive = improvement (CO₂ avoided)
        annual_saved = (sys_before - sys_after)
        
        pv_life = _safe_int(pv.get("lifetime_years", None), default=None)
        
        if has_pv:
            # Ensure the key exists (core may store 'pv' or may not)
            pb_pv = paybacks.get("pv")
            if not isinstance(pb_pv, dict):
                pb_pv = {"label": "PV"}
        
            # Force annual benefit shown in UI = Variant saved (System delta)
            pb_pv["annual_benefit_kgCO2e"] = float(annual_saved)
            pb_pv["lifetime_years"] = pb_pv.get("lifetime_years", pv_life)
        
            # If no real curve from core, build impact curve (no payback without upfront)
            if (not _curve_has_data(pb_pv)) and abs(annual_saved) > 1e-9:
                ys = list(range(0, horizon_years + 1))
                cs = [y * float(annual_saved) for y in ys]
                pb_pv["curve"] = {
                    "years": ys,
                    "cumulative_kgCO2e": cs,
                    "payback_year": None,
                    "horizon_benefit_kgCO2e": float(cs[-1]),
                }
        
            # Write back (single PV)
            paybacks["pv"] = pb_pv
        
        # -------------------------
        # 2) Build table + series FROM paybacks only (single pass => no duplicates)
        # -------------------------
        for comp_key, pb in (paybacks or {}).items():
            if not isinstance(pb, dict):
                continue
        
            label = pb.get("label", comp_key)
            upfront = pb.get("upfront_kgCO2e", None)
            annual = pb.get("annual_benefit_kgCO2e", None)
            # Enforce consistent convention for Battery: use NET annual benefit (avoided - embodied annualized)
            if str(comp_key).strip().lower() == "battery" or str(label).strip().lower() == "battery":
                bat_avoided = float(totals_after.get("battery_avoided_kgCO2e_after", 0.0) or 0.0)
                bat_emb_annual = float(totals_after.get("battery_embodied_annualized_kgCO2e_after", 0.0) or 0.0)
                annual = bat_avoided - bat_emb_annual

            life = pb.get("lifetime_years", None)
        
            curve = pb.get("curve", None) or {}
            years_curve = curve.get("years", None)
            cum_curve = curve.get("cumulative_kgCO2e", None)
            payback_year = curve.get("payback_year", None)
            horizon_benefit = curve.get("horizon_benefit_kgCO2e", None)
        
            # Extend curve to horizon for display
            ys2, cs2 = _extend_to_horizon(years_curve, cum_curve, horizon_years)
        
            rows_pb.append({
                "Component": label,
                "Upfront [kgCO₂e]": upfront,
                "Annual benefit [kgCO₂e/y]": annual,
                "Lifetime [y]": life,
                "Payback year": payback_year,
                "Horizon benefit [kgCO₂e]": horizon_benefit,
            })
        
            if ys2 is not None and cs2 is not None:
                series.append((str(label), ys2, cs2))
        
        # 3) Table
        df_pb_tbl = pd.DataFrame(rows_pb) if rows_pb else pd.DataFrame(
            columns=["Component", "Upfront [kgCO₂e]", "Annual benefit [kgCO₂e/y]", "Lifetime [y]", "Payback year", "Horizon benefit [kgCO₂e]"]
        )
        st.dataframe(df_pb_tbl, use_container_width=True, hide_index=True)
        
        # 4) Plot
        if series:
            fig_pb = go.Figure()
        
            # We want: per-component bars colored by sign + optional payback vertical line
            # We'll plot cumulative values as bars (one bar per year).
            # If multiple series exist, we overlay (works OK for 1-2 components). If later you have many,
            # we can switch to a dropdown selector.
        
            for label, ys, cs in series:
                # Build colors by sign (green if >=0, red if <0)
                colors = []
                for v in cs:
                    try:
                        vv = float(v)
                    except Exception:
                        vv = 0.0
                    colors.append("rgba(46, 204, 113, 0.85)" if vv >= 0 else "rgba(231, 76, 60, 0.85)")
        
                fig_pb.add_trace(
                    go.Bar(
                        x=ys,
                        y=cs,
                        name=str(label),
                        marker=dict(color=colors),
                        opacity=0.95,
                    )
                )
        
                # Add payback vertical line if available in table rows (match by Component label)
                # We take the first matching row with a numeric payback year.
                pb_year = None
                try:
                    for r in rows_pb:
                        if str(r.get("Component", "")).strip() == str(label).strip():
                            py = r.get("Payback year", None)
                            if py not in (None, "", "None"):
                                pb_year = float(py)
                            break
                except Exception:
                    pb_year = None
        
                if pb_year is not None:
                    # vertical dashed line at payback year
                    fig_pb.add_vline(
                        x=pb_year,
                        line_width=2,
                        line_dash="dash",
                        line_color="rgba(0,0,0,0.35)"
                    )
        
            fig_pb.add_hline(y=0, line_width=1, line_dash="solid", line_color="rgba(0,0,0,0.25)")
            fig_pb.update_layout(
                title=f"{bat_nom} — CO₂ payback / impact curves (cumulative)",
                xaxis_title="Years",
                yaxis_title="Cumulative CO₂ balance / saved [kgCO₂e] (>=0 means payback reached)",
                height=380,
                margin=dict(l=10, r=10, t=50, b=10),
                barmode="group",
                legend=dict(orientation="h"),
            )
            st.plotly_chart(fig_pb, use_container_width=True)
        
            st.caption(
                "Notes: 1) Curves are displayed up to the analysis horizon. "
                "2) When upfront is missing, PV is shown as an 'impact' curve (no payback year)."
            )
        else:
            st.caption("No payback/impact curve available for proposed components (missing annual benefit and/or curve data).")

    # ==========================================================
    # Header + data roots
    # ==========================================================
    st.markdown("## Variants")

    results = project.get("results", {}) or {}
    flows = (results.get("flows", {}) or {}).get("batiments", {}) or {}
    co2_root = (results.get("co2") or {}).get("by_batiment", {}) or {}

    # PV proposed container (existing logic)
    pv_sim_all = (project.get("pv_simulated") or results.get("pv_proposed_by_batiment") or {})
    if not isinstance(pv_sim_all, dict):
        pv_sim_all = {}

    batiments = project.get("batiments", []) or []
    if not batiments:
        st.warning("No building in the project.")
        return

    # ==========================================================
    # Building selector (top, shared)
    # ==========================================================
    opts = []
    for bi, bat in enumerate(batiments):
        bat_id = bat.get("id") or bat.get("batiment_id") or bi
        bat_nom = bat.get("nom") or f"Building {bi+1}"
        opts.append((bat_id, bat_nom))

    labels = [f"{bid} – {bnom}" for bid, bnom in opts]
    idx_sel = st.selectbox("Building", list(range(len(opts))), format_func=lambda i: labels[i])
    bat_id, bat_nom = opts[idx_sel]

    blocks = flows.get(bat_id) or flows.get(str(bat_id)) or []
    batt_summary = _get_batt_proposed_summary(results, bat_id)
    ts_payload = _get_ts_payload(results, bat_id)

    # Existing demand + existing PV totals (from flows)
    demand_annual = _sum_block_totals(blocks, "demand_elec", "demand_elec_kWh")
    pv_exist_prod = _sum_block_totals(blocks, "pv", "pv_prod_kWh")
    pv_exist_auto = _sum_block_totals(blocks, "pv", "pv_auto_kWh")
    pv_exist_inj = _sum_block_totals(blocks, "pv", "pv_inj_kWh")

    has_demand = demand_annual > 0
    has_pv_exist = pv_exist_prod > 0

    # Proposed PV dict (already computed upstream)
    sim = pv_sim_all.get(bat_id) or pv_sim_all.get(str(bat_id))
    has_pv_sim = isinstance(sim, dict) and bool(sim)
    sim_annual = 0.0

    if has_pv_sim:
        sim_annual = float(sim.get("annual_kwh", 0.0) or 0.0)
        if sim_annual <= 0:
            has_pv_sim = False

    # ==========================================================
    # Tabs inside Variants page
    # ==========================================================
    tab_energy, tab_finance, tab_co2 = st.tabs(["⚡ Energy", "💰 Finance", "🌿 CO₂ (Scope 2)"])

    # ==========================================================
    # TAB 1 — ENERGY
    # ==========================================================
    with tab_energy:
        # ------------------------------
        # Proposed PV (roof) — energy
        # ------------------------------
        if not has_pv_sim:
            st.info("No proposed PV (roof) is available for this building.")
        else:
            # Required keys only (no fallback defaults)
            if "selfc_pct" not in sim:
                st.error("Proposed PV: missing selfc_pct in results. Re-run Phase 1 / calculations.")
            else:
                s_usable = float(sim.get("surface_usable_total_m2", 0.0) or 0.0)
                p_tot_kw = float(sim.get("p_tot_kw", 0.0) or 0.0)
                selfc_pct_param = float(sim["selfc_pct"])

                sim_selfc_param_kwh = float(sim.get("selfc_kwh", sim_annual * selfc_pct_param) or 0.0)
                sim_inj_param_kwh = float(sim.get("inj_kwh", max(sim_annual - sim_selfc_param_kwh, 0.0)) or 0.0)

                st.markdown("### 🟠 Proposed PV (roof)")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Used roof area [m²]", f"{s_usable:,.1f}")
                c2.metric("Proposed capacity [kW]", f"{p_tot_kw:,.2f}")
                c3.metric("Annual production [kWh/year]", f"{sim_annual:,.0f}")
                c4.metric("Self-consumption [%]", f"{selfc_pct_param*100:,.0f}%")

                st.markdown("#### Proposed PV flows (fixed self-consumption assumption)")

                N_PV_SIM = "PV_SIM"
                N_SELF = "SELF"
                N_GRID = "GRID"
                N_LOAD = "LOAD"

                selfc_kwh_hyp = max(float(sim_selfc_param_kwh), 0.0)
                inj_kwh_hyp = max(float(sim_inj_param_kwh), 0.0)

                # Balance closure if mismatch
                if abs((selfc_kwh_hyp + inj_kwh_hyp) - sim_annual) > 1e-6:
                    inj_kwh_hyp = max(sim_annual - selfc_kwh_hyp, 0.0)

                if has_demand:
                    self_used = min(selfc_kwh_hyp, demand_annual)
                    grid_to_load = max(demand_annual - self_used, 0.0)
                    self_unused_to_grid = max(selfc_kwh_hyp - self_used, 0.0)

                    nodes_sim = [
                        {"id": N_PV_SIM, "label": "Proposed PV"},
                        {"id": N_SELF, "label": f"Self-consumption ({int(selfc_pct_param*100)}%)"},
                        {"id": N_GRID, "label": "Grid"},
                        {"id": N_LOAD, "label": "Electric consumption"},
                    ]
                    links_sim = [
                        {"source": N_PV_SIM, "target": N_SELF, "value": selfc_kwh_hyp},
                        {"source": N_PV_SIM, "target": N_GRID, "value": inj_kwh_hyp},
                        {"source": N_SELF, "target": N_LOAD, "value": self_used},
                        {"source": N_GRID, "target": N_LOAD, "value": grid_to_load},
                    ]
                    if self_unused_to_grid > 0:
                        links_sim.append({"source": N_SELF, "target": N_GRID, "value": self_unused_to_grid})
                else:
                    nodes_sim = [
                        {"id": N_PV_SIM, "label": "Proposed PV"},
                        {"id": N_SELF, "label": f"Self-consumption ({int(selfc_pct_param*100)}%)"},
                        {"id": N_GRID, "label": "Grid"},
                    ]
                    links_sim = [
                        {"source": N_PV_SIM, "target": N_SELF, "value": selfc_kwh_hyp},
                        {"source": N_PV_SIM, "target": N_GRID, "value": inj_kwh_hyp},
                    ]

                st.plotly_chart(_make_sankey(f"{bat_nom} — Proposed PV", nodes_sim, links_sim), use_container_width=True)

                with st.expander("Selected roof planes details", expanded=False):
                    eligible = sim.get("eligible_roofs", []) or []
                    if not eligible:
                        st.info("No detailed roof plane available.")
                    else:
                        df = pd.DataFrame(eligible)
                        preferred_cols = [c for c in [
                            "orientation_deg", "orientation_code",
                            "inclinaison_deg",
                            "surface_usable_m2", "surface_utilisable_m2",
                            "p_kw",
                        ] if c in df.columns]
                        st.dataframe(df[preferred_cols] if preferred_cols else df, use_container_width=True)

                # Combined display (existing PV + proposed PV)
                if has_pv_exist:
                    st.markdown("---")
                    st.markdown("### 🔵 Existing PV + Proposed PV (combined)")

                    N_PV_EXIST = "PV_EXIST"

                    nodes_comb = [
                        {"id": N_PV_EXIST, "label": "Existing PV"},
                        {"id": N_PV_SIM, "label": "Proposed PV"},
                        {"id": N_GRID, "label": "Grid"},
                        {"id": N_LOAD, "label": "Electric consumption"},
                    ]

                    if has_demand:
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

                    st.plotly_chart(_make_sankey(f"{bat_nom} — Combined", nodes_comb, links_comb), use_container_width=True)

                    comb_prod = pv_exist_prod + sim_annual
                    comb_auto = pv_exist_auto + sim_selfc_param_kwh
                    comb_inj = pv_exist_inj + sim_inj_param_kwh

                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric("Annual demand [kWh]", f"{demand_annual:,.0f}")
                    k2.metric("Combined PV production [kWh]", f"{comb_prod:,.0f}")
                    k3.metric("Combined self-consumption [kWh]", f"{comb_auto:,.0f}")
                    k4.metric("Combined export [kWh]", f"{comb_inj:,.0f}")

        # ------------------------------
        # Proposed battery — energy (before/after)
        # ------------------------------
        st.markdown("---")
        st.markdown("## 🔋 Proposed battery — energy (before / after)")
        if not batt_summary:
            st.caption("No proposed battery detected in results for this building.")
        else:
            st.markdown("### Concrete result — Battery sizing")
            cap_total = float(batt_summary.get("capacity_kwh", 0.0) or 0.0)
            pack_caps = batt_summary.get("pack_capacities_kwh") or []
            p_target_kw = batt_summary.get("p_target_kw")
            hours_target = batt_summary.get("hours_target")
            eta_global = batt_summary.get("eta_global")

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
            c1.metric("Total capacity [kWh]", f"{cap_total:,.1f}")
            c2.metric("Number of packs [-]", f"{n_packs:d}")
            c3.metric("Average capacity / pack [kWh]", f"{cap_pack_avg:,.1f}" if n_packs > 0 else "—")
            c4.metric("Global efficiency η [-]", f"{float(eta_global):.3f}" if eta_global is not None else "—")

            d1, d2 = st.columns(2)
            d1.metric("Target power (charge/discharge) [kW]", f"{float(p_target_kw):,.1f}" if p_target_kw is not None else "—")
            d2.metric("Target hours [h]", f"{float(hours_target):.1f}" if hours_target is not None else "—")

            if pack_caps_clean:
                with st.expander("Packs details (individual capacities)", expanded=False):
                    df_packs = pd.DataFrame({
                        "Pack": list(range(1, len(pack_caps_clean) + 1)),
                        "Capacity [kWh]": pack_caps_clean,
                    })
                    st.dataframe(df_packs, use_container_width=True)

            st.markdown("### Sankey — before vs after battery")
            b_before = batt_summary.get("totals_before") or {}
            b_after = batt_summary.get("totals_after") or {}

            N_PV = "PV"
            N_GRID = "GRID"
            N_LOAD = "LOAD"
            N_BATT = "BATT"
            N_LOSS = "LOSS"

            nodes_base = [
                {"id": N_PV, "label": "PV"},
                {"id": N_GRID, "label": "Grid"},
                {"id": N_LOAD, "label": "Electric consumption"},
            ]
            links_before = [
                {"source": N_PV, "target": N_LOAD, "value": float(b_before.get("pv_to_load_kwh", 0.0) or 0.0)},
                {"source": N_GRID, "target": N_LOAD, "value": float(b_before.get("grid_to_load_kwh", 0.0) or 0.0)},
                {"source": N_PV, "target": N_GRID, "value": float(b_before.get("pv_to_grid_kwh", 0.0) or 0.0)},
            ]

            nodes_after = [
                {"id": N_PV, "label": "PV"},
                {"id": N_BATT, "label": "Battery"},
                {"id": N_LOSS, "label": "Losses"},
                {"id": N_GRID, "label": "Grid"},
                {"id": N_LOAD, "label": "Electric consumption"},
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
                st.plotly_chart(_make_sankey(f"{bat_nom} — Before battery", nodes_base, links_before, height=420), use_container_width=True)
            with cR:
                st.plotly_chart(_make_sankey(f"{bat_nom} — After battery", nodes_after, links_after, height=420), use_container_width=True)

            demand_kwh = float(b_after.get("demand_kwh", b_before.get("demand_kwh", 0.0)) or 0.0)
            pv_prod_kwh = float(b_after.get("pv_prod_kwh", b_before.get("pv_prod_kwh", 0.0)) or 0.0)
            pv_to_load_before = float(b_before.get("pv_to_load_kwh", 0.0) or 0.0)
            pv_to_load_after = float(b_after.get("pv_to_load_kwh", 0.0) or 0.0)
            batt_to_load = float(b_after.get("batt_to_load_kwh", 0.0) or 0.0)

            sc_before = (pv_to_load_before / pv_prod_kwh) if pv_prod_kwh > 0 else 0.0
            sc_after = ((pv_to_load_after + batt_to_load) / pv_prod_kwh) if pv_prod_kwh > 0 else 0.0

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Annual demand [kWh]", f"{demand_kwh:,.0f}")
            k2.metric("PV production [kWh]", f"{pv_prod_kwh:,.0f}")
            k3.metric("Self-consumption before [%]", f"{sc_before*100:,.1f}%")
            k4.metric("Self-consumption after [%]", f"{sc_after*100:,.1f}%")

            st.markdown("### Detailed time-series curves")
            if not ts_payload:
                st.info("Battery time series not available in results['timeseries_by_batiment']. Re-run the calculations.")
            else:
                idx_str = ts_payload.get("index") or []
                if not idx_str:
                    st.info("Time series: empty index.")
                else:
                    idx_dt = pd.to_datetime(idx_str)
                    idx_dt = pd.DatetimeIndex(idx_dt)

                    before = (ts_payload.get("before") or {})
                    after = (ts_payload.get("after") or {})

                    load_kwh = pd.Series(before.get("load_kWh") or [], index=idx_dt)
                    pv_kwh = pd.Series(before.get("pv_prod_kWh") or [], index=idx_dt)
                    pv_to_load = pd.Series(before.get("pv_to_load_kWh") or [], index=idx_dt)

                    pv_to_batt = after.get("pv_to_batt_kWh")
                    batt_to_load_ts = after.get("batt_to_load_kWh")
                    soc_ts = after.get("soc_kWh")

                    if len(load_kwh) == 0 or len(pv_kwh) == 0:
                        st.info("Time series: empty load/PV series.")
                    else:
                        fig1 = go.Figure()
                        fig1.add_trace(go.Scatter(x=idx_dt, y=load_kwh, mode="lines", name="Consumption [kWh]"))
                        fig1.add_trace(go.Scatter(x=idx_dt, y=pv_kwh, mode="lines", name="PV production [kWh]"))
                        fig1.add_trace(go.Scatter(x=idx_dt, y=pv_to_load, mode="lines", name="PV self-consumption [kWh]"))
                        fig1.update_layout(
                            title=f"{bat_nom} — Before battery (time detail)",
                            xaxis_title="Time",
                            yaxis_title="Energy [kWh]",
                            height=380,
                            legend_title="Flows",
                        )
                        st.plotly_chart(fig1, use_container_width=True)

                        fig2 = go.Figure()
                        fig2.add_trace(go.Scatter(x=idx_dt, y=load_kwh, mode="lines", name="Consumption [kWh]"))
                        fig2.add_trace(go.Scatter(x=idx_dt, y=pv_kwh, mode="lines", name="PV production [kWh]"))
                        fig2.add_trace(go.Scatter(x=idx_dt, y=pv_to_load, mode="lines", name="PV self-consumption direct [kWh]"))

                        if isinstance(pv_to_batt, list) and len(pv_to_batt) == len(idx_dt):
                            fig2.add_trace(go.Scatter(x=idx_dt, y=pv_to_batt, mode="lines", name="PV → Battery [kWh]"))
                        if isinstance(batt_to_load_ts, list) and len(batt_to_load_ts) == len(idx_dt):
                            fig2.add_trace(go.Scatter(x=idx_dt, y=batt_to_load_ts, mode="lines", name="Battery → Load [kWh]"))

                        fig2.update_layout(
                            title=f"{bat_nom} — After battery (time detail)",
                            xaxis_title="Time",
                            yaxis_title="Energy [kWh]",
                            height=380,
                            legend_title="Flows",
                        )
                        st.plotly_chart(fig2, use_container_width=True)

                        if isinstance(soc_ts, list) and len(soc_ts) == len(idx_dt):
                            fig_soc = go.Figure()
                            fig_soc.add_trace(go.Scatter(x=idx_dt, y=soc_ts, mode="lines", name="SOC [kWh]"))
                            fig_soc.update_layout(
                                title=f"{bat_nom} — Battery SOC",
                                xaxis_title="Time",
                                yaxis_title="SOC [kWh]",
                                height=280,
                            )
                            st.plotly_chart(fig_soc, use_container_width=True)
                            
                            # ---- Battery cycles (Equivalent Full Cycles) — variants only ----
                            cap_kwh = float(batt_summary.get("capacity_kwh", 0.0) or 0.0)
                            
                            if isinstance(batt_to_load_ts, list) and len(batt_to_load_ts) == len(idx_dt) and cap_kwh > 0:
                                discharged_kwh = float(np.nansum(np.array(batt_to_load_ts, dtype=float)))
                            
                                # Equivalent full cycles over the plotted period
                                efc = discharged_kwh / cap_kwh if cap_kwh > 0 else None
                            
                                # Optional: "per year" only if period looks like ~1 year
                                period_days = (idx_dt[-1] - idx_dt[0]).days if len(idx_dt) >= 2 else 0
                                efc_per_year = None
                                if period_days >= 330:  # approx year, avoid false precision
                                    efc_per_year = efc * (365.0 / max(period_days, 1))
                            
                                # ---- Battery cycles (Equivalent Full Cycles) — variants only ----
                                if efc_per_year is not None:
                                    st.metric("Battery equivalent full cycles [cycles/year]", f"{efc_per_year:,.0f}")
                                else:
                                    st.metric("Battery equivalent full cycles [cycles]", f"{efc:,.0f}")
                        # -------------------------------------------------------------
                        # Ranked view (Variants): sorted by PV production (full horizon)
                        # Focus on self-consumption readability (direct vs indirect)
                        # -------------------------------------------------------------
                        st.markdown("### Ranked timesteps — sorted by PV production (full horizon)")
                        st.caption(
                            "Each point is one timestep. The x-axis is a ranking (highest PV production first). "
                            "Consumption and self-consumption remain aligned to the same timestep."
                        )
                        st.info("Tip: Click on a legend item (e.g., PV production) to isolate it. Double-click to focus on a single curve/series.")

                        # Build ranked dataframe from existing time series (no recomputation)
                        df_rank_v = pd.DataFrame({
                            "timestep": idx_dt,
                            "Electric consumption [kWh]": load_kwh.values,
                            "PV production [kWh]": pv_kwh.values,
                            "PV self-consumption direct [kWh]": pv_to_load.values,
                        })
                        
                        # Battery → Load (indirect self-consumption)
                        if isinstance(batt_to_load_ts, list) and len(batt_to_load_ts) == len(idx_dt):
                            batt_to_load_s = pd.Series(batt_to_load_ts, index=idx_dt, dtype=float)
                        else:
                            batt_to_load_s = pd.Series([0.0] * len(idx_dt), index=idx_dt, dtype=float)
                        
                        df_rank_v["Battery → Load [kWh]"] = batt_to_load_s.values
                        
                        # Sort all timesteps by PV production (highest first)
                        df_rank_v = df_rank_v.sort_values(
                            "PV production [kWh]",
                            ascending=False
                        ).reset_index(drop=True)
                        
                        df_rank_v["rank"] = np.arange(1, len(df_rank_v) + 1)
                        
                        # -------------------------
                        # Plot
                        # -------------------------
                        fig_rank_pv = go.Figure()
                        
                        # Electric consumption (line)
                        fig_rank_pv.add_trace(go.Scatter(
                            x=df_rank_v["rank"],
                            y=df_rank_v["Electric consumption [kWh]"],
                            mode="lines",
                            name="Electric consumption",
                            customdata=df_rank_v[["timestep"]],
                            hovertemplate="Rank=%{x}<br>Consumption=%{y:.3f} kWh<br>Timestep=%{customdata[0]}<extra></extra>",
                        ))
                        
                        # PV production (line)
                        fig_rank_pv.add_trace(go.Scatter(
                            x=df_rank_v["rank"],
                            y=df_rank_v["PV production [kWh]"],
                            mode="lines",
                            name="PV production",
                            customdata=df_rank_v[["timestep"]],
                            hovertemplate="Rank=%{x}<br>PV=%{y:.3f} kWh<br>Timestep=%{customdata[0]}<extra></extra>",
                        ))
                        
                        # --- filter zero values for point clouds (visual clarity only) ---
                        mask_direct_pos = df_rank_v["PV self-consumption direct [kWh]"] > 0.0
                        mask_indirect_pos = df_rank_v["Battery → Load [kWh]"] > 0.0
                        
                        # Direct self-consumption (PV → Load) — RED
                        fig_rank_pv.add_trace(go.Scatter(
                            x=df_rank_v.loc[mask_direct_pos, "rank"],
                            y=df_rank_v.loc[mask_direct_pos, "PV self-consumption direct [kWh]"],
                            mode="markers",
                            name="Self-consumption (direct PV→Load)",
                            marker=dict(size=4, opacity=0.55, color="red"),
                            customdata=df_rank_v.loc[mask_direct_pos, ["timestep"]],
                            hovertemplate="Rank=%{x}<br>Direct=%{y:.3f} kWh<br>Timestep=%{customdata[0]}<extra></extra>",
                        ))
                        
                        # Indirect self-consumption (Battery → Load) — GREEN
                        fig_rank_pv.add_trace(go.Scatter(
                            x=df_rank_v.loc[mask_indirect_pos, "rank"],
                            y=df_rank_v.loc[mask_indirect_pos, "Battery → Load [kWh]"],
                            mode="markers",
                            name="Self-consumption (indirect Battery→Load)",
                            marker=dict(size=4, opacity=0.55, color="green"),
                            customdata=df_rank_v.loc[mask_indirect_pos, ["timestep"]],
                            hovertemplate="Rank=%{x}<br>Indirect=%{y:.3f} kWh<br>Timestep=%{customdata[0]}<extra></extra>",
                        ))

                        
                        fig_rank_pv.update_layout(
                            height=380,
                            margin=dict(l=40, r=20, t=40, b=40),
                            xaxis_title="Rank (sorted by PV production, highest first)",
                            yaxis_title="Energy per timestep [kWh]",
                            hovermode="x unified",
                            legend_title="Flows",
                        )
                        
                        st.plotly_chart(fig_rank_pv, use_container_width=True)

                        # -------------------------------------------------------------
                        # KPI: indirect self-consumption when PV production is zero
                        # (Battery→Load > 0 while PV production == 0)
                        # -------------------------------------------------------------
                        mask_indirect_when_pv0 = (
                            (df_rank_v["Battery → Load [kWh]"] > 0.0) &
                            (df_rank_v["PV production [kWh]"] <= 0.0)
                        )
                        
                        n_steps_indirect_when_pv0 = int(mask_indirect_when_pv0.sum())
                        kwh_indirect_when_pv0 = float(df_rank_v.loc[mask_indirect_when_pv0, "Battery → Load [kWh]"].sum())
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Timesteps with Battery→Load while PV=0", f"{n_steps_indirect_when_pv0:d}")
                        with col_b:
                            st.metric("Energy Battery→Load while PV=0", f"{kwh_indirect_when_pv0:.1f} kWh")

                        



                        st.markdown("### Summary by 3-month periods")
                        df_load_q = _agg_by_quarter(idx_dt, load_kwh)
                        df_pv_q = _agg_by_quarter(idx_dt, pv_kwh)

                        if isinstance(batt_to_load_ts, list) and len(batt_to_load_ts) == len(idx_dt):
                            auto_total = pv_to_load + pd.Series(batt_to_load_ts, index=idx_dt)
                        else:
                            auto_total = pv_to_load
                        df_auto_tot_q = _agg_by_quarter(idx_dt, auto_total)

                        dfQ = (
                            df_load_q.rename(columns={"kWh": "Consumption [kWh]"})
                            .merge(df_pv_q.rename(columns={"kWh": "PV production [kWh]"}), on="Quarter", how="outer")
                            .merge(df_auto_tot_q.rename(columns={"kWh": "Total self-consumption [kWh]"}), on="Quarter", how="outer")
                            .fillna(0.0)
                        )

                        figQ = go.Figure()
                        figQ.add_trace(go.Bar(x=dfQ["Quarter"], y=dfQ["Consumption [kWh]"], name="Consumption [kWh]"))
                        figQ.add_trace(go.Bar(x=dfQ["Quarter"], y=dfQ["PV production [kWh]"], name="PV production [kWh]"))
                        figQ.add_trace(go.Bar(x=dfQ["Quarter"], y=dfQ["Total self-consumption [kWh]"], name="Total self-consumption [kWh]"))
                        figQ.update_layout(
                            barmode="group",
                            height=380,
                            xaxis_title="Period",
                            yaxis_title="Energy [kWh]",
                            legend_title="Flows",
                        )
                        st.plotly_chart(figQ, use_container_width=True)

    # ==========================================================
    # TAB 2 — FINANCE
    # ==========================================================
    with tab_finance:
        st.markdown("## 💰 Finance (Variants)")
        st.caption("Displayed values come from results. No recomputation is performed in Phase 2.")

        # Proposed PV economics
        st.markdown("### 🟠 Proposed PV (roof) — economics")
        if not has_pv_sim:
            st.info("No proposed PV (roof) is available for this building.")
        else:
            econ = (sim.get("economics") or {})
            capex = econ.get("capex_total_CHF", None)
            opex = econ.get("opex_annual_CHF", None)
            lcoe = econ.get("lcoe_chf_kwh", None)

            try:
                capex = float(capex) if capex not in (None, "") else None
            except Exception:
                capex = None
            try:
                opex = float(opex) if opex not in (None, "") else None
            except Exception:
                opex = None
            try:
                lcoe = float(lcoe) if lcoe not in (None, "") else None
            except Exception:
                lcoe = None

            e1, e2, e3 = st.columns(3)
            e1.metric("Total CAPEX [CHF]", f"{capex:,.0f}" if capex is not None else "—")
            e2.metric("Annual OPEX [CHF/year]", f"{opex:,.0f}" if opex is not None else "—")
            e3.metric("LCOE [CHF/kWh]", f"{lcoe:.3f}" if lcoe is not None else "—")

            st.markdown("#### PV profitability (cumulative)")
            years = econ.get("years")
            cashflow_cum = econ.get("cashflow_cum_CHF")
            if isinstance(years, list) and isinstance(cashflow_cum, list) and len(years) == len(cashflow_cum):
                st.plotly_chart(_bar_cum_cashflow(years, cashflow_cum, f"{bat_nom} — Proposed PV (cumul)"), use_container_width=True)

                payback_year = econ.get("payback_year", None)
                benefit_25 = float(econ.get("benefit_25_CHF", 0.0) or 0.0)

                ru_amount = econ.get("ru_amount_CHF", None)
                try:
                    ru_amount = float(ru_amount) if ru_amount not in (None, "") else None
                except Exception:
                    ru_amount = None

                r1, r2, r3 = st.columns(3)
                r1.metric("RU after 1 year [CHF]", f"{ru_amount:,.0f}" if ru_amount is not None else "—")
                r1.caption("Estimated period: April 2025 → April 2026")
                r2.metric("Payback", f"Year {payback_year}" if payback_year is not None else "—")
                r3.metric("Benefit at 25 years [CHF]", f"{benefit_25:,.0f}")
            else:
                st.info("No cumulative cashflow available in economics (years / cashflow_cum_CHF).")

        # Proposed battery economics
        st.markdown("---")
        st.markdown("### 🔋 Proposed battery — economics")
        stor_econ = (results.get("storage_economics_by_batiment") or {}).get(bat_id) \
                    or (results.get("storage_economics_by_batiment") or {}).get(str(bat_id)) \
                    or []
        econ_b = None
        for it in stor_econ:
            if (it.get("mode") == "proposed") and (it.get("engine") == "bat_li_ion"):
                econ_b = it.get("economics") or {}
                break

        if not econ_b:
            st.info("No proposed battery economics available. Re-run the calculations.")
        else:
            years = econ_b.get("years")
            cashflow_cum = econ_b.get("cashflow_cum_CHF")
            if isinstance(years, list) and isinstance(cashflow_cum, list) and len(years) == len(cashflow_cum):
                st.plotly_chart(_bar_cum_cashflow(years, cashflow_cum, f"{bat_nom} — Battery (cumul)"), use_container_width=True)

                payback_year = econ_b.get("payback_year", None)
                benefit_h = float(econ_b.get("benefit_horizon_CHF", econ_b.get("benefit_25_CHF", 0.0)) or 0.0)

                capex_b = econ_b.get("capex_total_CHF", None)
                try:
                    capex_b = float(capex_b) if capex_b not in (None, "") else None
                except Exception:
                    capex_b = None

                m1, m2, m3 = st.columns(3)
                m1.metric("CAPEX [CHF]", f"{capex_b:,.0f}" if capex_b is not None else "—")
                m2.metric("Payback", f"Year {payback_year}" if payback_year is not None else "—")
                m3.metric("Horizon benefit [CHF]", f"{benefit_h:,.0f}")
            else:
                st.info("Battery economics: missing years/cashflow_cum_CHF.")

     # ==========================================================
    # TAB — CO2 (Scope 2) — Before vs After (read results only)
    # ==========================================================
    with tab_co2:
        st.markdown("## CO₂ (Scope 2) — Before vs Proposed")
        _show_variants_co2_scope2(results, bat_id, bat_nom, pv_sim_all=pv_sim_all)

        with st.expander("Proof of data (read-only from results)", expanded=False):
            co2_root = (results.get("co2") or {})
            by_bat = (co2_root.get("by_batiment") or {})
            b = by_bat.get(bat_id) or by_bat.get(str(bat_id)) or {}
            scope2 = (b.get("scope2") or {})
            tot = (scope2.get("totals") or {})
            aft = (scope2.get("totals_after") or {})
        
            st.write("grid_import_kWh before:", tot.get("grid_import_kWh"))
            st.write("grid_import_kWh after :", aft.get("grid_import_kWh"))
            st.write("pv_self_kWh before:", tot.get("pv_self_kWh"))
            st.write("pv_self_kWh after :", aft.get("pv_self_kWh"))
        
            # si tu as battery_proposed_by_batiment
            bp = (results.get("battery_proposed_by_batiment") or {})
            bs = bp.get(bat_id) or bp.get(str(bat_id)) or {}
            ta = (bs.get("totals_after") or {})
            st.write("battery totals_after.batt_to_load_kwh:", ta.get("batt_to_load_kwh"))
            st.write("battery totals_after.pv_to_batt_kwh:", ta.get("pv_to_batt_kwh"))
            
        batt_to_load = ta.get("batt_to_load_kwh") if isinstance(ta, dict) else None
        pv_to_batt = ta.get("pv_to_batt_kwh") if isinstance(ta, dict) else None
        
        try:
            batt_to_load = float(batt_to_load) if batt_to_load is not None else None
        except Exception:
            batt_to_load = None
        try:
            pv_to_batt = float(pv_to_batt) if pv_to_batt is not None else None
        except Exception:
            pv_to_batt = None
        
        if (batt_to_load == 0.0) and (pv_to_batt == 0.0):
            st.info(
                "No battery dispatch detected in the proposed variant (PV→Battery = 0, Battery→Load = 0). "
                "Therefore, grid imports and Scope 2 CO₂ remain unchanged."
            )

# def show_decision(project: dict):
#     """
#     Phase 2 decision support:
#     - READ-ONLY from results (no physical recomputation)
#     - Compare Baseline vs Proposed (if available)
#     - Multi-criteria scoring with min-max normalization
#     """
#     results = project.get("results", {}) or {}

#     st.markdown("## 🧠 Decision / Comparison")
#     st.caption("Decision support based on already computed KPIs (results only).")

#     # ------------------------------------------------------------------
#     # 1) Collect building IDs robustly (NO assumption on project['batiments'])
#     # ------------------------------------------------------------------
#     batiments = project.get("batiments", []) or []
#     name_by_id = {}
#     for i, b in enumerate(batiments):
#         bid = b.get("id") or b.get("batiment_id") or i
#         name_by_id[bid] = b.get("nom") or f"Building {i+1}"

#     flows_by_bat = ((results.get("flows") or {}).get("batiments") or {}) or {}
#     co2_by_bat = ((results.get("co2") or {}).get("by_batiment") or {}) or {}
#     ex_by_bat = ((results.get("exergy") or {}).get("by_batiment") or {}) or {}
#     econ_by_bat = (results.get("economics_by_batiment") or {}) or {}
#     stor_by_bat = (results.get("storage_economics_by_batiment") or {}) or {}

#     bat_ids = set()
#     bat_ids.update(list(flows_by_bat.keys()))
#     bat_ids.update(list(co2_by_bat.keys()))
#     bat_ids.update(list(ex_by_bat.keys()))
#     bat_ids.update(list(econ_by_bat.keys()))
#     bat_ids.update(list(stor_by_bat.keys()))
#     bat_ids.update(list(name_by_id.keys()))

#     if not bat_ids:
#         st.warning("No buildings found in results. Run calculations in Phase 1.")
#         return

#     # Keep stable display order
#     def _sort_key(x):
#         try:
#             return (0, int(x))
#         except Exception:
#             return (1, str(x))

#     bat_ids_sorted = sorted(list(bat_ids), key=_sort_key)

#     # Display labels
#     def _label(bid):
#         # Try name map (int keys), else fallback
#         if bid in name_by_id:
#             return f"{bid} – {name_by_id[bid]}"
#         # Try string/int cross
#         if str(bid) in name_by_id:
#             return f"{bid} – {name_by_id[str(bid)]}"
#         if isinstance(bid, str):
#             try:
#                 ib = int(bid)
#                 if ib in name_by_id:
#                     return f"{bid} – {name_by_id[ib]}"
#             except Exception:
#                 pass
#         return f"{bid} – Building {bid}"

#     idx = st.selectbox(
#         "Building",
#         options=list(range(len(bat_ids_sorted))),
#         format_func=lambda i: _label(bat_ids_sorted[i]),
#         index=0,
#         key="decision_bat_select",
#     )
#     bat_id = bat_ids_sorted[idx]

#     # Convenience getters with int/str fallback
#     def _get(d: dict, k):
#         return d.get(k) or d.get(str(k)) or d.get(int(k)) if isinstance(k, str) and k.isdigit() else d.get(k)

#     # ------------------------------------------------------------------
#     # 2) Extract KPIs (baseline / proposed) from results ONLY
#     # ------------------------------------------------------------------
#     # Energy + CO2 from results["co2"]["by_batiment"][bat_id]["scope2"]["totals|totals_after"]
#     co2_root = results.get("co2") or {}
#     by_bat = co2_root.get("by_batiment") or {}
#     co2_b = _get(by_bat, bat_id) or {}
#     scope2 = co2_b.get("scope2") or {}
#     tot = scope2.get("totals") or {}
#     tot_after = scope2.get("totals_after") or {}

#     # Proposed exists if totals_after is a non-empty dict
#     has_proposed = isinstance(tot_after, dict) and len(tot_after) > 0

#     # KPI candidates (confirmed in core/co2.py outputs)
#     grid_import_base = tot.get("grid_import_kWh", None)
#     grid_import_prop = tot_after.get("grid_import_kWh", None) if has_proposed else None

#     co2_base = tot.get("system_total_kgCO2e", None)
#     co2_prop = tot_after.get("system_total_kgCO2e", None) if has_proposed else None

#     # Economy (generic): use storage economics if available (benefit vs baseline)
#     # Baseline benefit is defined as 0 by convention (reference)
#     benefit_base = 0.0
#     benefit_prop = None
#     stor_list = _get(stor_by_bat, bat_id) or []
#     if isinstance(stor_list, list):
#         best = None
#         for it in stor_list:
#             if not isinstance(it, dict):
#                 continue
#             if it.get("mode") != "proposed":
#                 continue
#             econ = it.get("economics") or {}
#             # keys used in show_economics(): benefit_horizon_CHF fallback benefit_25_CHF
#             b = econ.get("benefit_horizon_CHF")
#             if b is None:
#                 b = econ.get("benefit_25_CHF")
#             if b is None:
#                 continue
#             try:
#                 b = float(b)
#             except Exception:
#                 continue
#             best = b if best is None else max(best, b)
#         benefit_prop = best

#     # Exergy: results["exergy"]["by_batiment"][bat_id]["heating"]["eta_ex_global"]
#     ex_root = results.get("exergy") or {}
#     ex_by = ex_root.get("by_batiment") or {}
#     ex_b = _get(ex_by, bat_id) or {}
#     heating = ex_b.get("heating") or {}
#     ex_base = heating.get("eta_ex_global", None)
#     # If you later store an "after" exergy, extend here. For now: no assumption.
#     ex_prop = None

#     # ------------------------------------------------------------------
#     # 3) Build comparison table (baseline + proposed if available)
#     # ------------------------------------------------------------------
#     rows = [{
#         "Variant": "Baseline",
#         "grid_import_kWh": grid_import_base,
#         "system_total_kgCO2e": co2_base,
#         "benefit_CHF": benefit_base,
#         "eta_ex_global": ex_base,
#     }]

#     if has_proposed:
#         rows.append({
#             "Variant": "Proposed (after)",
#             "grid_import_kWh": grid_import_prop,
#             "system_total_kgCO2e": co2_prop,
#             "benefit_CHF": benefit_prop,
#             "eta_ex_global": ex_prop,
#         })

#     df = pd.DataFrame(rows)

#     if len(df) < 2:
#         st.info("No proposed variant available for this building. Add a proposed system in Phase 1 to compare.")
#         st.dataframe(df, use_container_width=True)
#         return

#     # ------------------------------------------------------------------
#     # 4) Weights UI (no hardcoding; user-driven)
#     # ------------------------------------------------------------------
#     st.markdown("### Weights")
#     c1, c2, c3, c4 = st.columns(4)
#     w_energy = c1.slider("Energy (min)", 0.0, 1.0, 0.25, 0.01, key="decision_w_energy")
#     w_co2 = c2.slider("CO₂ (min)", 0.0, 1.0, 0.25, 0.01, key="decision_w_co2")
#     w_econ = c3.slider("Economy (max)", 0.0, 1.0, 0.25, 0.01, key="decision_w_econ")
#     w_ex = c4.slider("Exergy (max)", 0.0, 1.0, 0.25, 0.01, key="decision_w_exergy")

#     w_sum = w_energy + w_co2 + w_econ + w_ex
#     if w_sum <= 0:
#         st.warning("All weights are zero. Set at least one weight.")
#         st.dataframe(df, use_container_width=True)
#         return

#     weights = {
#         "energy": w_energy / w_sum,
#         "co2": w_co2 / w_sum,
#         "economy": w_econ / w_sum,
#         "exergy": w_ex / w_sum,
#     }

#     # ------------------------------------------------------------------
#     # 5) Criteria config (only what we actually have)
#     # ------------------------------------------------------------------
#     criteria = {
#         "energy": {"col": "grid_import_kWh", "sense": "min"},
#         "co2": {"col": "system_total_kgCO2e", "sense": "min"},
#         "economy": {"col": "benefit_CHF", "sense": "max"},
#         "exergy": {"col": "eta_ex_global", "sense": "max"},
#     }

#     # Keep only criteria available for ALL variants (robust + defendable)
#     active = []
#     excluded = []
#     for k, meta in criteria.items():
#         col = meta["col"]
#         if col in df.columns and df[col].notna().all():
#             active.append(k)
#         else:
#             excluded.append(k)

#     if excluded:
#         st.warning(f"Excluded criteria due to missing data: {', '.join(excluded)}")

#     if not active:
#         st.info("No common criteria across variants. Cannot compute a global score.")
#         st.dataframe(df, use_container_width=True)
#         return

#     # ------------------------------------------------------------------
#     # 6) Min-max normalization -> score in [0..1], 1 = best
#     # ------------------------------------------------------------------
#     def _minmax_score(series: pd.Series, sense: str) -> pd.Series:
#         s = pd.to_numeric(series, errors="coerce")
#         mn = float(s.min())
#         mx = float(s.max())
#         if abs(mx - mn) < 1e-12:
#             return pd.Series([0.5] * len(s), index=s.index)
#         if sense == "min":
#             return (mx - s) / (mx - mn)
#         return (s - mn) / (mx - mn)

#     for k in active:
#         col = criteria[k]["col"]
#         sense = criteria[k]["sense"]
#         df[f"score_{k}"] = _minmax_score(df[col], sense)

#     # Renormalize weights over active criteria
#     w_act_sum = sum(weights[k] for k in active)
#     w_act = {k: weights[k] / w_act_sum for k in active}

#     df["score_global"] = 0.0
#     for k in active:
#         df["score_global"] += df[f"score_{k}"] * float(w_act[k])

#     df = df.sort_values("score_global", ascending=False).reset_index(drop=True)

#     # ------------------------------------------------------------------
#     # 7) Output: table + charts
#     # ------------------------------------------------------------------
#     st.markdown("### Comparison table")
#     cols = ["Variant"]
#     cols += [criteria[k]["col"] for k in active]
#     cols += [f"score_{k}" for k in active]
#     cols += ["score_global"]
#     st.dataframe(df[cols], use_container_width=True)

#     st.markdown("### Global score")
#     fig = go.Figure()
#     fig.add_trace(go.Bar(x=df["Variant"], y=df["score_global"], name="Global score"))
#     fig.update_layout(height=320, yaxis_title="Score [-] (higher is better)")
#     st.plotly_chart(fig, use_container_width=True)

#     st.markdown("### Weighted contributions")
#     fig2 = go.Figure()
#     for k in active:
#         fig2.add_trace(go.Bar(
#             x=df["Variant"],
#             y=df[f"score_{k}"] * float(w_act[k]),
#             name=f"{k} (w={w_act[k]:.2f})"
#         ))
#     fig2.update_layout(barmode="stack", height=360, yaxis_title="Weighted contribution [-]")
#     st.plotly_chart(fig2, use_container_width=True)

#     st.info(
#         "Notes: min-max scaling is computed only over the displayed variants; "
#         "ranking depends on chosen weights."
#     )




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
