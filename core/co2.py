# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 10:33:47 2026

@author: infor
"""

# core/co2.py
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, List
import numpy as np


def compute_co2_results(project: Dict[str, Any], results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute CO2 results and return a dict to be stored in results["co2"].

    Contract implemented (PV hybrid: grid avoided minus PV emissions):
    - avoided_by_pv_for_consumption_kgCO2e = pv_self_kWh * (eco2_grid - eco2_pv)
    - diff_total_vs_no_pv_kgCO2e = avoided_by_pv_for_consumption_kgCO2e - pv_export_kWh * eco2_pv

    Notes:
    - grid factor comes from project["params"]["grid_import_factor_kg_per_kWh"] (no hardcoding)
    - PV factor is read from PV flow block meta.techno_params_used["CO2 emissions"]["Values"] (if present)
    - Phase 2 must only read results (no recomputation there)
    """
    import numpy as np
    from typing import Any, Dict, List, Optional, Tuple

    params = project.get("params", {}) or {}

    grid_factor = params.get("grid_import_factor_kg_per_kWh", None)
    if grid_factor is None:
        raise ValueError("Missing parameter: params.grid_import_factor_kg_per_kWh (kgCO2e/kWh).")
    try:
        grid_factor = float(grid_factor)
    except Exception:
        raise ValueError("Invalid parameter: params.grid_import_factor_kg_per_kWh must be numeric.")

    # Optional traceability fields if you later add them in Phase 1
    grid_factor_source = params.get("grid_import_factor_source", None)  # e.g. "KBOB"
    grid_factor_notes = params.get("grid_import_factor_notes", None)

    flows_by_bat = ((results.get("flows") or {}).get("batiments") or {}) or {}
    ts_by_bat = (results.get("timeseries_by_batiment") or {}) or {}

    out: Dict[str, Any] = {
        "meta": {
            "method": "scope2_operational_electricity_only_plus_pv_tech_factor",
            "export_credit": "none",  # explicit: no negative credit for exports in Scope 2
            "factors": {
                "grid_import_factor_kg_per_kWh": grid_factor,
                "source": grid_factor_source,
            },
            "notes": grid_factor_notes,
            "units": {
                "grid_import_kWh": "kWh",
                "grid_export_kWh": "kWh",
                "net_kgCO2e": "kgCO2e",
            },
        },
        "global": {"scope1": {"totals": {}}, "scope2": {"totals": {}}},
        "by_batiment": {},
        "timeseries_by_batiment": {},
        "tech_factors": _collect_raw_tech_factors(project),
    }

    # ---- helpers ----
    def _get_global_block_totals(blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        # prefer energy_global_with_storage if present
        for b in blocks or []:
            if b.get("type") == "energy_global_with_storage":
                return b.get("totals", {}) or {}
        for b in blocks or []:
            if b.get("type") == "energy_global":
                return b.get("totals", {}) or {}
        return {}

    def _get_measured_ts_payload(bat_ts: Dict[str, Any]) -> Tuple[List[Any], Dict[str, Any]]:
        idx = bat_ts.get("index") or []
        measured = (bat_ts.get("measured") or {})
        return idx, measured

    def _get_after_ts_payload_if_any(bat_ts: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # New recommended format: bat_ts["battery_proposed"]["after"]
        bp = bat_ts.get("battery_proposed")
        if isinstance(bp, dict):
            after = bp.get("after")
            if isinstance(after, dict):
                return after

        # Legacy/fallback format: bat_ts["after"]
        after = bat_ts.get("after")
        if isinstance(after, dict):
            return after

        return None

    def _safe_float_list(v: Any, n: int) -> Optional[List[float]]:
        if v is None:
            return None
        if not isinstance(v, list):
            return None
        if len(v) != n:
            return None
        outv = []
        for x in v:
            try:
                outv.append(float(x))
            except Exception:
                outv.append(np.nan)
        return outv
    
    def _find_pv_factor_kg_per_kWh_from_blocks(blocks: List[Dict[str, Any]]) -> Optional[float]:
        """
        PV factor in kgCO2e/kWh produced, stored in PV flow block meta.techno_params_used['CO2 emissions']['Values'].
        Returns None if not found.
        """
        for b in (blocks or []):
            if b.get("type") != "pv":
                continue
            meta = b.get("meta") or {}
            techno_params = meta.get("techno_params_used") or {}
            co2_item = techno_params.get("CO2 emissions") or {}
            try:
                if isinstance(co2_item, dict):
                    v = co2_item.get("Values")
                else:
                    v = None
                return float(v) if v is not None else None
            except Exception:
                return None
        return None

    def _sum_storage_avoided_import_and_losses_kWh(
        blocks: List[Dict[str, Any]], bat_id_any: Any
    ) -> Tuple[float, float]:
        """
        PV-only measured storage:
        - avoided_import_kWh should match Battery -> Load (delivered energy)
        - losses_kWh should match Battery -> Losses
        Source of truth: storage_elec flow block totals.
        Fallback: impacts.delta_grid_import_kWh if totals missing.

        Returns: (avoided_import_kWh, losses_kWh)
        """
        avoided = 0.0
        losses = 0.0

        for b in (blocks or []):
            if b.get("type") != "storage_elec":
                continue

            meta = b.get("meta") or {}
            if str(meta.get("batiment_id")) != str(bat_id_any):
                continue
            if (meta.get("mode") or "").lower() != "measured":
                continue

            # --- Prefer totals (Sankey-consistent) ---
            tot = b.get("totals") or {}
            btol = tot.get("batt_to_load_kWh", None)
            bloss = tot.get("batt_losses_kWh", None)

            got_any = False
            try:
                if btol is not None:
                    avoided += float(btol or 0.0)
                    got_any = True
            except Exception:
                pass

            try:
                if bloss is not None:
                    losses += float(bloss or 0.0)
                    got_any = True
            except Exception:
                pass

            # --- Fallback: impacts (only if totals absent) ---
            if not got_any:
                impacts = b.get("impacts") or {}
                try:
                    d = float(impacts.get("delta_grid_import_kWh", 0.0) or 0.0)
                except Exception:
                    d = 0.0
                # In your current model: positive means avoided
                avoided += max(d, 0.0)

        return float(avoided), float(losses)

    
    def _find_first_elec_storage_cfg(project: Dict[str, Any], bat_id_any: Any) -> Optional[Dict[str, Any]]:
        """
        Return first electric storage config for this building from Phase 1 structure.
        """
        for ouv in (project.get("ouvrages") or []):
            if str(ouv.get("batiment_id")) != str(bat_id_any):
                continue
            for stor in (ouv.get("stockages") or []):
                tg = stor.get("type_general")
                if (tg == "Electric") or (isinstance(tg, str) and tg.lower() == "electric"):
                    return stor
        return None
    
    def _get_storage_capacity_kWh(stor_cfg: Dict[str, Any]) -> float:
        # Phase 1 already stores capacity_kwh for electric storage
        for k in ("capacity_kwh", "capacity_kWh", "capacity_total_kwh", "capacity_total_kWh"):
            v = stor_cfg.get(k, None)
            if v not in (None, "", 0):
                try:
                    return float(v)
                except Exception:
                    pass
        return 0.0
    
    def _get_storage_embodied_factor_kg_per_kWhcap(stor_cfg: Dict[str, Any]) -> float:
        """
        Storage embodied factor in kgCO2e/kWhcap (kWh refers to CAPACITY).
        Stored in stor_cfg['parametres']['CO2 emissions']['Values'] (from Excel).
        """
        params_any = (stor_cfg.get("parametres") or stor_cfg.get("Parameters") or {}) or {}
        item = params_any.get("CO2 emissions")
        try:
            if isinstance(item, dict):
                v = item.get("Values")
            else:
                v = item
            return float(v) if v not in (None, "", 0) else 0.0
        except Exception:
            return 0.0
    

    def _get_pv_factor_from_flow_blocks(blocks: List[Dict[str, Any]]) -> Optional[float]:
        """
        PV tech/LCA factor (kgCO2e/kWh PV) from PV flow block meta.techno_params_used["CO2 emissions"]["Values"].
        Returns None if not found.
        """
        for bb in (blocks or []):
            if bb.get("type") != "pv":
                continue
            pv_meta = bb.get("meta") or {}
            techno_params = pv_meta.get("techno_params_used") or {}
            co2_item = techno_params.get("CO2 emissions") or {}
            try:
                v = co2_item.get("Values") if isinstance(co2_item, dict) else None
                if v is None:
                    continue
                return float(v)
            except Exception:
                continue
        return None

    # ---- per building ----
    global_import_kWh = 0.0
    global_export_kWh = 0.0
    global_net_kg = 0.0
    global_scope1_kg = 0.0
    global_fuel_in_kWh = 0.0
    global_heat_out_th_kWh = 0.0


    for bat_id, blocks in flows_by_bat.items():
        totals = _get_global_block_totals(blocks or [])

        grid_import_kWh = float(totals.get("grid_to_load_kWh", 0.0) or 0.0)
        grid_export_kWh = float(totals.get("pv_to_grid_kWh", 0.0) or 0.0)

        # PV kWh (self + export) from global totals
        pv_self_kWh = float(totals.get("pv_auto_used_kWh", 0.0) or 0.0)
        pv_export_kWh = float(totals.get("pv_to_grid_kWh", 0.0) or 0.0)

        # Factors
        pv_factor = _find_pv_factor_kg_per_kWh_from_blocks(blocks or [])

        # Fallback: proposed PV techno factor stored in results["pv_proposed_by_batiment"][bat_id]
        if pv_factor is None:
            try:
                pp_all = (results.get("pv_proposed_by_batiment") or {})
                pp = pp_all.get(bat_id) or pp_all.get(str(bat_id)) or {}
                v = pp.get("pv_factor_kg_per_kWh", None)
                pv_factor = float(v) if v not in (None, "", 0) else None
            except Exception:
                pv_factor = None


        # Fallback (no hardcoding): allow providing PV factor from project params
        # (e.g. Phase 1 later, or techno import). If missing, keep None.
        if pv_factor is None:
            try:
                pv_factor = params.get("pv_factor_kg_per_kWh", None)
                pv_factor = float(pv_factor) if pv_factor is not None else None
            except Exception:
                pv_factor = None

        pv_factor_for_delta = float(pv_factor) if pv_factor is not None else 0.0



        # --- Scope 2 net (grid only, no export credit) ---
        net_kg = grid_import_kWh * grid_factor

        # --- PV CO2 storytelling pieces ---
        pv_self_grid_avoided_kg = pv_self_kWh * grid_factor
        pv_self_pv_cost_kg = pv_self_kWh * (float(pv_factor) if pv_factor is not None else 0.0)
        pv_export_pv_cost_kg = pv_export_kWh * (float(pv_factor) if pv_factor is not None else 0.0)

        avoided_by_pv_for_consumption_kg = pv_self_kWh * max((grid_factor - pv_factor_for_delta), 0.0)
        # Your definition (corrected): total difference vs no PV = avoided on self-consumption - PV export PV cost
        diff_total_vs_no_pv_kg = avoided_by_pv_for_consumption_kg - pv_export_kWh * (float(pv_factor) if pv_factor is not None else 0.0)

        # --- Storage (measured) avoided imports ---
        avoided_by_storage_kWh, storage_losses_kWh = _sum_storage_avoided_import_and_losses_kWh(blocks or [], bat_id)
        avoided_by_storage_kg = avoided_by_storage_kWh * max((grid_factor - pv_factor_for_delta), 0.0)
        
        # --- Battery embodied (installation) — MEASURED ONLY ---
        # IMPORTANT: baseline/show_CO2 must not use proposed battery specs.
        stor_cfg = _find_first_elec_storage_cfg(project, bat_id)
        
        batt_capacity_kWh = _get_storage_capacity_kWh(stor_cfg) if isinstance(stor_cfg, dict) else 0.0
        batt_factor_kg_per_kWhcap = _get_storage_embodied_factor_kg_per_kWhcap(stor_cfg) if isinstance(stor_cfg, dict) else 0.0
        batt_embodied_kg = float(batt_capacity_kWh) * float(batt_factor_kg_per_kWhcap)
        
        batt_lifetime_years = 0.0
        try:
            p = (stor_cfg.get("parametres") or stor_cfg.get("Parameters") or {}) if isinstance(stor_cfg, dict) else {}
            life_item = p.get("Lifetime")
            if isinstance(life_item, dict):
                batt_lifetime_years = float(life_item.get("Values") or 0.0)
            elif life_item is not None:
                batt_lifetime_years = float(life_item)
        except Exception:
            batt_lifetime_years = 0.0
        
        batt_embodied_annualized_kg = 0.0
        if batt_embodied_kg > 0 and batt_lifetime_years > 0:
            batt_embodied_annualized_kg = float(batt_embodied_kg) / float(batt_lifetime_years)


        
        # Annualize embodied CO2 for annual charts (kgCO2e/year)
        batt_lifetime_years = 0.0
        try:
            # read from storage techno params if present
            p = (stor_cfg.get("parametres") or stor_cfg.get("Parameters") or {}) if isinstance(stor_cfg, dict) else {}
            life_item = p.get("Lifetime")
            if isinstance(life_item, dict):
                batt_lifetime_years = float(life_item.get("Values") or 0.0)
            elif life_item is not None:
                batt_lifetime_years = float(life_item)
        except Exception:
            batt_lifetime_years = 0.0
        
        batt_embodied_annualized_kg = 0.0
        if batt_embodied_kg > 0 and batt_lifetime_years and batt_lifetime_years > 0:
            batt_embodied_annualized_kg = float(batt_embodied_kg) / float(batt_lifetime_years)


        # --- CO2 payback curve (linear) ---
        horizon = int(params.get("analysis_horizon_years", 25) or 25)
        avoided_per_year_kg = float(avoided_by_storage_kg)
        payback_years = None
        if batt_embodied_kg > 0 and avoided_per_year_kg > 0:
            payback_years = float(batt_embodied_kg / avoided_per_year_kg)

        years_curve = None
        cum_curve = None
        if horizon > 0 and batt_embodied_kg > 0 and avoided_per_year_kg > 0:
            years_curve = list(range(0, horizon + 1))
            cum_curve = [(-batt_embodied_kg + y * avoided_per_year_kg) for y in years_curve]


    
        # Components for Phase 2 bars (no recompute there)
        # Convention: positive = emissions/costs, negative = benefits/savings
        demand_elec_kWh = float(totals.get("demand_elec_kWh", 0.0) or 0.0)
        grid_baseline_kg = demand_elec_kWh * grid_factor
        
        # Net PV benefit on self-consumption = avoided grid - PV operational cost (if pv_factor available)
        pv_net_benefit_selfc_kg = float(pv_self_grid_avoided_kg or 0.0) - float(pv_self_pv_cost_kg or 0.0)
        
        components_kg = {
            # Baseline grid emissions (stable reference for decomposition)
            "grid_import": float(grid_baseline_kg or 0.0),
        
            # Benefit (negative in stack)
            "pv_net_benefit_self_consumption": -float(pv_net_benefit_selfc_kg or 0.0),
        
            # Export PV cost (positive in stack)
            "pv_export_pv_cost": float(pv_export_pv_cost_kg or 0.0),
        
            # Battery embodied (annualized)
            "battery_embodied": float(batt_embodied_annualized_kg or 0.0),
        }

        
        system_total_kgCO2e = float(sum(float(v or 0.0) for v in components_kg.values()))

        # Keep attribution separately (still used for payback/table)
        battery_avoided_kgCO2e = float(avoided_by_storage_kg or 0.0)

        
        # NEW: “system total” consistent with Phase 2 stacked bars (annual)

        
        # ==========================================================
        # PROPOSED / AFTER (battery proposed) — annual totals for Phase 2
        # Source of truth: results["battery_proposed_by_batiment"][bat_id]["totals_after"]
        # (Phase 2 must not recompute)
        # ==========================================================
        totals_after_scope2 = {}
        
        bp_all = (results.get("battery_proposed_by_batiment") or {})
        bp = bp_all.get(bat_id) or bp_all.get(str(bat_id)) or {}
        bp_after = (bp.get("totals_after") or {}) if isinstance(bp, dict) else {}
        
        # If no battery proposed "after" totals exist, fall back to PV proposed "after" totals.
        pp_all = (results.get("pv_proposed_by_batiment") or {})
        pp = pp_all.get(bat_id) or pp_all.get(str(bat_id)) or {}
        pp_after = (pp.get("totals_after") or {}) if isinstance(pp, dict) else {}
        
        after_src = bp_after if (isinstance(bp_after, dict) and bp_after) else (pp_after if (isinstance(pp_after, dict) and pp_after) else {})
        
        # ----------------------------------------------------------
        # Fallback: PV proposed exists but totals_after missing
        # → reconstruct a minimal "after_src" from pv_proposed fields
        # (still CORE, Phase 2 must not recompute)
        # ----------------------------------------------------------
        if (not after_src) and isinstance(pp, dict) and pp:
            # Need annual electric demand (kWh) from results["flows"]["batiments"]
            demand_elec_kWh = 0.0
            try:
                fb_list = ((results.get("flows") or {}).get("batiments") or {}).get(bat_id) \
                          or ((results.get("flows") or {}).get("batiments") or {}).get(str(bat_id)) \
                          or []
                for fb in fb_list:
                    if isinstance(fb, dict) and fb.get("type") == "energy_global":
                        demand_elec_kWh = float(((fb.get("totals") or {}).get("demand_elec_kWh")) or 0.0)
                        break
            except Exception:
                demand_elec_kWh = 0.0
        
            pv_self_kWh = float(pp.get("selfc_kwh", 0.0) or 0.0)
            pv_inj_kWh = float(pp.get("inj_kwh", 0.0) or 0.0)
        
            pv_to_load_kWh = float(min(pv_self_kWh, demand_elec_kWh))
            grid_to_load_kWh = float(max(demand_elec_kWh - pv_to_load_kWh, 0.0))
            pv_to_grid_kWh = float(max(pv_inj_kWh, 0.0))
        
            after_src = {
                "grid_to_load_kwh": grid_to_load_kWh,
                "pv_to_load_kwh": pv_to_load_kWh,
                "pv_to_grid_kwh": pv_to_grid_kWh,
                "batt_to_load_kwh": 0.0,
                "pv_prod_kwh": float(pp.get("annual_kwh", 0.0) or 0.0),
            }

        
        batt_embodied_kg_after = float(bp.get("batt_embodied_kgCO2e", 0.0) or 0.0)
        batt_embodied_annual_after = float(bp.get("batt_embodied_annualized_kgCO2e", 0.0) or 0.0)
        
        if isinstance(after_src, dict) and after_src:
            # kWh from results (already computed in core)
            grid_import_after_kWh = float(after_src.get("grid_to_load_kwh", 0.0) or 0.0)
            pv_to_load_after_kWh = float(after_src.get("pv_to_load_kwh", 0.0) or 0.0)
            batt_to_load_after_kWh = float(after_src.get("batt_to_load_kwh", 0.0) or 0.0)
            pv_export_after_kWh = float(after_src.get("pv_to_grid_kwh", 0.0) or 0.0)
        

        
            # Scope 2 net (grid only, no export credit)
            net_after_kg = grid_import_after_kWh * grid_factor
        
            # PV self-consumption "effective" (PV direct + PV via battery)
            pv_self_after_kWh = pv_to_load_after_kWh + batt_to_load_after_kWh
        
            # PV pieces (same convention as before)
            pv_self_grid_avoided_after_kg = pv_self_after_kWh * grid_factor
            pv_self_pv_cost_after_kg = pv_self_after_kWh * (float(pv_factor) if pv_factor is not None else 0.0)
            pv_export_pv_cost_after_kg = pv_export_after_kWh * (float(pv_factor) if pv_factor is not None else 0.0)
        
            avoided_by_pv_for_consumption_after_kg = pv_self_after_kWh * max((grid_factor - pv_factor_for_delta), 0.0)
        
            # Battery benefit in "after" = delivered from battery to load (kWh) valued with same delta factor
            battery_avoided_after_kg = batt_to_load_after_kWh * max((grid_factor - pv_factor_for_delta), 0.0)
            
            # --- CO2 payback curve (proposed) — with replacements at end-of-life ---
            horizon = int(params.get("analysis_horizon_years", 25) or 25)
            
            payback_years_after = None
            years_curve_after = None
            cum_curve_after = None
            
            # lifetime (years) comes from results["battery_proposed_by_batiment"][bat_id]
            batt_life_y = 0
            try:
                batt_life_y = int(float(bp.get("batt_lifetime_years", 0.0) or 0.0))
            except Exception:
                batt_life_y = 0
            
            if horizon > 0 and batt_embodied_kg_after > 0 and battery_avoided_after_kg > 0:
                years_curve_after = list(range(0, horizon + 1))
                cum_curve_after = []
                for y in years_curve_after:
                    # replacements at y = lifetime, 2*lifetime, ...
                    n_repl = (y // batt_life_y) if (batt_life_y and batt_life_y > 0) else 0
                    cum = (y * battery_avoided_after_kg) - ((1 + n_repl) * batt_embodied_kg_after)
                    cum_curve_after.append(float(cum))
            
                # payback = first year where cumulative becomes >= 0
                for y, cum in zip(years_curve_after, cum_curve_after):
                    if cum >= 0:
                        payback_years_after = float(y)
                        break
            
            totals_after_scope2["battery_payback_years_after"] = payback_years_after
            totals_after_scope2["battery_payback_curve_years_after"] = years_curve_after
            totals_after_scope2["battery_payback_curve_cum_kgCO2e_after"] = cum_curve_after
            

            
            # --- Proposed battery embodied (annualized) + CO2 payback (Option A) ---
            # We reuse the same embodied already computed above if available,
            # but if the building has no measured battery, we try to read proposed battery config.
            batt_embodied_install_kg = float(batt_embodied_kg)  # default (measured config case)
            
            if batt_embodied_install_kg <= 0.0:
                # Try to read proposed battery constraints from project (no hardcoding, just lookup)
                try:
                    bat_cfg = None
                    for b0 in (project.get("batiments") or []):
                        # match by name when ids are not stored consistently
                        if str(b0.get("nom")) == str(bat_nom) or str(b0.get("nom")) == str(bp.get("batiment_nom")):
                            bat_cfg = b0
                            break
                    if bat_cfg is None and isinstance(bat_id, int) and (project.get("batiments") or []) and bat_id < len(project["batiments"]):
                        bat_cfg = project["batiments"][bat_id]
            
                    pb = (bat_cfg.get("pv_battery_proposed") or {}) if isinstance(bat_cfg, dict) else {}
                    tc = (pb.get("tech_constraints") or {}) if isinstance(pb, dict) else {}
            
                    cap_kwh = float(tc.get("fixed_capacity_kwh") or tc.get("capacity_total_kwh") or bp.get("capacity_kwh") or 0.0)
            
                    # factor may exist under different keys depending on Excel mapping
                    factor = (
                        tc.get("embodied_factor_kg_per_kWhcap")
                        or tc.get("factor_kgCO2e_per_kWhcap")
                        or tc.get("battery_embodied_factor_kg_per_kWhcap")
                    )
                    factor = float(factor) if factor is not None else 0.0
            
                    batt_embodied_install_kg = cap_kwh * factor
                except Exception:
                    batt_embodied_install_kg = 0.0
            
            # Prefer "after" source-of-truth if available (results["battery_proposed_by_batiment"])
            # batt_embodied_kg_after already read earlier from bp["batt_embodied_kgCO2e"]
            if batt_embodied_kg_after > 0.0:
                batt_embodied_install_kg = float(batt_embodied_kg_after)
            
            # Lifetime (prefer results/bp, fallback to measured/project)
            batt_life = float(bp.get("batt_lifetime_years", 0.0) or 0.0)
            if batt_life <= 0.0:
                batt_life = float(batt_lifetime_years) if batt_lifetime_years else 0.0
            if batt_life <= 0.0:
                # fallback to proposed tech_constraints lifetime if present
                try:
                    bat_cfg = bat_cfg if "bat_cfg" in locals() else None
                    pb = (bat_cfg.get("pv_battery_proposed") or {}) if isinstance(bat_cfg, dict) else {}
                    tc = (pb.get("tech_constraints") or {}) if isinstance(pb, dict) else {}
                    batt_life = float(tc.get("lifetime_years") or 0.0)
                except Exception:
                    batt_life = 0.0
            
            batt_embodied_annualized_after_kg = (batt_embodied_install_kg / batt_life) if (batt_embodied_install_kg > 0 and batt_life > 0) else 0.0
            
            # payback (years): embodied install / avoided per year (proposed)
            avoided_per_year_after_kg = float(battery_avoided_after_kg)
            payback_years_after = None
            if batt_embodied_install_kg > 0 and avoided_per_year_after_kg > 0:
                payback_years_after = float(batt_embodied_install_kg / avoided_per_year_after_kg)
            
            # payback curve with replacement drops at end-of-life (like economics)
            years_curve_after = None
            cum_curve_after = None
            if horizon > 0 and batt_embodied_install_kg > 0 and avoided_per_year_after_kg >= 0:
                years_curve_after = list(range(0, horizon + 1))
                cum_curve_after = []
                life_int = int(round(batt_life)) if batt_life and batt_life > 0 else 0
            
                cum = -batt_embodied_install_kg  # year 0: initial embodied
                cum_curve_after.append(cum)
            
                for y in range(1, horizon + 1):
                    cum += avoided_per_year_after_kg
                    # replacement at end of lifetime (y = life, 2*life, ...)
                    if life_int > 0 and (y % life_int == 0):
                        cum -= batt_embodied_install_kg
                    cum_curve_after.append(cum)
            
            # Components for Phase 2 bars (after)
            # Convention: positive = emissions/costs, negative = benefits/savings
            demand_elec_kWh = float(totals.get("demand_elec_kWh", 0.0) or 0.0)
            grid_baseline_kg = demand_elec_kWh * grid_factor
            
            # PV pieces (same convention as before)
            pv_self_grid_avoided_after_kg = pv_self_after_kWh * grid_factor
            pv_self_pv_cost_after_kg = pv_self_after_kWh * (float(pv_factor) if pv_factor is not None else 0.0)
            pv_export_pv_cost_after_kg = pv_export_after_kWh * (float(pv_factor) if pv_factor is not None else 0.0)
            
            # Net PV benefit on self-consumption = avoided grid - PV operational cost
            pv_net_benefit_selfc_after_kg = float(pv_self_grid_avoided_after_kg or 0.0) - float(pv_self_pv_cost_after_kg or 0.0)
            
            # Baseline grid emissions (stable reference for decomposition, same as before)
            grid_baseline_after_kg = float(demand_elec_kWh or 0.0) * grid_factor
            
            components_after = {
                # Baseline grid emissions (stable reference for decomposition)
                "grid_import": float(grid_baseline_after_kg or 0.0),
            
                # Benefit (negative in stack), same sign convention as before
                "pv_net_benefit_self_consumption": -float(pv_net_benefit_selfc_after_kg or 0.0),
            
                # Export PV cost (positive in stack)
                "pv_export_pv_cost": float(pv_export_pv_cost_after_kg or 0.0),
            
                # Battery embodied (annualized)
                # IMPORTANT: battery_avoided is attribution only -> excluded from system stack/total
                "battery_embodied": float(
                    (batt_embodied_annualized_after_kg if batt_embodied_annualized_after_kg is not None else 0.0) or 0.0
                ),
            }
            
            system_total_after_kgCO2e = float(sum(float(v or 0.0) for v in components_after.values()))



            battery_avoided_kgCO2e_after = float(battery_avoided_after_kg or 0.0)

            # NEW: system total AFTER (annual)

            def _build_payback_curve(upfront_kg: float, annual_benefit_kg: float, horizon_y: int, lifetime_y: float | None = None):
                """
                Returns dict with years, cumulative, payback_year, horizon_benefit.
                Convention: upfront_kg >0 (cost at year 0), annual_benefit_kg >0 (savings per year).
                Replacements: if lifetime_y is provided and < horizon, subtract upfront at y = lifetime, 2*lifetime, ...
                """
                upfront_kg = float(upfront_kg or 0.0)
                annual_benefit_kg = float(annual_benefit_kg or 0.0)
                if horizon_y <= 0 or upfront_kg <= 0 or annual_benefit_kg <= 0:
                    return {"years": None, "cumulative_kgCO2e": None, "payback_year": None, "horizon_benefit_kgCO2e": None}
            
                years = list(range(0, horizon_y + 1))
                cum = []
                life_int = int(round(float(lifetime_y))) if (lifetime_y and float(lifetime_y) > 0) else 0
            
                c = -upfront_kg
                cum.append(c)
                for y in range(1, horizon_y + 1):
                    c += annual_benefit_kg
                    if life_int > 0 and (y % life_int == 0):
                        c -= upfront_kg
                    cum.append(float(c))
            
                payback = None
                for y, v in zip(years, cum):
                    if v >= 0:
                        payback = float(y)
                        break
            
                return {
                    "years": years,
                    "cumulative_kgCO2e": cum,
                    "payback_year": payback,
                    "horizon_benefit_kgCO2e": float(cum[-1]),
                }
            
            payback_curves_after = {}
            
            payback_curves_after["battery"] = {
                "label": "Battery",
                "upfront_kgCO2e": float(batt_embodied_install_kg or 0.0),
                "annual_benefit_kgCO2e": float(battery_avoided_after_kg or 0.0),
                "lifetime_years": float(batt_life or 0.0) if batt_life else None,
                "curve": _build_payback_curve(batt_embodied_install_kg, battery_avoided_after_kg, horizon, batt_life),
            }

            pv_saved_annual_kg = -(
                float(components_after.get("pv_net_benefit_self_consumption", 0.0) or 0.0)
                + float(components_after.get("pv_export_pv_cost", 0.0) or 0.0)
            )
            
            pv_upfront_kg = 0.0
            pv_life_y = None
            try:
                pv_life_y = float(((pp.get("economics") or {}).get("pv_lifetime_years")) or 0.0) or None
            except Exception:
                pv_life_y = None
            
            # NOTE: if you later store PV embodied install kgCO2e in pv_proposed_by_batiment[bat_id]["pv_embodied_kgCO2e"],
            # this will automatically activate payback.
            try:
                pv_upfront_kg = float(pp.get("pv_embodied_kgCO2e", 0.0) or 0.0)
            except Exception:
                pv_upfront_kg = 0.0
            
            payback_curves_after["pv"] = {
                "label": "PV",
                "upfront_kgCO2e": (float(pv_upfront_kg) if pv_upfront_kg > 0 else None),
                "annual_benefit_kgCO2e": float(pv_saved_annual_kg or 0.0),
                "lifetime_years": pv_life_y,
                "curve": _build_payback_curve(pv_upfront_kg, pv_saved_annual_kg, horizon, pv_life_y),
            }

        
            totals_after_scope2 = {
                "grid_import_kWh_after": grid_import_after_kWh,
                "pv_self_kWh_after": pv_self_after_kWh,
                "pv_export_kWh_after": pv_export_after_kWh,
                "net_kgCO2e_after": net_after_kg,
                # PV split after (for traceability / UI must not recompute)
                "pv_self_grid_avoided_kgCO2e_after": float(pv_self_grid_avoided_after_kg or 0.0),
                "pv_self_pv_cost_kgCO2e_after": float(pv_self_pv_cost_after_kg or 0.0),
                "pv_export_pv_cost_kgCO2e_after": float(pv_export_pv_cost_after_kg or 0.0),
                "components_kgCO2e_after": components_after,
                "battery_payback_years_after": payback_years_after,
                "battery_payback_curve_years_after": years_curve_after,
                "battery_payback_curve_cum_kgCO2e_after": cum_curve_after,
                "battery_embodied_annualized_kgCO2e_after": batt_embodied_annual_after,
                "system_total_kgCO2e_after": system_total_after_kgCO2e,
                "battery_avoided_kgCO2e_after": battery_avoided_kgCO2e_after,
                "payback_curves_after": payback_curves_after,
            }

        
        out["by_batiment"][bat_id] = {
            "scope2": {
                "totals": {
                    "grid_import_kWh": grid_import_kWh,
                    "grid_export_kWh": grid_export_kWh,
                    "net_kgCO2e": net_kg,

                    # PV factors + kWh
                    "pv_factor_kg_per_kWh": pv_factor,
                    "pv_self_kWh": pv_self_kWh,
                    "pv_export_kWh": pv_export_kWh,

                    # PV split (gross + PV cost + net benefit)
                    "pv_self_grid_avoided_kgCO2e": pv_self_grid_avoided_kg,
                    "pv_self_pv_cost_kgCO2e": pv_self_pv_cost_kg,
                    "pv_export_pv_cost_kgCO2e": pv_export_pv_cost_kg,
                    "avoided_by_pv_for_consumption_kgCO2e": avoided_by_pv_for_consumption_kg,
                    "diff_total_vs_no_pv_kgCO2e": diff_total_vs_no_pv_kg,

                    # Storage (measured)
                    "avoided_by_storage_kWh": float(avoided_by_storage_kWh),
                    "avoided_by_storage_kgCO2e": float(avoided_by_storage_kg),

                    # Battery embodied (installation)
                    "battery_capacity_kWh": float(batt_capacity_kWh),
                    "battery_embodied_factor_kg_per_kWhcap": float(batt_factor_kg_per_kWhcap),
                    "battery_embodied_kgCO2e": float(batt_embodied_kg),
                    "battery_lifetime_years": float(batt_lifetime_years) if batt_lifetime_years else None,
                    "battery_embodied_annualized_kgCO2e": float(batt_embodied_annualized_kg),


                    # Battery payback
                    "battery_payback_years": payback_years,
                    "battery_payback_curve_years": years_curve,
                    "battery_payback_curve_cum_kgCO2e": cum_curve,

                    # Components for Phase 2 bar plot
                    "components_kgCO2e": components_kg,
                    "system_total_kgCO2e": system_total_kgCO2e,
                    "battery_avoided_kgCO2e": battery_avoided_kgCO2e,
                },
                "totals_after": totals_after_scope2,
            }
        }
        
        # ---- Scope 1 (thermal combustion) — read-only from thermal flow blocks totals ----
        thermal_blocks = [
            b for b in (blocks or [])
            if isinstance(b.get("type"), str) and b.get("type").startswith("thermal_")
        ]
        
        scope1_fuel_kg = 0.0
        scope1_fuel_in_kWh = 0.0
        scope1_heat_out_th_kWh = 0.0
        
        for tb in thermal_blocks:
            tt = tb.get("totals", {}) or {}
        
            v_kg = tt.get("fuel_scope1_kgCO2e", 0.0)
            v_fuel = tt.get("fuel_in_kWh", 0.0)
            v_heat = tt.get("heat_out_th_kWh", 0.0)
        
            try:
                scope1_fuel_kg += float(v_kg or 0.0)
            except Exception:
                pass
            try:
                scope1_fuel_in_kWh += float(v_fuel or 0.0)
            except Exception:
                pass
            try:
                scope1_heat_out_th_kWh += float(v_heat or 0.0)
            except Exception:
                pass
        
        # Always create scope1 entry (even if 0), so Phase 2 can rely on a stable contract
        out["by_batiment"].setdefault(bat_id, {})
        out["by_batiment"][bat_id]["scope1"] = {
            "totals": {
                "fuel_scope1_kgCO2e": float(scope1_fuel_kg),
                "fuel_in_kWh": float(scope1_fuel_in_kWh),
                "heat_out_th_kWh": float(scope1_heat_out_th_kWh),
                "components_kgCO2e": {
                    "fuel_combustion_scope1_kgCO2e": float(scope1_fuel_kg),
                },
            }
        }
        # -------------------------------------------------------------
        # CECB-like classification for DIRECT CO2 (Scope 1) — Tab. 42
        # Unit: kgCO2/(m²·a) normalized by SRE
        # Limits corrected by f_cor = 1 + (9.4 - theta_e_avg) * 0.06
        # NOTE: correction f_cor is applied to LIMITS only (as in doc)
        # -------------------------------------------------------------
        def _compute_theta_e_avg_from_project() -> Optional[float]:
            """
            Best-effort, NO hardcoding:
            Priority:
              1) project["meta"]["climate_data"]["T_ext"] (your JSON already contains it)
              2) project["params"] may contain station name -> core.climate_data.get_station_monthly()
            If missing, return None (caller will default to 9.4 => f_cor=1).
            """
            month_days = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=float)
        
            # --- 1) From project["meta"]["climate_data"]["T_ext"] ---
            try:
                meta = project.get("meta", {}) or {}
                cdat = meta.get("climate_data", {}) or {}
                t_ext = cdat.get("T_ext", None)
        
                if isinstance(t_ext, list) and len(t_ext) >= 12:
                    t = np.array(t_ext[:12], dtype=float)
                    if np.isfinite(t).any():
                        return float(np.nansum(t * month_days) / np.nansum(month_days))
            except Exception:
                pass
        
            # --- 2) From station name (meta or params) ---
            try:
                params = project.get("params", {}) or {}
                meta = project.get("meta", {}) or {}
        
                station = (
                    params.get("climate_station")
                    or params.get("climate_station_name")
                    or params.get("meteo_station")
                    or params.get("station_name")
                    or meta.get("station_meteo")
                )
                if not station:
                    return None
        
                from core.climate_data import get_station_monthly
                dfm = get_station_monthly(str(station))
                if dfm is None or dfm.empty or "T_ext" not in dfm.columns:
                    return None
        
                t = np.array(dfm["T_ext"].values[:12], dtype=float)
                if len(t) < 12 or np.all(np.isnan(t)):
                    return None
        
                return float(np.nansum(t * month_days) / np.nansum(month_days))
            except Exception:
                return None

        
        def _f_cor(theta_e_avg: Optional[float]) -> float:
            # Reference annual mean temperature 9.4°C + slope 0.06 K^-1 (Eq.57)
            th = 9.4 if theta_e_avg is None else float(theta_e_avg)
            return float(1.0 + (9.4 - th) * 0.06)
        
        def _co2_class_from_spec(spec: float, fcor: float) -> str:
            # Tab. 42 thresholds (kgCO2/(m²·a)) with limits multiplied by f_cor
            # A: 0 exactly
            if spec <= 0.0:
                return "A"
            if spec <= 5.0 * fcor:
                return "B"
            if spec <= 10.0 * fcor:
                return "C"
            if spec <= 15.0 * fcor:
                return "D"
            if spec <= 20.0 * fcor:
                return "E"
            if spec <= 25.0 * fcor:
                return "F"
            return "G"
        
        # --- Need SRE for normalization (source of truth: project["categories_ouvrages"]) ---
        sre_m2 = 0.0
        try:
            units = project.get("categories_ouvrages", []) or []
            batiments = project.get("batiments", []) or []
        
            def _bat_id_from_unit_row(row):
                if "batiment_id" in row and row.get("batiment_id") not in (None, ""):
                    try:
                        return str(int(row.get("batiment_id")))
                    except Exception:
                        return str(row.get("batiment_id"))
                bname = row.get("batiment")
                if isinstance(bname, str) and bname.strip():
                    matches = []
                    for i, b in enumerate(batiments):
                        if str(b.get("nom", "")).strip() == bname.strip():
                            matches.append(i)
                    if len(matches) == 1:
                        return str(matches[0])
                    return None
                if len(batiments) == 1:
                    return "0"
                return None
        
            for row in units:
                if not isinstance(row, dict):
                    continue
                bid = _bat_id_from_unit_row(row)
                if bid == str(bat_id):
                    sre_m2 += float(row.get("sre") or 0.0)
        except Exception:
            sre_m2 = 0.0
        
        theta_e_avg = _compute_theta_e_avg_from_project()
        fcor = _f_cor(theta_e_avg)
        
        direct_spec = None
        direct_class = None
        thresholds = None
        note = None
        
        if sre_m2 <= 0.0:
            note = "SRE is 0 (cannot normalize)."
        else:
            direct_spec = float(scope1_fuel_kg) / float(sre_m2)  # kgCO2/(m²·a)
            direct_class = _co2_class_from_spec(direct_spec, fcor)
            thresholds = {
                "A": (0.0, 0.0),
                "B": (0.0, 5.0 * fcor),
                "C": (5.0 * fcor, 10.0 * fcor),
                "D": (10.0 * fcor, 15.0 * fcor),
                "E": (15.0 * fcor, 20.0 * fcor),
                "F": (20.0 * fcor, 25.0 * fcor),
                "G": (25.0 * fcor, None),
            }
            if theta_e_avg is None:
                note = "No climate station provided → f_cor computed with theta_e_avg = 9.4°C (i.e., f_cor=1)."
        import calendar
                
        # --- Compute theta_e_avg from monthly T_ext (CECB) ---
        T_ext = (
            (project.get("meta") or {})
            .get("climate_data", {})
            .get("T_ext")
        )
        
        theta_e_avg_C = None
        if isinstance(T_ext, list) and len(T_ext) == 12:
            try:
                days = [calendar.monthrange(2021, m)[1] for m in range(1, 13)]
                theta_e_avg_C = sum(t * d for t, d in zip(T_ext, days)) / sum(days)
            except Exception:
                theta_e_avg_C = None
        
        # --- Compute f_cor (Eq. 57) ---
        if theta_e_avg_C is not None:
            f_cor = 1.0 + (9.4 - float(theta_e_avg_C)) * 0.06
            note = "f_cor computed from project.meta.climate_data.T_ext (day-weighted annual mean)."
        else:
            f_cor = 1.0
            theta_e_avg_C = 9.4
            note = "Missing or invalid T_ext → fallback θe,avg = 9.4°C (f_cor=1.0)."

        out["by_batiment"][bat_id]["scope1"]["direct_cecb_label"] = {
            "ok": (direct_class is not None),
            "class": direct_class,
            "direct_kgCO2e_per_m2a": direct_spec,
            "sre_m2_used": float(sre_m2),
            "theta_e_avg_C": theta_e_avg_C,
            "f_cor": float(fcor),
            "thresholds_kgCO2_per_m2a": thresholds,
            "note": note,
        }

        
        global_scope1_kg += scope1_fuel_kg
        global_fuel_in_kWh += scope1_fuel_in_kWh
        global_heat_out_th_kWh += scope1_heat_out_th_kWh

        global_import_kWh += grid_import_kWh
        global_export_kWh += grid_export_kWh
        global_net_kg += net_kg


        # ---- timeseries (kept as grid-only Scope2 baseline vs with PV self-consumption) ----
        bat_ts = ts_by_bat.get(bat_id) or ts_by_bat.get(str(bat_id)) or {}
        idx, measured = _get_measured_ts_payload(bat_ts)
        if idx:
            n = len(idx)
            load = _safe_float_list(measured.get("load_kWh"), n)
            pv_to_load = _safe_float_list(measured.get("pv_to_load_kWh"), n)

            if load is not None and pv_to_load is not None:
                grid_import_ts = np.maximum(np.array(load) - np.array(pv_to_load), 0.0)
                net_ts = (grid_import_ts * grid_factor).tolist()
                baseline_no_pv_ts = (np.array(load) * grid_factor).tolist()
                delta_vs_no_pv_ts = (np.array(baseline_no_pv_ts) - np.array(net_ts)).tolist()

                # Optional after (battery proposed)
                after = _get_after_ts_payload_if_any(bat_ts)
                net_after_ts = None
                if isinstance(after, dict):
                    pv_to_load_after = _safe_float_list(after.get("pv_to_load_kWh"), n)
                    batt_to_load = _safe_float_list(after.get("batt_to_load_kWh"), n)

                    if pv_to_load_after is not None:
                        batt = np.array(batt_to_load) if batt_to_load is not None else np.zeros(n)
                        grid_import_after = np.maximum(np.array(load) - np.array(pv_to_load_after) - batt, 0.0)
                        net_after_ts = (grid_import_after * grid_factor).tolist()
                        
                # ---- Annual aggregation for Phase 2 (NO recompute there) ----
                net_after_kg = None
                delta_after_vs_measured_kg = None
                try:
                    if isinstance(net_after_ts, list) and len(net_after_ts) == n:
                        net_after_kg = float(np.nansum(np.array(net_after_ts, dtype=float)))
                        net_measured_kg = float(np.nansum(np.array(net_ts, dtype=float)))
                        delta_after_vs_measured_kg = float(net_after_kg - net_measured_kg)
                except Exception:
                    net_after_kg = None
                    delta_after_vs_measured_kg = None


                out["timeseries_by_batiment"][bat_id] = {
                    "index": idx,
                    "scope2": {
                        "net_kgCO2e": net_ts,
                        "net_kgCO2e_after": net_after_ts,
                        "baseline_no_pv_kgCO2e": baseline_no_pv_ts,
                        "delta_vs_no_pv_kgCO2e": delta_vs_no_pv_ts,
                        "net_after_kgCO2e": net_after_kg,
                        "delta_after_vs_measured_kgCO2e": delta_after_vs_measured_kg,
                    },
                }

    out["global"]["scope2"]["totals"] = {
        "grid_import_kWh": float(global_import_kWh),
        "grid_export_kWh": float(global_export_kWh),
        "net_kgCO2e": float(global_net_kg),
    }
    
    out["global"]["scope1"]["totals"] = {
        "fuel_scope1_kgCO2e": float(global_scope1_kg),
        "fuel_in_kWh": float(global_fuel_in_kWh),
        "heat_out_th_kWh": float(global_heat_out_th_kWh),
    }


    return out



def _collect_raw_tech_factors(project: Dict[str, Any]) -> Dict[str, Any]:
    """
    Collect raw CO2-related parameters from project config (Phase 1),
    without interpreting them into the Scope 2 net balance.
    This keeps traceability + extensibility for future embodied/LCA work.
    """
    out: Dict[str, Any] = {"electricity_producers": [], "electricity_storage": []}

    ouvrages = project.get("ouvrages", []) or []
    for ouv in ouvrages:
        # Producers
        for prod in (ouv.get("producteurs") or []):
            params = (prod.get("parametres") or {})
            # Keep only keys that look CO2-related
            co2_like = {k: v for k, v in params.items() if "co2" in str(k).lower()}
            if co2_like:
                out["electricity_producers"].append(
                    {
                        "ouvrage_id": ouv.get("id"),
                        "ouvrage_nom": ouv.get("nom"),
                        "engine": prod.get("engine"),
                        "techno": prod.get("techno"),
                        "params": co2_like,
                    }
                )

        # Storage
        for stor in (ouv.get("stockages") or []):
            params = (stor.get("parametres") or {})
            co2_like = {k: v for k, v in params.items() if "co2" in str(k).lower()}
            if co2_like:
                out["electricity_storage"].append(
                    {
                        "ouvrage_id": ouv.get("id"),
                        "ouvrage_nom": ouv.get("nom"),
                        "engine": stor.get("engine"),
                        "techno": stor.get("techno"),
                        "params": co2_like,
                    }
                )

    return out

