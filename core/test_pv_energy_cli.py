# core/test_pv_energy_cli.py
# -*- coding: utf-8 -*-

from core.pv_module import (
    compute_pv_monthly_energy_m2,
    compute_pv_monthly_energy_per_kw,
)

def main():
    print("=== TEST ÉNERGIE PV AVEC CLIMAT SIA ===")

    station = input("Station météo SIA (ex: Sion, Pully, Chur) : ").strip()
    ori = float(input("Orientation (°) — 0=Sud, 90=Ouest, -90=Est : "))
    tilt = float(input("Inclinaison (°) : "))

    p_mod = float(input("Puissance par module [kW] (ex: 0.4) : "))
    a_mod = float(input("Surface par module [m²] (ex: 1.7) : "))
    
    panel_eff = float(input("Rendement module (en %, ex 20) : "))

    try:
        monthly_m2 = compute_pv_monthly_energy_m2(station, ori, tilt)
        monthly_kw = compute_pv_monthly_energy_per_kw(
            station_name=station,
            orientation_deg=ori,
            tilt_deg=tilt,
            p_module_kw=p_mod,
            area_module_m2=a_mod,
            panel_efficiency=panel_eff,
        )
    except Exception as e:
        print("Erreur :", e)
        return

    print("\n=== Énergie solaire utile (kWh/m² / mois) ===")
    print(monthly_m2.to_string(index=True, float_format=lambda x: f"{x:.2f}"))

    print("\n=== Énergie électrique (kWh/kW / mois) ===")
    print(monthly_kw.to_string(index=True, float_format=lambda x: f"{x:.2f}"))

    print("\n=== Totaux annuels ===")
    print(f"Total kWh/m²/an : {monthly_m2.sum():.2f}")
    print(f"Total kWh/kW/an : {monthly_kw.sum():.2f}")

if __name__ == "__main__":
    main()
