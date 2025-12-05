# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 10:41:37 2025

@author: vujic
"""

import pandas as pd
import os

def load_electricity_producers(file_path: str = None) -> dict:
    if file_path is None:
        # Dynamically find the Excel file in ../data/
        current_dir = os.path.dirname(__file__)  # ← = core/loaders/
        file_path = os.path.join(current_dir, "..", "..", "data", "electricity_producers.xlsx")
        file_path = os.path.abspath(file_path)

    xl = pd.read_excel(file_path, sheet_name=None)
    data = {}


    for sheet_name, df in xl.items():
        tech_data = {}
        df = df.dropna(subset=[df.columns[0]])
        for _, row in df.iterrows():
            param = str(row[df.columns[0]]).strip()
            valeur = row[df.columns[1]]
            unite = row[df.columns[2]] if len(row) > 2 else ""
            tech_data[param] = {"valeur": valeur, "unite": unite}
        data[sheet_name] = tech_data
    return data
