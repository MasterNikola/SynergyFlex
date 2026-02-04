import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

with open("tariffs.json", "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)

# 1) Conversion robuste en datetime
df["start_timestamp"] = pd.to_datetime(
    df["start_timestamp"], errors="coerce", utc=True  # utc=True gère bien les +01:00
)

# Optionnel: enlever les lignes non converties
df = df.dropna(subset=["start_timestamp"])

# 2) Trier par temps
df = df.sort_values("start_timestamp")

# 3) Colonnes dérivées (maintenant .dt fonctionne)
df["date"] = df["start_timestamp"].dt.date
df["time"] = df["start_timestamp"].dt.time
df["month"] = df["start_timestamp"].dt.month

# 6) Saisons perso
def saison(m):
    if 1 <= m <= 3:
        return "Jan-Mar"
    elif 4 <= m <= 6:
        return "Apr-Jun"
    elif 7 <= m <= 9:
        return "Jul-Sep"
    else:
        return "Oct-Dec"

df["saison"] = df["month"].apply(saison)

# 7) Journée moyenne par saison (moyenne 15 minutes par 15 minutes)
group = (
    df.groupby(["saison", "time"])[["vario_plus", "vario_grid", "dt_plus"]]
      .mean()
      .reset_index()
)

# 8) Heure sur l’axe x
group["time_dt"] = pd.to_datetime(group["time"].astype(str))

plt.figure(figsize=(12, 6))
for season in ["Jan-Mar", "Apr-Jun", "Jul-Sep", "Oct-Dec"]:
    g = group[group["saison"] == season].sort_values("time_dt")
    plt.plot(g["time_dt"], g["vario_plus"], label=season)

plt.xlabel("Heure de la journée")
plt.ylabel("electricity tariffs Ct./kWh")
plt.title("Journée moyenne (vario_plus) par saison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 5) Tri simple par vario_plus décroissant
vario_sorted = df["vario_plus"].sort_values(ascending=False).reset_index(drop=True)

plt.figure(figsize=(10, 5))
plt.plot(vario_sorted, label="vario_plus trié")
plt.xlabel("Index trié")
plt.ylabel(df["unit"].iloc[0])
plt.title("vario_plus du plus grand au plus petit")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# 6) Custom seasons
def season(m):
    if 1 <= m <= 3:
        return "Jan–Mar"
    elif 4 <= m <= 6:
        return "Apr–Jun"
    elif 7 <= m <= 9:
        return "Jul–Sep"
    else:
        return "Oct–Dec"

df["season"] = df["month"].apply(season)

# 7) Average day per season (15-minute resolution)
group = (
    df.groupby(["season", "time"])[["vario_plus", "vario_grid", "dt_plus"]]
      .mean()
      .reset_index()
)

# 8) Time on x-axis
group["time_dt"] = pd.to_datetime(group["time"].astype(str))

plt.figure(figsize=(12, 6))
for season in ["Jan–Mar", "Apr–Jun", "Jul–Sep", "Oct–Dec"]:
    g = group[group["season"] == season].sort_values("time_dt")
    plt.plot(g["time_dt"], g["vario_plus"], label=season)

plt.xlabel("Time of day")
plt.ylabel("Electricity tariffs (Ct./kWh)")
plt.title("Average daily profile of vario_plus by season")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

