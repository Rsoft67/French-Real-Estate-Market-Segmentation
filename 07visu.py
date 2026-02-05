
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

OUTPUT_FIGURES = Path("output/figures")
OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)

print("Chargement des anomalies pour visualisation...")
df = pd.read_parquet("data/processed/dvf_anomalies_isolation_forest.parquet")

# Création d'un flag lisible
df["is_anomaly"] = df["anomaly_flag"] == -1

#Scatter Prix / Surface avec anomalies
#Échantillon pour lisibilité
df_sample = df.sample(n=min(8000, len(df)), random_state=42)

plt.figure(figsize=(10, 6))

#Points normaux
plt.scatter(
    df_sample.loc[~df_sample["is_anomaly"], "Surface reelle bati"],
    df_sample.loc[~df_sample["is_anomaly"], "prix_m2"],
    s=8,
    alpha=0.3,
    label="Biens normaux"
)

# Anomalies
plt.scatter(
    df_sample.loc[df_sample["is_anomaly"], "Surface reelle bati"],
    df_sample.loc[df_sample["is_anomaly"], "prix_m2"],
    s=15,
    color="red",
    alpha=0.7,
    label="Anomalies"
)

plt.xlabel("Surface réelle bâtie (m²)")
plt.ylabel("Prix au m² (€)")
plt.title("Anomalies détectées – Prix au m² vs Surface")
plt.legend()
plt.tight_layout()

plt.savefig(OUTPUT_FIGURES / "anomalies_scatter_prix_surface.png", dpi=150)
plt.close()

#Proportion d'anomalies par cluster
anomaly_rate = (
    df.groupby("cluster")["is_anomaly"]
    .mean()
    .reset_index()
)

plt.figure(figsize=(8, 5))
sns.barplot(
    data=anomaly_rate,
    x="cluster",
    y="is_anomaly"
)

plt.ylabel("Proportion d'anomalies")
plt.xlabel("Cluster")
plt.title("Proportion de biens atypiques par cluster")
plt.ylim(0, anomaly_rate["is_anomaly"].max() * 1.2)
plt.tight_layout()

plt.savefig(OUTPUT_FIGURES / "anomalies_rate_by_cluster.png", dpi=150)
plt.close()

#Distribution des scores d'anomalie par cluster
plt.figure(figsize=(10, 5))
sns.boxplot(
    data=df.sample(n=min(10000, len(df)), random_state=42),
    x="cluster",
    y="anomaly_score"
)

plt.title("Distribution des scores d'anomalie par cluster")
plt.xlabel("Cluster")
plt.ylabel("Score Isolation Forest")
plt.tight_layout()

plt.savefig(OUTPUT_FIGURES / "anomalies_score_distribution.png", dpi=150)
plt.close()

print("Visualisations des anomalies sauvegardées.")
