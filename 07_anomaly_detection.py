import pandas as pd
import numpy as np
import os

from sklearn.ensemble import IsolationForest

RANDOM_STATE = 42

# Paramètres Isolation Forest
CONTAMINATION = 0.03 
N_ESTIMATORS = 200

# PATHS
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FEATURES_PATH = os.path.join(BASE_DIR, "data", "processed", "dvf_2024_features.parquet")
KMEANS_PATH = os.path.join(BASE_DIR, "data", "processed", "dvf_kmeans_global.parquet")

OUTPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "dvf_anomalies_isolation_forest.parquet")

# VARIABLES UTILISÉES
ANOMALY_FEATURES = [
    "prix_m2",
    "Surface reelle bati",
    "Nombre pieces principales",
    "ratio_surface_terrain",
    "prix_median_commune"
]

def main():

    print("Chargement des données")
    df_feat = pd.read_parquet(FEATURES_PATH)
    df_cluster = pd.read_parquet(KMEANS_PATH)

    df = df_feat.copy()
    df["cluster"] = df_cluster["cluster_kmeans"].values

    anomaly_scores = []
    anomaly_flags = []

    print("Isolation Forest par cluster")

    for cluster_id in sorted(df["cluster"].unique()):
        print(f"Cluster {cluster_id}")

        df_c = df[df["cluster"] == cluster_id].copy()

        X = df_c[ANOMALY_FEATURES].values

        iso = IsolationForest(
            n_estimators=N_ESTIMATORS,
            contamination=CONTAMINATION,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

        iso.fit(X)

        scores = iso.decision_function(X)
        preds = iso.predict(X)  # -1 = anomalie, 1 = normal

        df_c["anomaly_score"] = scores
        df_c["anomaly_flag"] = preds

        anomaly_scores.append(df_c[["anomaly_score"]])
        anomaly_flags.append(df_c[["anomaly_flag"]])

    # Reconstruction des résultats
    df["anomaly_score"] = pd.concat(anomaly_scores).sort_index()
    df["anomaly_flag"] = pd.concat(anomaly_flags).sort_index()

    # Sauvegarde
    df.to_parquet(OUTPUT_PATH, index=False)

    n_anomalies = (df["anomaly_flag"] == -1).sum()
    print(f"Anomalies détectées : {n_anomalies} ({n_anomalies / len(df):.2%})")
    print(f"Fichier sauvegardé : {OUTPUT_PATH}")

    print("Détection d’anomalies terminée.")


if __name__ == "__main__":
    main()

