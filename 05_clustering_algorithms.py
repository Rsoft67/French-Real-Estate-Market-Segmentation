import pandas as pd
import numpy as np
import os

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
import hdbscan

RANDOM_STATE = 42
BATCH_SIZE = 10_000
N_PCA_CLUSTER = 3
SIL_SAMPLE_SIZE = 20_000

# K-means
N_CLUSTERS_GLOBAL = 6
N_CLUSTERS_APPART = 5
N_CLUSTERS_MAISON = 5

# HDBSCAN
HDBSCAN_SAMPLE_SIZE = 30_000
MIN_CLUSTER_SIZE = 100
MIN_SAMPLES = 5


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PCA_PATH = os.path.join(BASE_DIR, "data", "processed", "dvf_2024_pca20.parquet")

OUT_KMEANS_GLOBAL = os.path.join(BASE_DIR, "data", "processed", "dvf_kmeans_global.parquet")
OUT_KMEANS_APPART = os.path.join(BASE_DIR, "data", "processed", "dvf_kmeans_appart.parquet")
OUT_KMEANS_MAISON = os.path.join(BASE_DIR, "data", "processed", "dvf_kmeans_maison.parquet")
OUT_HDBSCAN = os.path.join(BASE_DIR, "data", "processed", "dvf_hdbscan_sample.parquet")

def run_kmeans(X, n_clusters, label, name):
    print(f"K-Means ({name})")

    km = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=BATCH_SIZE,
        random_state=RANDOM_STATE,
        n_init="auto"
    )

    labels = km.fit_predict(X)

    sil = silhouette_score(
        X,
        labels,
        sample_size=min(SIL_SAMPLE_SIZE, len(X)),
        random_state=RANDOM_STATE
    )

    print(f"Silhouette ({name}) : {sil:.3f}")
    return labels, sil


# MAIN
def main():

    print("Chargement PCA")
    df = pd.read_parquet(PCA_PATH)

    pca_cols = [c for c in df.columns if c.startswith("PC")][:N_PCA_CLUSTER]

    # K-MEANS GLOBAL
    X_global = df[pca_cols].values
    labels_global, _ = run_kmeans(X_global, N_CLUSTERS_GLOBAL, df.index, "global")

    df_global = df.copy()
    df_global["cluster_kmeans"] = labels_global
    df_global.to_parquet(OUT_KMEANS_GLOBAL, index=False)


    pca_cols_subset = [c for c in df.columns if c.startswith("PC")][:5] # On prend 5 PCs au lieu de 3
    
    # OPTIMISATION APPARTEMENTS 
    print("\nOptimisation K-Means (Appartements) ")
    df_app = df[df["Type local"] == "Appartement"].copy()
    X_app = df_app[pca_cols_subset].values

    best_sil_app = -1
    best_labels_app = None
    best_k_app = 3

    for k in range(3, 7):
        labels, sil = run_kmeans(X_app, k, df_app.index, f"appart k={k}")
        if sil > best_sil_app:
            best_sil_app = sil
            best_labels_app = labels
            best_k_app = k
    
    print(f"RETENU (Appartements) : k={best_k_app} avec Silhouette={best_sil_app:.3f}")
    df_app["cluster_kmeans"] = best_labels_app
    df_app.to_parquet(OUT_KMEANS_APPART, index=False)

    # OPTIMISATION MAISONS 
    print("\nOptimisation K-Means (Maisons)")
    df_mai = df[df["Type local"] == "Maison"].copy()
    X_mai = df_mai[pca_cols_subset].values

    best_sil_mai = -1
    best_labels_mai = None
    best_k_mai = 3

    for k in range(3, 7):
        labels, sil = run_kmeans(X_mai, k, df_mai.index, f"maison k={k}")
        if sil > best_sil_mai:
            best_sil_mai = sil
            best_labels_mai = labels
            best_k_mai = k

    print(f"RETENU (Maisons) : k={best_k_mai} avec Silhouette={best_sil_mai:.3f}")
    df_mai["cluster_kmeans"] = best_labels_mai
    df_mai.to_parquet(OUT_KMEANS_MAISON, index=False)
    
    # HDBSCAN (ÉCHANTILLON CONTRÔLÉ)
    print("HDBSCAN (échantillon contrôlé)")

    HDBSCAN_SAMPLE_SIZE = 30_000

    rng = np.random.RandomState(RANDOM_STATE)
    sample_idx = rng.choice(df.index, HDBSCAN_SAMPLE_SIZE, replace=False)

    X_sample = df.loc[sample_idx, pca_cols].values

    hdb = hdbscan.HDBSCAN(
        min_cluster_size=100,
        min_samples=5,
        metric="euclidean"
    )

    labels_hdb = hdb.fit_predict(X_sample)

    # Nombre de clusters (hors bruit)
    clusters_hdb = np.unique(labels_hdb)
    clusters_hdb = clusters_hdb[clusters_hdb != -1]

    n_clusters_hdb = len(clusters_hdb)
    n_noise = np.sum(labels_hdb == -1)

    print(f"Clusters HDBSCAN détectés : {n_clusters_hdb}")
    print(f"Points considérés comme bruit : {n_noise} ({n_noise / len(labels_hdb):.1%})")

    # Silhouette HDBSCAN (hors bruit)
    mask = labels_hdb != -1

    if mask.sum() > 1 and len(np.unique(labels_hdb[mask])) > 1:
        sil_hdb = silhouette_score(
            X_sample[mask],
            labels_hdb[mask],
            sample_size=min(20_000, mask.sum()),
            random_state=RANDOM_STATE
        )
        print(f"Silhouette score HDBSCAN (hors bruit) : {sil_hdb:.3f}")
    else:
        print("Silhouette HDBSCAN non calculable")

    # Sauvegarde
    df_hdb = df.loc[sample_idx].copy()
    df_hdb["cluster_hdbscan"] = labels_hdb
    df_hdb.to_parquet(OUT_HDBSCAN, index=False)


    print("Clustering terminé.")


if __name__ == "__main__":
    main()
