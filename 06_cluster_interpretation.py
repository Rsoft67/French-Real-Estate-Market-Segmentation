import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
from sklearn.preprocessing import MinMaxScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_PATH = os.path.join(BASE_DIR, "data", "processed", "dvf_2024_features.parquet")
KMEANS_PATH = os.path.join(BASE_DIR, "data", "processed", "dvf_kmeans_global.parquet")
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "cluster_analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", context="paper") 
PALETTE = "viridis" 

def load_data():
    print(" Chargement des données ")
    df_feat = pd.read_parquet(FEATURES_PATH)
    df_cluster = pd.read_parquet(KMEANS_PATH)
    
    df = df_feat.copy()
    df["cluster"] = df_cluster["cluster_kmeans"].values
    return df

#PROFILS STATISTIQUES (CSV)
def compute_cluster_profiles(df):
    print("Calcul des profils ")
    
    profile = df.groupby("cluster").agg(
        prix_m2_mean=("prix_m2", "mean"),
        prix_m2_median=("prix_m2", "median"),
        surface_mean=("Surface reelle bati", "mean"),
        pieces_mean=("Nombre pieces principales", "mean"),
        ratio_terrain_mean=("ratio_surface_terrain", "mean"),
        part_maisons=("Type local", lambda x: (x == "Maison").mean()),
        prix_median_commune=("prix_median_commune", "mean"),
        n_biens=("cluster", "count")
    ).reset_index()

    output_csv = os.path.join(OUTPUT_DIR, "cluster_profiles.csv")
    profile.to_csv(output_csv, index=False)
    print(f"Profils sauvegardés : {output_csv}")
    return profile

#BOXPLOTS 
def plot_smart_boxplots(df):
    print("Génération des Boxplots")
    
    vars_to_plot = {
        "prix_m2": "Prix au m² (€)",
        "Surface reelle bati": "Surface (m²)",
        "ratio_surface_terrain": "Ratio Terrain/Bâti"
    }
    
    # Tri des clusters par prix médian pour l'ordre d'affichage
    order = df.groupby("cluster")["prix_m2"].median().sort_values().index
    
    for var, label in vars_to_plot.items():
        plt.figure(figsize=(10, 6))
        sns.boxplot(
            data=df, x="cluster", y=var, order=order, 
            palette=PALETTE, showfliers=False, linewidth=1.2
        )
        plt.title(f"Distribution : {label}", fontsize=14)
        plt.xlabel("Cluster", fontsize=11)
        plt.ylabel(label, fontsize=11)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"boxplot_clean_{var}.png"), dpi=300)
        plt.close()

# RADAR CHART
def plot_sexy_radar(profile):
    print("Génération du Radar Chart")
    
    features = ['prix_m2_mean', 'surface_mean', 'pieces_mean', 'ratio_terrain_mean', 'part_maisons']
    labels = ['Prix m²', 'Surface', 'Pièces', 'Terrain', '% Maisons']
    
    # Normalisation
    scaler = MinMaxScaler()
    data_scaled = pd.DataFrame(scaler.fit_transform(profile[features]), columns=labels)
    data_scaled['cluster'] = profile['cluster']
    
    # Angles
    angles = [n / float(len(labels)) * 2 * pi for n in range(len(labels))]
    angles += angles[:1]
    
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    
    plt.xticks(angles[:-1], labels, color='black', size=11)
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75], ["25%", "50%", "75%"], color="grey", size=8)
    plt.ylim(0, 1)
    
    palette = sns.color_palette(PALETTE, n_colors=len(profile))
    
    for i, row in data_scaled.iterrows():
        values = row.drop('cluster').values.flatten().tolist()
        values += values[:1]
        cluster_id = int(row['cluster'])
        ax.plot(angles, values, linewidth=2, label=f"Cluster {cluster_id}", color=palette[i])
        ax.fill(angles, values, color=palette[i], alpha=0.15)
        
    plt.title("Profils Comparés", size=15, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=9)
    plt.savefig(os.path.join(OUTPUT_DIR, "radar_chart.png"), dpi=300, bbox_inches='tight')
    plt.close()

# SCATTER PLOT ALLÉGÉ
def plot_light_scatter(df):
    print("Génération du Scatter Plot Bivarié ")
    
    n_sample = min(5000, len(df))
    df_sample = df.sample(n=n_sample, random_state=42)
    
    plt.figure(figsize=(10, 7))
    
    sns.scatterplot(
        data=df_sample,
        x="Surface reelle bati",
        y="prix_m2",
        hue="cluster",
        palette=PALETTE,
        alpha=0.7,  
        s=30,       
        edgecolor="w", 
        linewidth=0.5
    )
    
    plt.title(f"Segmentation Prix / Surface (Échantillon de {n_sample} biens)", fontsize=14)
    plt.xlabel("Surface (m²)", fontsize=12)
    plt.ylabel("Prix au m² (€)", fontsize=12)
    
    #Zoom pour éviter les extrêmes qui écrasent
    plt.xlim(0, 250)
    plt.ylim(0, 12000)
    
    plt.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    
    plt.savefig(os.path.join(OUTPUT_DIR, "scatter_prix_surface_light.png"), dpi=300)
    plt.close()
    print("Sauvegardé : scatter_prix_surface_light.png")

#ANALYSE GÉOGRAPHIQUE
def process_geo_communes(df):
    print(" Analyse Géographique ")
    
    #Génération CSV Communes
    geo = df.groupby(["Code departement", "Commune", "cluster"]).size().reset_index(name="count")
    dominant = geo.sort_values("count", ascending=False).groupby(["Code departement", "Commune"]).first().reset_index()
    dominant.to_csv(os.path.join(OUTPUT_DIR, "dominant_cluster_by_commune.csv"), index=False)
    
    #Graphique Départements
    top_depts = dominant["Code departement"].value_counts().nlargest(15).index
    df_top = dominant[dominant["Code departement"].isin(top_depts)]
    
    ct = pd.crosstab(df_top["Code departement"], df_top["cluster"], normalize='index')
    
    ct.plot(kind='bar', stacked=True, figsize=(12, 6), colormap=PALETTE, width=0.8)
    plt.title("Répartition des Clusters par Département (Top 15)", fontsize=14)
    plt.xlabel("Département", fontsize=12)
    plt.ylabel("Proportion", fontsize=12)
    plt.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "geo_distrib_departements.png"), dpi=300)
    plt.close()

def main():
    df = load_data()
    profile = compute_cluster_profiles(df)
    plot_smart_boxplots(df)
    plot_sexy_radar(profile)
    plot_light_scatter(df)
    process_geo_communes(df)
    
    print("\nVisualisations générées dans output/cluster_analysis")

if __name__ == "__main__":
    main()