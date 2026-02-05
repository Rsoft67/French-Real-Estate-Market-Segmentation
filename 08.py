import pandas as pd
import os

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# NOM DU FICHIER CORRIGÉ ICI :
ANOMALIES_PATH = os.path.join(BASE_DIR, "data", "processed", "dvf_anomalies_isolation_forest.parquet")
PROFILES_PATH = os.path.join(BASE_DIR, "output", "cluster_analysis", "cluster_profiles.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "output", "top_opportunities.csv")

def main():
    print(" Génération Express des Opportunités ")
    
    # 1. Chargement
    if not os.path.exists(ANOMALIES_PATH):
        print(f"ERREUR : Fichier toujours introuvable : {ANOMALIES_PATH}")
        return

    # Gestion souple du chemin des profils
    if os.path.exists(PROFILES_PATH):
        df_profiles = pd.read_csv(PROFILES_PATH)
    else:
        alt_path = os.path.join(BASE_DIR, "data", "processed", "cluster_profiles.csv")
        if os.path.exists(alt_path):
             df_profiles = pd.read_csv(alt_path)
        else:
             print("Profils introuvables. Impossible de calculer la décote exacte.")
             return

    # 2. Calculs
    df_ano = pd.read_parquet(ANOMALIES_PATH)
    
    # Fusion pour avoir le prix moyen du cluster
    df_merged = df_ano.merge(df_profiles[['cluster', 'prix_m2_mean']], on='cluster', how='left')
    
    # Calcul décote
    df_merged['decote'] = (df_merged['prix_m2_mean'] - df_merged['prix_m2']) / df_merged['prix_m2_mean']
    
    # Filtre : On garde les vraies opportunités (décote > 20%)
    opportunities = df_merged[df_merged['decote'] > 0.20].copy()
    
    # Tri
    opportunities = opportunities.sort_values('decote', ascending=False)
    
    # Sauvegarde
    opportunities.to_csv(OUTPUT_CSV, index=False)
    print(f"Fichier sauvegardé : {OUTPUT_CSV}")

    
    view = opportunities.head(5).copy()
    view['Cluster'] = view['cluster']
    view['Commune'] = view['Commune']
    view['Type'] = view['Type local']
    view['Surface'] = view['Surface reelle bati'].astype(int).astype(str) + " m²"
    view['Prix m²'] = view['prix_m2'].round(0).astype(int).astype(str) + " €"
    view['Moyenne Cluster'] = view['prix_m2_mean'].round(0).astype(int).astype(str) + " €"
    view['Décote'] = (view['decote'] * 100).round(1).astype(str) + " %"
    
    cols = ['Cluster', 'Commune', 'Type', 'Surface', 'Prix m²', 'Moyenne Cluster', 'Décote']
    print(view[cols].to_markdown(index=False))

if __name__ == "__main__":
    main()