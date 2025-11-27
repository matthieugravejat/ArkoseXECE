import open3d as o3d
import os

# Parcourt simplement tous les fichiers du dossier 'data' et les charge
data_dir = "../Matching_zones/data_PLY"

for fname in os.listdir(data_dir):
    chemin_fichier = os.path.join(data_dir, fname)
    print(f"Chargement de {chemin_fichier}...")

    # Lecture du fichier (sans vérification)
    pcd = o3d.io.read_point_cloud(chemin_fichier)

    # Afficher quelques infos
    print(f"Le nuage contient {len(pcd.points)} points.")
    print("Affichage en cours... (Fermez la fenêtre pour passer au fichier suivant)")

    # Visualisation (bloquante jusqu'à la fermeture de la fenêtre)
    o3d.visualization.draw_geometries([pcd],
                                      window_name=f"Visualisation - {fname}",
                                      width=800,
                                      height=600)
    