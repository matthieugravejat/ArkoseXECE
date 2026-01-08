### Pour lancer le programme il faut être dans le dossier testDetecBySelec

import open3d as o3d
import os

# Parcourt simplement tous les fichiers du dossier 'data' et les charge
data_dir = "mur3\mur_color.ply"

print(f"Chargement de {data_dir}...")

# Lecture du fichier (sans vérification)cd 
pcd = o3d.io.read_point_cloud(data_dir)

# Afficher quelques infos
print(f"Le nuage contient {len(pcd.points)} points.")
print("Affichage en cours !")

# Visualisation (bloquante jusqu'à la fermeture de la fenêtre)
o3d.visualization.draw_geometries([pcd],
                                    window_name=f"Visualisation - {data_dir}",
                                    width=800,
                                    height=600)
    