import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# PARAMÈTRES À RÉGLER
# ==========================================
FILENAME = "lecteur_data/data/mur_de_prise_3.ply"

# RANSAC (Mur)
RANSAC_THRESH = 0.015      # Distance (m) pour être considéré comme "mur" (2cm)

# DBSCAN (Clustering)
DBSCAN_EPS = 0.015        # Distance (m) max entre deux points pour les lier (2.5cm)
DBSCAN_MIN_POINTS = 10    # Minimum de points pour former un cluster valide

# FILTRAGE PAR TAILLE (Le cœur de ta demande)
MIN_POINTS_IN_HOLD = 100   # En dessous, c'est du bruit (poussière)
MAX_POINTS_IN_HOLD = 2500 # Au dessus, c'est un "Gros Bloc" (sol, plafond, mur adjacent)
# ==========================================

def main():
    # 1. CHARGEMENT ET PRÉPARATION
    print(f"Chargement de {FILENAME}...")
    # On lit le Mesh et on le convertit en Nuage de Points (si nécessaire)
    mesh = o3d.io.read_triangle_mesh(FILENAME)
    if len(mesh.vertices) == 0:
        # Si c'est déjà un nuage de points pur
        pcd = o3d.io.read_point_cloud(FILENAME)
    else:
        # Conversion propre Mesh -> Point Cloud
        pcd = mesh.sample_points_poisson_disk(number_of_points=100000)
    
    print(f"Nuage d'origine : {len(pcd.points)} points")

    # 2. RANSAC : ENLEVER LE MUR PRINCIPAL
    print("Exécution de RANSAC pour isoler le mur...")
    plane_model, inliers = pcd.segment_plane(distance_threshold=RANSAC_THRESH,
                                             ransac_n=3,
                                             num_iterations=1000)
    
    # Séparation
    wall_cloud = pcd.select_by_index(inliers)             # Le Mur
    wall_cloud.paint_uniform_color([0.6, 0.6, 0.6])
    potential_holds = pcd.select_by_index(inliers, invert=True) # Le reste (Prises + Gros Blocs + Bruit)
    
    #########################################################################################
    
    print("Nettoyage du bruit résiduel...")
    # nb_neighbors : combien de voisins on regarde
    # std_ratio : agressivité du filtre (plus c'est bas, plus ça nettoie)
    potential_holds, _ = potential_holds.remove_statistical_outlier(nb_neighbors=20,
                                                                    std_ratio=1)
    
    ##########################################################################################
    
    print(f"Points restants après suppression du mur : {len(potential_holds.points)}")

    # 3. DBSCAN : SÉPARER LES OBJETS RESTANTS
    print("Clustering DBSCAN en cours (cela peut prendre quelques secondes)...")
    # Retourne une liste d'étiquettes (labels). -1 = bruit non classé.
    labels = np.array(potential_holds.cluster_dbscan(eps=DBSCAN_EPS, 
                                                     min_points=DBSCAN_MIN_POINTS, 
                                                     print_progress=True))

    max_label = labels.max()
    print(f"Nombre d'objets détectés (avant filtrage) : {max_label + 1}")

    # ... (Le début du code reste identique jusqu'à la boucle for) ...

    # 4. EXTRACTION ET SAUVEGARDE
    print("Extraction des prises individuelles...")
    
    # Création d'un dossier pour sauvegarder (si tu veux)
    import os
    if not os.path.exists("mes_prises_extraites"):
        os.makedirs("mes_prises_extraites")

    visualisation_list = [wall_cloud] # On commence la liste d'affichage avec le mur
    
    count_holds = 0

    for i in range(max_label + 1):
        # Récupérer les indices du cluster i
        cluster_indices = np.where(labels == i)[0]
        nb_points = len(cluster_indices)

        # LOGIQUE DE TRI
        if MIN_POINTS_IN_HOLD < nb_points < MAX_POINTS_IN_HOLD:
            # === C'EST UNE PRISE VALIDE ===
            count_holds += 1
            
            # 1. On extrait les points pour créer un petit nuage indépendant
            single_hold_pcd = potential_holds.select_by_index(cluster_indices)
            
            ################################################################
            
            obb = single_hold_pcd.get_oriented_bounding_box()
            extent = obb.extent # Renvoie [largeur, hauteur, profondeur]
        
            # On cherche l'épaisseur minimale de l'objet
            thickness = min(extent) 
        
            # SEUIL D'ÉPAISSEUR (ex: 5mm)
            # Si l'objet fait moins de 5mm d'épaisseur, c'est probablement une tache de mur
            if thickness < 0.055: 
                continue # On passe au cluster suivant, on ignore celui-ci
            
            #################################################################
            
            # 2. On lui donne une couleur aléatoire pour la visibilité
            col = plt.get_cmap("tab20")(count_holds % 20)[:3]
            single_hold_pcd.paint_uniform_color(col)
            
            # 3. (Optionnel) On calcule la Boîte Englobante (Bounding Box)
            # AABB = Axis Aligned Bounding Box (boite droite)
            # OBB = Oriented Bounding Box (boite qui suit l'orientation de la prise)
            #bbox = single_hold_pcd.get_axis_aligned_bounding_box()
            #bbox.color = (1, 0, 0) 
            
            # 4. On ajoute à la liste pour visualiser à la fin
            visualisation_list.append(single_hold_pcd)
            #visualisation_list.append(bbox)
            
            # 5. SAUVEGARDE SUR LE DISQUE
            # On sauvegarde chaque prise individuellement
            filename = f"mes_prises_extraites/prise_{count_holds}.ply"
            o3d.io.write_point_cloud(filename, single_hold_pcd)
            print(f" -> Prise {count_holds} sauvegardée ({nb_points} points)")

    print(f"Terminé ! {count_holds} prises extraites et sauvegardées.")

    # 5. VISUALISATION FINALE
    o3d.visualization.draw_geometries(visualisation_list, 
                                      window_name="Extraction des Prises",
                                      width=800, height=600)
        
        
if __name__ == "__main__":
    main()