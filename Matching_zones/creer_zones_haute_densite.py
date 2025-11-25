"""
Créer une version haute densité des zones sélectionnées
Un seul dossier de sortie avec meilleure qualité
"""

import open3d as o3d
import numpy as np
import os
from pathlib import Path


def upsample_zones(input_dir="zones_selectionnees", output_dir="zones_haute_densite", factor=3.0):
    """
    Crée une version haute densité des zones en créant des points intermédiaires
    factor=3.0 = multiplie le nombre de points par 3
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"CRÉATION ZONES HAUTE DENSITÉ - AUGMENTATION DE DENSITÉ")
    print(f"{'='*80}\n")
    print(f"📁 Dossier source: {input_dir}")
    print(f"📁 Dossier destination: {output_dir}")
    print(f"🎯 Facteur d'augmentation: x{factor} (création de points intermédiaires)\n")
    
    # Trouver tous les fichiers PLY
    ply_files = list(Path(input_dir).glob("*.ply"))
    
    if not ply_files:
        print(f"❌ Aucun fichier PLY trouvé dans {input_dir}")
        return
    
    print(f"✓ {len(ply_files)} fichier(s) trouvé(s)\n")
    print(f"{'Fichier':<35} {'Pts Anciens':<15} {'Pts Nouveaux':<15} {'Augmentation':<15}")
    print("-" * 80)
    
    total_before = 0
    total_after = 0
    
    for ply_file in ply_files:
        # Charger
        pcd = o3d.io.read_point_cloud(str(ply_file))
        original_points = len(pcd.points)
        
        # Upsampling : créer des points intermédiaires par interpolation
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else None
        
        # Construire KDTree pour trouver les voisins
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        
        # Créer nouveaux points en trouvant les k plus proches voisins
        new_points_list = list(points)
        k_neighbors = max(2, int(factor))  # Nombre de voisins à utiliser
        
        for point in points:
            # Trouver les k plus proches voisins
            [k, idx, _] = pcd_tree.search_knn_vector_3d(point, k_neighbors)
            neighbor_points = points[idx]
            
            # Créer des points intermédiaires
            for i in range(1, k_neighbors):
                # Interpolation linéaire entre le point et ses voisins
                interpolated = (point + neighbor_points[i]) / 2.0
                new_points_list.append(interpolated)
        
        # Créer nouveau nuage de points
        pcd_upsampled = o3d.geometry.PointCloud()
        pcd_upsampled.points = o3d.utility.Vector3dVector(np.array(new_points_list))
        
        # Copier couleurs si existantes
        if colors is not None:
            # Répéter les couleurs pour les nouveaux points
            new_colors = []
            for i, point in enumerate(points):
                new_colors.append(colors[i])
                for j in range(1, k_neighbors):
                    new_colors.append(colors[i])
            pcd_upsampled.colors = o3d.utility.Vector3dVector(np.array(new_colors))
        
        sampled_points = len(pcd_upsampled.points)
        
        # Enregistrer
        output_path = os.path.join(output_dir, ply_file.name)
        o3d.io.write_point_cloud(output_path, pcd_upsampled)
        
        # Calculs pour affichage
        difference = sampled_points - original_points
        ratio = (difference / original_points) * 100 if original_points > 0 else 0
        
        # Accumuler totaux
        total_before += original_points
        total_after += sampled_points
        
        # Affichage avec symboles visuels
        symbol = "📈" if difference > 0 else "📉"
        print(f"{ply_file.name:<35} {original_points:>14,} {sampled_points:>14,} {symbol} {ratio:>+8.1f}%")
    
    # Affichage du résumé total
    print("-" * 80)
    total_diff = total_after - total_before
    total_ratio = (total_diff / total_before) * 100 if total_before > 0 else 0
    print(f"{'TOTAL':<35} {total_before:>14,} {total_after:>14,} {symbol} {total_ratio:>+8.1f}%")
    
    print("\n" + "="*80)
    print(f"✅ {len(ply_files)} fichier(s) créé(s) dans '{output_dir}'")
    print(f"📊 Total: {total_before:,} points → {total_after:,} points (+{total_ratio:.1f}%)")
    print("="*80)


def visualize_zones(zones_dir="zones_haute_densite"):
    """Visualise les zones créées"""
    
    print(f"\n📊 Chargement des zones pour visualisation...")
    
    ply_files = sorted(Path(zones_dir).glob("*.ply"))
    
    if not ply_files:
        print(f"❌ Aucun fichier trouvé dans {zones_dir}")
        return
    
    geometries = []
    
    for i, ply_file in enumerate(ply_files):
        pcd = o3d.io.read_point_cloud(str(ply_file))
        
        # Translater pour les séparer visuellement
        points = o3d.utility.Vector3dVector(
            np.asarray(pcd.points) + np.array([i*3, 0, 0])
        )
        
        pcd_copy = o3d.geometry.PointCloud()
        pcd_copy.points = points
        
        # Colorier
        colors = [
            [1, 0, 0],      # Rouge
            [0, 1, 0],      # Vert
            [0, 0, 1],      # Bleu
        ]
        pcd_copy.paint_uniform_color(colors[i % len(colors)])
        
        geometries.append(pcd_copy)
        
        print(f"  {ply_file.name:<35} {len(pcd.points):>8} points")
    
    if geometries:
        print("\n📺 Affichage 3D...")
        print("   (Fermez la fenêtre pour continuer)")
        o3d.visualization.draw_geometries(
            geometries,
            window_name="Zones Haute Densité",
            width=1400,
            height=700
        )


if __name__ == "__main__":
    # Créer les zones haute densité
    upsample_zones(
        input_dir="zones_selectionnees",
        output_dir="zones_haute_densite",
        factor=3.0  # x3 plus de points
    )
    
    # Visualiser
    visualize_zones(zones_dir="zones_haute_densite")
    
    print("\n💡 Prochaines étapes:")
    print("   1. python matching_multiple.py (utilise zones_haute_densite)")
    print("   2. Voir les matchings avec plus de détails!")
