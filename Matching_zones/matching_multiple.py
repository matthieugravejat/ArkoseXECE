"""
Matching 3D Multiple
Compare une source (zone sélectionnée) avec plusieurs cibles
Affiche les résultats de tous les matchings
"""

import open3d as o3d
import numpy as np
import os
from datetime import datetime
from pathlib import Path


class MultiTargetMatcher:
    def __init__(self, source_path):
        """
        source_path: Zone sélectionnée (la source)
        """
        print(f"📂 Chargement source: {source_path}")
        self.source = o3d.io.read_point_cloud(source_path)
        self.source_path = source_path
        
        print(f"📊 Source: {len(self.source.points):,} points")
        
        self.results = []  # Stocker les résultats de tous les matchings
    
    def find_target_files(self):
        """Trouver tous les fichiers PLY à comparer"""
        target_files = []
        
        # Obtenir le répertoire du script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Dossier data
        data_dir = os.path.join(os.path.dirname(script_dir), "lecteur_data", "data")
        if os.path.exists(data_dir):
            for file in Path(data_dir).glob("*.ply"):
                target_files.append(str(file))
        
        # Dossier data_PLY (exclure "Mur de prise.ply")
        data_ply_dir = os.path.join(script_dir, "data_PLY")
        if os.path.exists(data_ply_dir):
            for file in Path(data_ply_dir).glob("*.ply"):
                if file.name != "Mur de prise.ply":
                    target_files.append(str(file))
        
        return sorted(target_files)
    
    def preprocess_cloud(self, pcd):
        """Prétraite un nuage"""
        pcd_down = pcd.voxel_down_sample(0.002)
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30)
        )
        pcd_down.translate(-pcd_down.get_center())
        return pcd_down
    
    def match_with_target(self, target_path):
        """Fait le matching avec une cible"""
        try:
            print(f"\n🎯 Matching avec: {os.path.basename(target_path)}")
            
            # Charger la cible
            target = o3d.io.read_point_cloud(target_path)
            print(f"   Points: {len(target.points):,}")
            
            # Prétraiter
            source_down = self.preprocess_cloud(self.source)
            target_down = self.preprocess_cloud(target)
            
            # Matching RANSAC
            source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                source_down,
                o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=100)
            )
            target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                target_down,
                o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=100)
            )
            
            result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                source_down,
                target_down,
                source_fpfh,
                target_fpfh,
                mutual_filter=False,
                max_correspondence_distance=0.05,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                ransac_n=3,
                checkers=[
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(0.05)
                ],
                criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
            )
            
            # Affiner avec ICP
            # Distance seuil pour appairage: 0.01 = 10mm, 0.02 = 20mm, 0.05 = 50mm
            max_correspondence_distance = 0.01  # À modifier si besoin
            result_icp = o3d.pipelines.registration.registration_icp(
                source_down,
                target_down,
                max_correspondence_distance,
                result_ransac.transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
            )
            
            # Stocker le résultat
            result_info = {
                'target_path': target_path,
                'target_name': os.path.basename(target_path),
                'target_points': len(target.points),
                'fitness': result_icp.fitness,
                'rmse': result_icp.inlier_rmse,
                'transformation': result_icp.transformation,
                'source_aligned': source_down.transform(result_icp.transformation),
                'target_down': target_down
            }
            
            self.results.append(result_info)
            
            print(f"   ✓ Fitness: {result_icp.fitness:.6f}")
            print(f"   ✓ RMSE: {result_icp.inlier_rmse:.6f}")
            
            return result_info
            
        except Exception as e:
            print(f"   ❌ Erreur: {e}")
            return None
    
    def match_all_targets(self):
        """Lance le matching avec toutes les cibles"""
        target_files = self.find_target_files()
        
        if not target_files:
            print("❌ Aucun fichier cible trouvé!")
            return
        
        print(f"\n🔍 Cibles trouvées: {len(target_files)}")
        for f in target_files:
            print(f"   - {os.path.basename(f)}")
        
        print("\n" + "="*70)
        print("LANCEMENT DES MATCHINGS")
        print("="*70)
        
        for target_path in target_files:
            self.match_with_target(target_path)
        
        self.print_summary()
    
    def print_summary(self):
        """Affiche un résumé de tous les résultats"""
        print("\n" + "="*70)
        print("RÉSUMÉ DE TOUS LES MATCHINGS")
        print("="*70)
        
        if not self.results:
            print("Aucun matching réussi!")
            return
        
        # Trier par fitness (le meilleur en premier)
        sorted_results = sorted(self.results, key=lambda x: x['fitness'], reverse=True)
        
        print(f"\n{'Classement':<10} {'Cible':<30} {'Fitness':<15} {'RMSE (mm)':<15} {'Points':<10}")
        print("-" * 80)
        
        for i, result in enumerate(sorted_results, 1):
            rmse_mm = result['rmse'] * 1000  # Convertir en mm
            print(f"{i:<10} {result['target_name']:<30} {result['fitness']:<15.6f} {rmse_mm:<15.3f} {result['target_points']:<10}")
        
        # Trouver le meilleur match
        best = sorted_results[0]
        print(f"\n🏆 MEILLEUR MATCH:")
        print(f"   Cible: {best['target_name']}")
        print(f"   Fitness: {best['fitness']:.6f} (Plus proche = 1.0)")
        print(f"   RMSE: {best['rmse']*1000:.3f} mm")
        print(f"   Points: {best['target_points']:,}")
    
    def visualize_all_results(self):
        """Affiche tous les résultats dans une même fenêtre"""
        if not self.results:
            print("Aucun résultat à afficher!")
            return
        
        # Trier par fitness
        sorted_results = sorted(self.results, key=lambda x: x['fitness'], reverse=True)
        
        # Préparer les géométries pour visualisation
        geometries = []
        
        # Ajouter les 3 meilleurs matchings
        for i, result in enumerate(sorted_results[:3]):
            # Translater pour les séparer visuellement
            target_down = result['target_down']
            source_aligned = result['source_aligned']
            
            # Copier et translater
            target_copy = o3d.geometry.PointCloud()
            target_copy.points = o3d.utility.Vector3dVector(np.asarray(target_down.points) + np.array([i*3, 0, 0]))
            if target_down.has_colors():
                target_copy.colors = o3d.utility.Vector3dVector(np.asarray(target_down.colors))
            else:
                target_copy.paint_uniform_color([0, 0.651, 0.929])  # Bleu
            
            source_copy = o3d.geometry.PointCloud()
            source_copy.points = o3d.utility.Vector3dVector(np.asarray(source_aligned.points) + np.array([i*3, 0, 0]))
            if source_aligned.has_colors():
                source_copy.colors = o3d.utility.Vector3dVector(np.asarray(source_aligned.colors))
            else:
                source_copy.paint_uniform_color([1, 0.706, 0])  # Orange
            
            geometries.append(target_copy)
            geometries.append(source_copy)
        
        print("\n📊 Visualisation des 3 meilleurs matchings...")
        print("   Bleu = Cibles | Orange = Zone alignée")
        print("   Fermez la fenêtre pour continuer...")
        
        o3d.visualization.draw_geometries(
            geometries,
            window_name="Matchings Multiples - Top 3",
            width=1400,
            height=600
        )
    
    def save_results(self):
        """Enregistre les résultats"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        resultats_dir = os.path.join(script_dir, "resultats_matching")
        os.makedirs(resultats_dir, exist_ok=True)
        
        if not self.results:
            print("Aucun résultat à sauvegarder!")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Fichier de résumé
        summary_path = os.path.join(resultats_dir, f"summary_multi_{timestamp}.txt")
        with open(summary_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("RÉSULTATS DES MATCHINGS MULTIPLES\n")
            f.write("="*80 + "\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Source: {self.source_path}\n")
            f.write(f"Nombre de cibles testées: {len(self.results)}\n\n")
            
            # Résultats triés
            sorted_results = sorted(self.results, key=lambda x: x['fitness'], reverse=True)
            
            f.write(f"{'Rang':<5} {'Cible':<40} {'Fitness':<15} {'RMSE (mm)':<15} {'Points':<10}\n")
            f.write("-" * 85 + "\n")
            
            for i, result in enumerate(sorted_results, 1):
                rmse_mm = result['rmse'] * 1000
                f.write(f"{i:<5} {result['target_name']:<40} {result['fitness']:<15.6f} {rmse_mm:<15.3f} {result['target_points']:<10}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("DÉTAILS DU MEILLEUR MATCH:\n")
            f.write("="*80 + "\n\n")
            best = sorted_results[0]
            f.write(f"Cible: {best['target_name']}\n")
            f.write(f"Fitness: {best['fitness']:.6f}\n")
            f.write(f"RMSE: {best['rmse']*1000:.3f} mm\n")
            f.write(f"Points de la cible: {best['target_points']:,}\n\n")
            f.write("Matrice de transformation:\n")
            f.write(str(best['transformation']) + "\n")
        
        print(f"✓ Résumé sauvegardé: {summary_path}")
        
        # Sauvegarder les nuages alignés des 3 meilleurs
        for i, result in enumerate(sorted_results[:3], 1):
            aligned_path = os.path.join(resultats_dir, f"aligned_multi_{i}_{timestamp}.ply")
            o3d.io.write_point_cloud(aligned_path, result['source_aligned'])
            print(f"✓ Alignement #{i} sauvegardé: {aligned_path}")


def main():
    # ✨ UTILISER LE DOSSIER HAUTE DENSITÉ
    # Obtenir le répertoire du script pour construire les chemins relatifs
    script_dir = os.path.dirname(os.path.abspath(__file__))
    zones_dir = os.path.join(script_dir, "zones_haute_densite")
    
    # Trouver le premier fichier zone du dossier choisi
    zone_files = list(Path(zones_dir).glob("*.ply"))
    
    if not zone_files:
        print(f"❌ Aucun fichier PLY trouvé dans {zones_dir}")
        print(f"   Crée d'abord le dossier avec: python creer_zones_haute_densite.py")
        return
    
    # Utiliser le plus récent
    source_path = os.path.join(script_dir, "zones_haute_densite/zone_20251125_140005.ply")
    
    if not os.path.exists(source_path):
        print(f"❌ Fichier source non trouvé: {source_path}")
        return
    
    print("\n" + "="*70)
    print("MATCHING 3D MULTIPLE - COMPARAISON AVEC PLUSIEURS CIBLES")
    print("="*70)
    
    matcher = MultiTargetMatcher(source_path)
    
    # Faire les matchings
    matcher.match_all_targets()
    
    # Visualiser les résultats
    matcher.visualize_all_results()
    
    # Sauvegarder
    matcher.save_results()
    
    print("\n" + "="*70)
    print("✅ Matchings multiples terminés!")
    print("="*70)


if __name__ == "__main__":
    main()
    