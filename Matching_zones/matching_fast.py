"""
Matching 3D rapide utilisant Point Cloud Registration 
Algorithme plus efficace basé sur les features
"""

import open3d as o3d
import numpy as np
import os
from datetime import datetime


class FastPointCloudMatcher:
    def __init__(self, source_path, target_path):
        """
        source_path: Zone sélectionnée
        target_path: Nuage de référence
        """
        print(f"📂 Chargement source: {source_path}")
        self.source = o3d.io.read_point_cloud(source_path)
        
        print(f"📂 Chargement cible: {target_path}")
        self.target = o3d.io.read_point_cloud(target_path)
        
        print(f"\n📊 Source: {len(self.source.points):,} points")
        print(f"📊 Cible: {len(self.target.points):,} points")
        
        self.source_path = source_path
        self.target_path = target_path
    
    def preprocess_clouds(self):
        """Prétraite les deux nuages"""
        print("\n🔧 Prétraitement...")
        
        # Downsampling
        voxel_size = 0.002
        self.source_down = self.source.voxel_down_sample(voxel_size)
        self.target_down = self.target.voxel_down_sample(voxel_size)
        
        print(f"  Source: {len(self.source_down.points):,} points")
        print(f"  Cible: {len(self.target_down.points):,} points")
        
        # Normales
        self.source_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30)
        )
        self.target_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30)
        )
        
        # Centrer
        self.source_down.translate(-self.source_down.get_center())
        self.target_down.translate(-self.target_down.get_center())
    
    def match_with_global_registration(self):
        """Matching avec Global Registration (plus rapide et robuste)"""
        print("\n🎯 Global Registration (FPFH + Fast Global Registration)...")
        
        # Calculer les features FPFH
        print("  - Calcul features FPFH...")
        source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            self.source_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=100)
        )
        target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            self.target_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=100)
        )
        
        # Fast Global Registration
        print("  - Fast Global Registration...")
        result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
            self.source_down,
            self.target_down,
            source_fpfh,
            target_fpfh,
            o3d.pipelines.registration.FastGlobalRegistrationOption(
                maximum_correspondence_distance=0.03
            )
        )
        
        print(f"  ✓ Fitness: {result.fitness:.6f}")
        print(f"  ✓ RMSE: {result.inlier_rmse:.6f}")
        
        return result
    
    def match_with_ransac(self):
        """Matching avec RANSAC"""
        print("\n🎯 Registration RANSAC...")
        
        # Features FPFH
        source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            self.source_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=100)
        )
        target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            self.target_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=100)
        )
        
        # RANSAC
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            self.source_down,
            self.target_down,
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
        
        print(f"  ✓ Fitness: {result.fitness:.6f}")
        print(f"  ✓ RMSE: {result.inlier_rmse:.6f}")
        
        return result
    
    def refine_with_icp(self, initial_transform):
        """Affine avec ICP"""
        print("\n🔧 Affinement ICP...")
        
        result = o3d.pipelines.registration.registration_icp(
            self.source_down,
            self.target_down,
            0.02,
            initial_transform.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
        )
        
        print(f"  ✓ Fitness finale: {result.fitness:.6f}")
        print(f"  ✓ RMSE final: {result.inlier_rmse:.6f}")
        
        return result
    
    def match(self):
        """Lance le matching complet"""
        # Prétraitement
        self.preprocess_clouds()
        
        # Essayer deux approches
        print("\n" + "-"*70)
        print("APPROCHE 1: Global Registration")
        print("-"*70)
        try:
            result_fgr = self.match_with_global_registration()
            result_fgr_refined = self.refine_with_icp(result_fgr)
            score_fgr = result_fgr_refined.inlier_rmse
        except Exception as e:
            print(f"⚠️  Erreur: {e}")
            result_fgr_refined = None
            score_fgr = float('inf')
        
        print("\n" + "-"*70)
        print("APPROCHE 2: RANSAC")
        print("-"*70)
        try:
            result_ransac = self.match_with_ransac()
            result_ransac_refined = self.refine_with_icp(result_ransac)
            score_ransac = result_ransac_refined.inlier_rmse
        except Exception as e:
            print(f"⚠️  Erreur: {e}")
            result_ransac_refined = None
            score_ransac = float('inf')
        
        # Sélectionner le meilleur
        print("\n" + "="*70)
        if score_fgr < score_ransac:
            print(f"✅ Global Registration meilleur (RMSE: {score_fgr:.6f})")
            self.best_result = result_fgr_refined
        else:
            print(f"✅ RANSAC meilleur (RMSE: {score_ransac:.6f})")
            self.best_result = result_ransac_refined
        
        return self.best_result
    
    def visualize(self):
        """Visualise le résultat"""
        print("\n📊 Visualisation...")
        
        # Appliquer la transformation
        source_aligned = self.source_down.transform(self.best_result.transformation)
        
        # Colorier
        self.target_down.paint_uniform_color([0, 0.651, 0.929])  # Bleu
        source_aligned.paint_uniform_color([1, 0.706, 0])        # Orange
        
        print("Bleu = Prise complète | Orange = Zone alignée")
        print("Fermez la fenêtre pour continuer...")
        
        o3d.visualization.draw_geometries(
            [self.target_down, source_aligned],
            window_name="Matching Rapide - Résultat",
            width=1200,
            height=800
        )
    
    def save_result(self):
        """Enregistre le résultat"""
        os.makedirs("resultats_matching", exist_ok=True)
        
        source_aligned = self.source.transform(self.best_result.transformation)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"resultats_matching/matched_fast_{timestamp}.ply"
        
        o3d.io.write_point_cloud(output_path, source_aligned)
        print(f"✓ Résultat sauvegardé: {output_path}")
        
        # Info
        info_path = f"resultats_matching/info_fast_{timestamp}.txt"
        with open(info_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("RÉSULTATS DU MATCHING RAPIDE\n")
            f.write("="*70 + "\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Source: {self.source_path}\n")
            f.write(f"Cible: {self.target_path}\n\n")
            f.write(f"Score final (RMSE): {self.best_result.inlier_rmse:.6f}\n")
            f.write(f"Fitness: {self.best_result.fitness:.6f}\n\n")
            f.write("Matrice de transformation:\n")
            f.write(str(self.best_result.transformation) + "\n\n")
            f.write(f"Fichier de sortie: {output_path}\n")
        
        print(f"✓ Infos sauvegardées: {info_path}")
        
        return output_path


def main():
    source_path = "zones_selectionnees/zone_20251125_115405.ply"
    target_path = "data_PLY/Prise sur mur.ply"
    
    if not os.path.exists(source_path):
        print(f"❌ Fichier source non trouvé: {source_path}")
        return
    if not os.path.exists(target_path):
        print(f"❌ Fichier cible non trouvé: {target_path}")
        return
    
    print("\n" + "="*70)
    print("MATCHING 3D RAPIDE - GLOBAL REGISTRATION + RANSAC")
    print("="*70)
    
    matcher = FastPointCloudMatcher(source_path, target_path)
    
    # Lancer le matching
    matcher.match()
    
    # Visualiser
    matcher.visualize()
    
    # Sauvegarder
    matcher.save_result()
    
    print("\n" + "="*70)
    print("✅ Matching rapide terminé!")
    print("="*70)


if __name__ == "__main__":
    main()
