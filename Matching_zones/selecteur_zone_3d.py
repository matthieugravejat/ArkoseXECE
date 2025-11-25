"""
Script de sélection de zone 3D sur un nuage de points PLY
- Affiche le modèle 3D
- Permet de sélectionner des points avec Shift + Clic
- Enregistre la zone isolée dans un fichier PLY
"""

import open3d as o3d
import numpy as np
import os
from datetime import datetime

class ZoneSelector3D:
    def __init__(self, pcd_path):
        self.pcd = o3d.io.read_point_cloud(pcd_path)
        self.pcd_path = pcd_path
        self.picked_indices = []
        
    def select_zone_interactive(self):
        """Lance l'interface de sélection avec picking"""
        print("\n" + "="*70)
        print("SÉLECTION DE ZONE - INSTRUCTIONS:")
        print("="*70)
        print("• Maintenez SHIFT et cliquez gauche sur les points pour les sélectionner")
        print("• Cliquez multiple fois pour tracer le contour de votre zone")
        print("• Appuyez sur 'Q' ou fermez la fenêtre quand vous avez terminé")
        print("="*70 + "\n")
        
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window(window_name="Sélecteur de Zone 3D", width=1200, height=800)
        vis.add_geometry(self.pcd)
        
        # Lance la visualisation interactive
        vis.run()
        vis.destroy_window()
        
        # Récupère les indices sélectionnés
        self.picked_indices = vis.get_picked_points()
        print(f"\n✓ {len(self.picked_indices)} points sélectionnés")
        
        return self.picked_indices
    
    def extract_zone(self):
        """Extrait la zone basée sur les points sélectionnés"""
        if len(self.picked_indices) == 0:
            print("❌ Aucun point sélectionné!")
            return None
        
        # Récupère les points sélectionnés
        selected_points = np.asarray(self.pcd.points)[self.picked_indices]
        
        # Calcule la boîte englobante
        min_bounds = selected_points.min(axis=0)
        max_bounds = selected_points.max(axis=0)
        
        # Ajoute 5% de marge
        margin = (max_bounds - min_bounds) * 0.05
        min_bounds -= margin
        max_bounds += margin
        
        print(f"\nBoîte englobante: min={min_bounds}, max={max_bounds}")
        
        # Filtre les points dans la boîte
        all_points = np.asarray(self.pcd.points)
        mask = (
            (all_points[:, 0] >= min_bounds[0]) & (all_points[:, 0] <= max_bounds[0]) &
            (all_points[:, 1] >= min_bounds[1]) & (all_points[:, 1] <= max_bounds[1]) &
            (all_points[:, 2] >= min_bounds[2]) & (all_points[:, 2] <= max_bounds[2])
        )
        
        # Crée le nuage extrait
        zone_pcd = o3d.geometry.PointCloud()
        zone_pcd.points = o3d.utility.Vector3dVector(all_points[mask])
        
        # Copie les couleurs si elles existent
        if self.pcd.has_colors():
            all_colors = np.asarray(self.pcd.colors)
            zone_pcd.colors = o3d.utility.Vector3dVector(all_colors[mask])
        else:
            zone_pcd.paint_uniform_color([1, 0, 0])  # Rouge par défaut
        
        print(f"✓ Zone extraite: {len(zone_pcd.points)} points")
        
        return zone_pcd
    
    def save_zone(self, zone_pcd):
        """Enregistre la zone isolée"""
        if zone_pcd is None:
            return None
        
        # Créer le dossier s'il n'existe pas
        os.makedirs("zones_selectionnees", exist_ok=True)
        
        # Générer le nom du fichier
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"zones_selectionnees/zone_{timestamp}.ply"
        
        # Enregistrer
        o3d.io.write_point_cloud(output_path, zone_pcd)
        print(f"✓ Fichier enregistré: {output_path}")
        
        return output_path
    
    def display_zone(self, zone_pcd):
        """Affiche la zone isolée"""
        print("\nAffichage de la zone isolée (Fermez la fenêtre pour quitter)...")
        
        zone_pcd.paint_uniform_color([1, 0, 0])  # Rouge
        
        o3d.visualization.draw_geometries(
            [zone_pcd],
            window_name="Zone Isolée",
            width=1200,
            height=800
        )


def main():
    mur_path = "data_PLY/Mur de prise.ply"
    
    # Vérifier que le fichier existe
    if not os.path.exists(mur_path):
        print(f"❌ Erreur: Fichier non trouvé - {mur_path}")
        return
    
    print(f"\n📁 Chargement: {mur_path}")
    
    # Initialiser le sélecteur
    selector = ZoneSelector3D(mur_path)
    print(f"✓ {len(selector.pcd.points)} points chargés\n")
    
    # Sélectionner la zone
    picked = selector.select_zone_interactive()
    
    if len(picked) < 3:
        print("❌ Sélection insuffisante (besoin d'au moins 3 points)")
        return
    
    # Extraire la zone
    print("\n📦 Extraction de la zone...")
    zone = selector.extract_zone()
    
    if zone is None or len(zone.points) == 0:
        print("❌ Erreur lors de l'extraction")
        return
    
    # Enregistrer
    print("\n💾 Enregistrement...")
    saved_path = selector.save_zone(zone)
    
    # Afficher
    if saved_path:
        selector.display_zone(zone)
        print(f"\n✅ Zone sauvegardée avec succès: {saved_path}")


if __name__ == "__main__":
    main()
