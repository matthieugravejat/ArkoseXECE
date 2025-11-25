"""
Script pour ouvrir et visualiser les zones enregistrées
"""

import open3d as o3d
import os
from pathlib import Path
from datetime import datetime


def list_zones():
    """Liste toutes les zones enregistrées"""
    zone_dir = "zones_selectionnees"
    
    if not os.path.exists(zone_dir):
        print(f"❌ Aucun dossier '{zone_dir}' trouvé")
        return []
    
    zone_files = sorted(Path(zone_dir).glob("zone_*.ply"), 
                       key=os.path.getmtime, reverse=True)
    
    return zone_files


def display_zone_list():
    """Affiche la liste des zones"""
    zones = list_zones()
    
    if not zones:
        print("❌ Aucune zone enregistrée")
        return None
    
    print("\n" + "="*70)
    print("ZONES ENREGISTRÉES (les plus récentes en premier):")
    print("="*70)
    
    for i, zone_file in enumerate(zones, 1):
        try:
            pcd = o3d.io.read_point_cloud(str(zone_file))
            mod_time = os.path.getmtime(zone_file)
            mod_date = datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M:%S")
            file_size = os.path.getsize(zone_file) / 1024  # KB
            
            print(f"\n{i}. {zone_file.name}")
            print(f"   Date: {mod_date}")
            print(f"   Points: {len(pcd.points):,}")
            print(f"   Taille: {file_size:.2f} KB")
        except Exception as e:
            print(f"{i}. {zone_file.name} (erreur: {e})")
    
    print("\n" + "="*70)
    return zones


def open_zone(zone_path):
    """Ouvre et affiche une zone"""
    try:
        print(f"\n📂 Chargement: {zone_path}")
        pcd = o3d.io.read_point_cloud(str(zone_path))
        print(f"✓ {len(pcd.points):,} points chargés")
        
        # Colorier en rouge
        pcd.paint_uniform_color([1, 0, 0])
        
        print("Affichage... (Fermez la fenêtre pour quitter)")
        o3d.visualization.draw_geometries(
            [pcd],
            window_name=f"Zone - {zone_path.name}",
            width=1200,
            height=800
        )
        
    except Exception as e:
        print(f"❌ Erreur: {e}")


def main():
    zones = display_zone_list()
    
    if not zones:
        return
    
    # Ouvrir la zone la plus récente
    latest_zone = zones[0]
    print(f"\n✨ Ouverture de la zone la plus récente...")
    open_zone(latest_zone)


if __name__ == "__main__":
    main()
