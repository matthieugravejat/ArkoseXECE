#!/usr/bin/env python3
"""
Script de test pour vérifier la conversion GLB vers PLY avec couleurs
"""

import os
import sys

# Ajouter le chemin du projet
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Importer la fonction de conversion
from frontend.matching_api import convert_glb_to_colored_ply

def test_conversion():
    """Test de conversion d'un fichier GLB en PLY"""
    
    # Chemin vers un fichier GLB de test
    glb_dir = os.path.join(os.path.dirname(__file__), "zip_mur_GLB")
    
    # Trouver un fichier GLB dans le dossier
    glb_files = [f for f in os.listdir(glb_dir) if f.endswith('.glb')]
    
    if not glb_files:
        print("❌ Aucun fichier GLB trouvé dans zip_mur_GLB/")
        return False
    
    # Prendre le premier fichier GLB
    glb_file = glb_files[0]
    glb_path = os.path.join(glb_dir, glb_file)
    
    # Créer un fichier PLY de sortie temporaire
    output_path = os.path.join(glb_dir, "test_conversion.ply")
    
    print(f"🔄 Test de conversion : {glb_file}")
    print(f"   Fichier source : {glb_path}")
    print(f"   Fichier sortie : {output_path}")
    
    try:
        # Convertir
        pcd = convert_glb_to_colored_ply(glb_path, output_path)
        
        # Vérifier le résultat
        if pcd is None:
            print("❌ La conversion a retourné None")
            return False
        
        if not pcd.has_colors():
            print("❌ Le point cloud n'a pas de couleurs")
            return False
        
        num_points = len(pcd.points)
        print(f"✅ Conversion réussie !")
        print(f"   Nombre de points : {num_points}")
        print(f"   Couleurs présentes : Oui")
        print(f"   Fichier PLY créé : {output_path}")
        
        # Nettoyer le fichier de test
        if os.path.exists(output_path):
            os.remove(output_path)
            print(f"   Fichier de test supprimé")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors de la conversion : {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("TEST DE CONVERSION GLB → PLY AVEC COULEURS")
    print("=" * 60)
    print()
    
    success = test_conversion()
    
    print()
    print("=" * 60)
    if success:
        print("✅ TOUS LES TESTS SONT PASSÉS")
    else:
        print("❌ CERTAINS TESTS ONT ÉCHOUÉ")
    print("=" * 60)
    
    sys.exit(0 if success else 1)
