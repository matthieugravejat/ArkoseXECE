#!/usr/bin/env python3
"""
Convertit les fichiers .glb (ou contenus dans des .zip) en .ply.
Utilise RANSAC pour détecter et supprimer le fond plat avant l'export.

Usage:
  python convert_glb_to_ply.py [--input-dir PATH] [--output-dir PATH] [--overwrite] [--distance-threshold FLOAT]
"""
import argparse
import os
import tempfile
import zipfile
import shutil
from pathlib import Path
import numpy as np

try:
    import trimesh
except Exception:
    trimesh = None

try:
    import open3d as o3d
except Exception:
    o3d = None


def ensure_dependencies():
    if trimesh is None:
        raise RuntimeError(
            "Le module 'trimesh' n'est pas disponible. Installez-le: pip install trimesh"
        )
    if o3d is None:
        raise RuntimeError(
            "Le module 'open3d' n'est pas disponible. Installez-le: pip install open3d"
        )


def remove_planar_background(mesh, distance_threshold=0.005, ransac_n=3, num_iterations=1000, visualize=False):
    """
    Détecte et supprime le fond plat d'un mesh en utilisant RANSAC.
    
    Args:
        mesh: trimesh.Trimesh
        distance_threshold: Distance maximale d'un point au plan pour être considéré comme inlier (en mètres)
        ransac_n: Nombre de points pour estimer le plan
        num_iterations: Nombre d'itérations RANSAC
        visualize: Afficher la visualisation avant/après
    
    Returns:
        trimesh.Trimesh: Mesh nettoyé sans le fond
    """
    print(f"  Détection du fond plat avec RANSAC...")
    print(f"    Vertices avant: {len(mesh.vertices)}")
    
    # Convertir en nuage de points Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(mesh.vertices)
    
    # Appliquer RANSAC pour détecter le plan principal
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations
    )
    
    [a, b, c, d] = plane_model
    print(f"    Plan détecté: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0")
    print(f"    Inliers (fond): {len(inliers)} points ({len(inliers)/len(mesh.vertices)*100:.1f}%)")
    
    # Créer un masque pour garder les points qui NE sont PAS sur le plan (outliers = la prise)
    outliers_mask = np.ones(len(mesh.vertices), dtype=bool)
    outliers_mask[inliers] = False
    
    # Filtrer les vertices et faces
    vertices_filtered = mesh.vertices[outliers_mask]
    
    # Créer un mapping des anciens indices vers les nouveaux
    old_to_new = np.full(len(mesh.vertices), -1, dtype=int)
    old_to_new[outliers_mask] = np.arange(np.sum(outliers_mask))
    
    # Filtrer les faces: garder seulement celles dont tous les sommets sont conservés
    faces_mask = np.all(outliers_mask[mesh.faces], axis=1)
    faces_filtered = mesh.faces[faces_mask]
    
    # Remapper les indices des faces
    faces_filtered = old_to_new[faces_filtered]
    
    print(f"    Vertices après: {len(vertices_filtered)} ({len(vertices_filtered)/len(mesh.vertices)*100:.1f}% conservés)")
    print(f"    Faces après: {len(faces_filtered)} (sur {len(mesh.faces)} originales)")
    
    # Créer le nouveau mesh
    mesh_cleaned = trimesh.Trimesh(
        vertices=vertices_filtered,
        faces=faces_filtered,
        process=False
    )
    
    # Conserver les couleurs si disponibles
    if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
        colors_filtered = mesh.visual.vertex_colors[outliers_mask]
        mesh_cleaned.visual.vertex_colors = colors_filtered
    
    # Visualisation optionnelle
    if visualize:
        visualize_removal(pcd, inliers, outliers_mask)
    
    return mesh_cleaned


def visualize_removal(pcd, inliers, outliers_mask):
    """Visualise le résultat de la détection RANSAC."""
    print("    Visualisation (Fermer la fenêtre pour continuer)...")
    
    # Nuage de points original avec couleurs
    inlier_cloud = pcd.select_by_index(inliers)
    outlier_cloud = pcd.select_by_index(np.where(outliers_mask)[0])
    
    # Colorer le fond en rouge et la prise en vert
    inlier_cloud.paint_uniform_color([1, 0, 0])  # Rouge = fond à enlever
    outlier_cloud.paint_uniform_color([0, 1, 0])  # Vert = prise à garder
    
    o3d.visualization.draw_geometries(
        [inlier_cloud, outlier_cloud],
        window_name="RANSAC: Rouge=Fond (supprimé), Vert=Prise (conservée)",
        width=1200,
        height=800
    )


def glb_to_ply(input_path: Path, output_path: Path, overwrite: bool = False, 
               remove_background: bool = True, distance_threshold: float = 0.005,
               visualize: bool = False):
    """
    Convertit un GLB en PLY avec option de suppression du fond.
    
    Args:
        input_path: Chemin du fichier GLB
        output_path: Chemin du fichier PLY de sortie
        overwrite: Écraser si existe
        remove_background: Appliquer RANSAC pour enlever le fond
        distance_threshold: Seuil de distance pour RANSAC (en mètres)
        visualize: Afficher la visualisation
    """
    ensure_dependencies()
    
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    out_file = output_path.with_suffix('.ply')
    
    if out_file.exists() and not overwrite:
        print(f"Skip (exists): {out_file}")
        return
    
    print(f"\nConverting: {input_path.name}")
    
    try:
        # Charger le mesh
        loaded = trimesh.load(str(input_path), force='mesh')
        mesh = None
        
        # trimesh.load may return a Trimesh or a Scene
        if isinstance(loaded, trimesh.Trimesh):
            mesh = loaded
        else:
            # Scene: concatenate geometries
            try:
                geometries = list(loaded.geometry.values())
            except Exception:
                # fallback: if loaded is iterable
                geometries = list(loaded)
            if not geometries:
                raise RuntimeError(f"Aucune géométrie trouvée dans {input_path}")
            mesh = trimesh.util.concatenate(geometries)
        
        # Supprimer le fond avec RANSAC
        if remove_background:
            mesh = remove_planar_background(
                mesh, 
                distance_threshold=distance_threshold,
                visualize=visualize
            )
            
            # Vérifier qu'il reste quelque chose
            if len(mesh.vertices) < 10:
                print(f"  ⚠️  Avertissement: Très peu de vertices restants ({len(mesh.vertices)})")
                print(f"  Le seuil de distance ({distance_threshold}) est peut-être trop strict")
        
        # Export as ply
        mesh.export(str(out_file))
        print(f"  ✓ Wrote: {out_file.name}")
        
    except Exception as e:
        print(f"  ❌ Erreur lors de la conversion de {input_path}: {e}")


def process_zip(zip_path: Path, output_dir: Path, overwrite: bool, 
                remove_background: bool, distance_threshold: float, visualize: bool):
    """Traite un fichier ZIP contenant des GLB."""
    print(f"\n📦 Processing zip: {zip_path.name}")
    
    with zipfile.ZipFile(str(zip_path), 'r') as zf:
        with tempfile.TemporaryDirectory() as tmpdir:
            for member in zf.namelist():
                if member.lower().endswith('.glb'):
                    extracted = zf.extract(member, path=tmpdir)
                    fname = Path(member).name
                    out_name = Path(fname).with_suffix('.ply')
                    out_path = output_dir / out_name
                    glb_to_ply(
                        Path(extracted), 
                        out_path, 
                        overwrite=overwrite,
                        remove_background=remove_background,
                        distance_threshold=distance_threshold,
                        visualize=visualize
                    )


def main():
    parser = argparse.ArgumentParser(
        description="Convert GLB (and .zip containing GLB) to PLY with optional background removal"
    )
    parser.add_argument('--input-dir', '-i', default=None, 
                       help='Dossier source (par défaut le dossier du script)')
    parser.add_argument('--output-dir', '-o', default=None, 
                       help='Dossier de sortie (par défaut ../Matching_zones/data_PLY)')
    parser.add_argument('--overwrite', action='store_true', 
                       help='Écraser les fichiers existants')
    parser.add_argument('--no-remove-background', action='store_true',
                       help='Ne pas supprimer le fond (garder le GLB complet)')
    parser.add_argument('--distance-threshold', '-d', type=float, default=0.01,
                       help='Seuil de distance RANSAC en mètres (défaut: 0.01 = 10mm)')
    parser.add_argument('--visualize', '-v', action='store_true',
                       help='Afficher la visualisation de la détection RANSAC pour chaque fichier')
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    input_dir = Path(args.input_dir) if args.input_dir else script_dir
    output_dir = Path(args.output_dir) if args.output_dir else (script_dir.parent / 'Matching_zones' / 'data_PLY')
    
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    
    print("="*70)
    print("GLB to PLY Converter with RANSAC Background Removal")
    print("="*70)
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Remove background: {not args.no_remove_background}")
    if not args.no_remove_background:
        print(f"Distance threshold: {args.distance_threshold}m ({args.distance_threshold*1000}mm)")
    print(f"Visualize: {args.visualize}")
    print("="*70)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect glb and zip files
    files = sorted(input_dir.glob('*.glb')) + sorted(input_dir.glob('*.GLB')) + sorted(input_dir.glob('*.zip'))
    
    if not files:
        print("❌ Aucun fichier .glb ou .zip trouvé dans le dossier d'entrée.")
        return
    
    print(f"\n📁 {len(files)} fichier(s) trouvé(s)\n")
    
    for f in files:
        if f.suffix.lower() == '.zip':
            process_zip(
                f, output_dir, args.overwrite,
                remove_background=not args.no_remove_background,
                distance_threshold=args.distance_threshold,
                visualize=args.visualize
            )
        else:
            out_name = f.name
            out_path = output_dir / Path(out_name).with_suffix('.ply')
            glb_to_ply(
                f, out_path, 
                overwrite=args.overwrite,
                remove_background=not args.no_remove_background,
                distance_threshold=args.distance_threshold,
                visualize=args.visualize
            )
    
    print("\n" + "="*70)
    print("✅ Conversion terminée!")
    print("="*70)


if __name__ == '__main__':
    main()