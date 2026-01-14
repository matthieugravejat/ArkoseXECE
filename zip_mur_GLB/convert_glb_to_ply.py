#!/usr/bin/env python3
"""
Convertit les fichiers .glb en .ply avec conservation des couleurs (textures UV).
Les fichiers sont classés automatiquement dans Database_Prise_Couleurs.

Usage:
  python convert_glb_to_ply.py [--input-dir PATH] [--output-dir PATH] [--overwrite] [--with-colors]
"""
import argparse
import os
import re
import tempfile
import zipfile
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


# Mapping des noms de fichiers GLB vers les dossiers de classification
# On utilise des patterns pour matcher les fichiers avec leurs catégories
PRISE_CATEGORIES = {
    "Expression_Pure_Eggs": "Expression_Pure_Eggs",
    "Pusher_Smoothie_2": "Pusher_smoothie_2",
    "Pusher_Smoothie_4": "Pusher_smoothie_4",
    "SNAP_5_S": "SNAP_5_S",
    "SNAP_10_LXL": "Snap_10_LXL",
    "Snap_10_LXL": "Snap_10_LXL",
    "Snap_2_XXL": "Snap_10_LXL",  # Grouper avec Snap_10_LXL
    "Teknik_BigBullies": "Teknik_BigBullies",
    "Teknik_BigCurledSlopers_1_5": "Teknik_BigCurledSlopers_1_5",
    "Tekink_FatSlopers": "Teknik_FatSlopers",  # Note: typo dans le nom de fichier
    "Teknik_FatSlopers": "Teknik_FatSlopers",
    "Teknik_NoShadowHands_1_5": "Teknik_NoShadowHands_1_5",
}


def ensure_dependencies():
    if trimesh is None:
        raise RuntimeError(
            "Le module 'trimesh' n'est pas disponible. Installez-le: pip install trimesh"
        )


def ensure_open3d():
    if o3d is None:
        raise RuntimeError(
            "Le module 'open3d' n'est pas disponible. Installez-le: pip install open3d"
        )


def get_category_from_filename(filename: str) -> str:
    """
    Détermine la catégorie de prise à partir du nom de fichier.
    
    Args:
        filename: Nom du fichier GLB
    
    Returns:
        Nom du dossier de catégorie ou None si non trouvé
    """
    for pattern, category in PRISE_CATEGORIES.items():
        if pattern.lower() in filename.lower():
            return category
    return None


def glb_to_ply_with_colors(input_path: Path, output_path: Path, overwrite: bool = False):
    """
    Convertit un GLB en PLY en conservant les couleurs des textures UV.
    
    Args:
        input_path: Chemin du fichier GLB
        output_path: Chemin du fichier PLY de sortie
        overwrite: Écraser si existe
    
    Returns:
        True si la conversion a réussi, False sinon
    """
    ensure_dependencies()
    ensure_open3d()
    
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    out_file = output_path.with_suffix('.ply')
    
    # Créer le dossier parent si nécessaire
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    if out_file.exists() and not overwrite:
        print(f"  ⏭️  Skip (exists): {out_file.name}")
        return True
    
    print(f"\n🔄 Converting with colors: {input_path.name}")
    
    try:
        # Charger le GLB
        tm = trimesh.load(str(input_path))
        
        # Fusionner les sous-meshes si c'est une scène
        if isinstance(tm, trimesh.Scene):
            print("  📦 GLB détecté comme Scene → fusion des sous-mesh")
            geometries = [m for m in tm.geometry.values()]
            if not geometries:
                raise RuntimeError(f"Aucune géométrie trouvée dans {input_path}")
            mesh = trimesh.util.concatenate(geometries)
        else:
            mesh = tm
        
        vertices = mesh.vertices
        faces = mesh.faces
        
        print(f"  📊 Vertices: {len(vertices)}")
        print(f"  📊 Faces: {len(faces)}")
        
        # Essayer d'extraire les couleurs des textures UV
        vertex_colors = None
        
        if hasattr(mesh.visual, 'kind') and mesh.visual.kind == "texture":
            try:
                # Récupérer la texture PIL
                base_tex = mesh.visual.material.baseColorTexture
                if base_tex is not None:
                    texture_image = base_tex.convert("RGB")
                    tex_w, tex_h = texture_image.size
                    tex_pixels = np.array(texture_image)
                    
                    # UV → pixels
                    uv = mesh.visual.uv  # Nx2
                    u = (uv[:, 0] * (tex_w - 1)).astype(int)
                    v = ((1 - uv[:, 1]) * (tex_h - 1)).astype(int)
                    
                    # Clamper les valeurs pour éviter les erreurs d'index
                    u = np.clip(u, 0, tex_w - 1)
                    v = np.clip(v, 0, tex_h - 1)
                    
                    vertex_colors = tex_pixels[v, u, :]
                    print(f"  🎨 Couleurs extraites de la texture ({tex_w}x{tex_h})")
            except Exception as tex_err:
                print(f"  ⚠️  Impossible d'extraire les couleurs de texture: {tex_err}")
        
        # Si pas de texture, essayer les vertex colors directement
        if vertex_colors is None and hasattr(mesh.visual, 'vertex_colors'):
            try:
                vc = mesh.visual.vertex_colors
                if vc is not None and len(vc) > 0:
                    vertex_colors = vc[:, :3]  # RGB only
                    print(f"  🎨 Couleurs extraites des vertex colors")
            except Exception:
                pass
        
        # Créer le mesh Open3D et sauvegarder
        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices)
        mesh_o3d.triangles = o3d.utility.Vector3iVector(faces)
        
        if vertex_colors is not None:
            # Normaliser les couleurs entre 0 et 1 si nécessaire
            if vertex_colors.max() > 1.0:
                vertex_colors = vertex_colors.astype(float) / 255.0
            mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
            print(f"  ✅ Couleurs appliquées au mesh")
        else:
            print(f"  ⚠️  Aucune couleur trouvée, export sans couleurs")
        
        mesh_o3d.compute_vertex_normals()
        o3d.io.write_triangle_mesh(str(out_file), mesh_o3d)
        print(f"  ✓ Wrote: {out_file.name}")
        return True
        
    except Exception as e:
        print(f"  ❌ Erreur lors de la conversion de {input_path}: {e}")
        return False


def glb_to_ply(input_path: Path, output_path: Path, overwrite: bool = False):
    """
    Convertit un GLB en PLY (sans couleurs).
    
    Args:
        input_path: Chemin du fichier GLB
        output_path: Chemin du fichier PLY de sortie
        overwrite: Écraser si existe
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
        
        print(f"  Vertices: {len(mesh.vertices)}")
        print(f"  Faces: {len(mesh.faces)}")
        
        # Export as ply
        mesh.export(str(out_file))
        print(f"  ✓ Wrote: {out_file.name}")
        
    except Exception as e:
        print(f"  ❌ Erreur lors de la conversion de {input_path}: {e}")


def process_zip(zip_path: Path, output_dir: Path, overwrite: bool, with_colors: bool):
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
                    
                    if with_colors:
                        glb_to_ply_with_colors(Path(extracted), out_path, overwrite=overwrite)
                    else:
                        glb_to_ply(Path(extracted), out_path, overwrite=overwrite)


def convert_to_database_colors(input_dir: Path, output_base_dir: Path, overwrite: bool):
    """
    Convertit tous les GLB et les classe dans Database_Prise_Couleurs.
    
    Args:
        input_dir: Dossier contenant les fichiers GLB
        output_base_dir: Dossier Database_Prise_Couleurs
        overwrite: Écraser les fichiers existants
    """
    print("=" * 70)
    print("🎨 GLB to PLY with Colors - Database Classification")
    print("=" * 70)
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_base_dir}")
    print("=" * 70)
    
    # Collecter les fichiers GLB
    files = sorted(input_dir.glob('*.glb')) + sorted(input_dir.glob('*.GLB'))
    
    if not files:
        print("❌ Aucun fichier .glb trouvé dans le dossier d'entrée.")
        return
    
    print(f"\n📁 {len(files)} fichier(s) GLB trouvé(s)\n")
    
    # Statistiques
    stats = {
        "success": 0,
        "skipped": 0,
        "failed": 0,
        "no_category": 0
    }
    
    for glb_file in files:
        # Déterminer la catégorie
        category = get_category_from_filename(glb_file.name)
        
        if category is None:
            print(f"\n⚠️  Catégorie non trouvée pour: {glb_file.name}")
            stats["no_category"] += 1
            continue
        
        # Construire le chemin de sortie
        category_dir = output_base_dir / category
        out_path = category_dir / glb_file.with_suffix('.ply').name
        
        # Convertir avec couleurs
        success = glb_to_ply_with_colors(glb_file, out_path, overwrite=overwrite)
        
        if success:
            stats["success"] += 1
        else:
            stats["failed"] += 1
    
    # Résumé
    print("\n" + "=" * 70)
    print("📊 Résumé de la conversion:")
    print(f"  ✅ Réussies: {stats['success']}")
    print(f"  ⏭️  Ignorées: {stats['skipped']}")
    print(f"  ❌ Échouées: {stats['failed']}")
    print(f"  ⚠️  Sans catégorie: {stats['no_category']}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Convert GLB (and .zip containing GLB) to PLY with optional colors"
    )
    parser.add_argument('--input-dir', '-i', default=None, 
                       help='Dossier source (par défaut le dossier du script)')
    parser.add_argument('--output-dir', '-o', default=None, 
                       help='Dossier de sortie (par défaut ../Matching_zones/data_PLY)')
    parser.add_argument('--overwrite', action='store_true', 
                       help='Écraser les fichiers existants')
    parser.add_argument('--with-colors', '-c', action='store_true',
                       help='Conserver les couleurs des textures UV')
    parser.add_argument('--to-database', '-d', action='store_true',
                       help='Convertir et classer dans Database_Prise_Couleurs')
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    input_dir = Path(args.input_dir) if args.input_dir else script_dir
    
    input_dir = input_dir.resolve()
    
    # Mode Database_Prise_Couleurs
    if args.to_database:
        database_dir = script_dir.parent / 'Code_Matching_Complet' / 'Database_Prise_Couleurs'
        convert_to_database_colors(input_dir, database_dir.resolve(), args.overwrite)
        return
    
    # Mode normal
    output_dir = Path(args.output_dir) if args.output_dir else (script_dir.parent / 'Matching_zones' / 'data_PLY')
    output_dir = output_dir.resolve()
    
    print("=" * 70)
    print("GLB to PLY Converter" + (" (with colors)" if args.with_colors else ""))
    print("=" * 70)
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    print("=" * 70)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect glb and zip files
    files = sorted(input_dir.glob('*.glb')) + sorted(input_dir.glob('*.GLB')) + sorted(input_dir.glob('*.zip'))
    
    if not files:
        print("❌ Aucun fichier .glb ou .zip trouvé dans le dossier d'entrée.")
        return
    
    print(f"\n📁 {len(files)} fichier(s) trouvé(s)\n")
    
    for f in files:
        if f.suffix.lower() == '.zip':
            process_zip(f, output_dir, args.overwrite, args.with_colors)
        else:
            out_name = f.name
            out_path = output_dir / Path(out_name).with_suffix('.ply')
            
            if args.with_colors:
                glb_to_ply_with_colors(f, out_path, overwrite=args.overwrite)
            else:
                glb_to_ply(f, out_path, overwrite=args.overwrite)
    
    print("\n" + "=" * 70)
    print("✅ Conversion terminée!")
    print("=" * 70)


if __name__ == '__main__':
    main()