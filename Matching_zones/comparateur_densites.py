"""
Comparateur de densités de points
==================================
Affiche le même modèle avec différentes densités côte à côte
"""

import open3d as o3d
import numpy as np

# ========================
# CONFIGURATION
# ========================
FICHIER = "data_PLY/Mur de prise.ply"

# Différentes densités à comparer
DENSITES = [5000, 20000, 50000, 100000]

# ========================
# CHARGEMENT
# ========================
print("=" * 60)
print("COMPARATEUR DE DENSITÉS")
print("=" * 60)

print(f"\n📂 Chargement de {FICHIER}...")
mesh = o3d.io.read_triangle_mesh(FICHIER)
mesh.compute_vertex_normals()

print(f"   Mesh : {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")

# ========================
# CRÉATION DES VERSIONS
# ========================
print(f"\n🔬 Création de {len(DENSITES)} versions avec différentes densités...")

geometries = []
offset_x = 0
spacing = 0.3  # Espacement entre les modèles

for densite in DENSITES:
    print(f"   → {densite:,} points...", end=" ")
    
    # Échantillonner
    pcd = mesh.sample_points_uniformly(number_of_points=densite)
    
    # Décaler pour affichage côte à côte
    bbox = pcd.get_axis_aligned_bounding_box()
    width = bbox.get_extent()[0]
    
    pcd.translate([offset_x, 0, 0])
    offset_x += width + spacing
    
    # Colorer selon la densité (du rouge au vert)
    ratio = DENSITES.index(densite) / (len(DENSITES) - 1)
    color = [1 - ratio, ratio, 0]
    pcd.paint_uniform_color(color)
    
    geometries.append(pcd)
    print("✓")

# ========================
# VISUALISATION
# ========================
print(f"\n👁️  Visualisation...")
print("   Les modèles sont affichés de gauche à droite :")
for i, densite in enumerate(DENSITES):
    ratio = i / (len(DENSITES) - 1)
    couleur = "🔴 Rouge" if ratio == 0 else "🟢 Vert" if ratio == 1 else "🟡 Orange"
    print(f"   {couleur} : {densite:,} points")

o3d.visualization.draw_geometries(
    geometries,
    window_name=f"Comparaison densités : {DENSITES[0]:,} à {DENSITES[-1]:,} points",
    width=1400,
    height=800
)

print("\n✅ Terminé !")
