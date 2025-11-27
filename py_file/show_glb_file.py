import trimesh
import open3d as o3d
import numpy as np

# -------------------------------
# 1. Charger le fichier GLB
# -------------------------------
fichier_glb = "Prise_sur_mur.glb"
mesh = trimesh.load(fichier_glb, force='mesh')

print(mesh)  # Affiche des infos sur le mesh : nombre de vertices, faces, matériaux

# -------------------------------
# 2. Convertir en Open3D pour visualisation
# -------------------------------
# Crée un Open3D TriangleMesh depuis les vertices et faces de trimesh
o3d_mesh = o3d.geometry.TriangleMesh(
    vertices=o3d.utility.Vector3dVector(mesh.vertices),
    triangles=o3d.utility.Vector3iVector(mesh.faces)
)

# Si le mesh contient des couleurs sur les vertices
if hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None:
    colors = np.array(mesh.visual.vertex_colors[:, :3]) / 255.0  # normalisation
    o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

# -------------------------------
# 3. Affichage avec Open3D
# -------------------------------
o3d.visualization.draw_geometries([o3d_mesh], window_name="GLB Viewer",
                                  width=800, height=600)
