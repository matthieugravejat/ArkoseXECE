### Pour lancer le programme il faut être dans le dossier testDetecBySelec

import trimesh
import numpy as np
from vedo import Mesh, show
from PIL import Image
import open3d as o3d
from sklearn.decomposition import PCA

###################################
#####  creation du glb color ######
###################################

# Charger le GLB
tm = trimesh.load("mur3\mur.glb")

# Fusionner les sous-meshes si c'est une scène
if isinstance(tm, trimesh.Scene):
    print("GLB détecté comme Scene → fusion des sous-mesh")
    mesh = trimesh.util.concatenate([m for m in tm.geometry.values()])
else:
    mesh = tm

vertices = mesh.vertices
faces = mesh.faces

# Vérifier qu'il y a une texture
if mesh.visual.kind != "texture":
    raise ValueError("Le GLB n'a pas de texture UV à baker.")

# Récupérer la texture PIL
base_tex = mesh.visual.material.baseColorTexture
if base_tex is None:
    raise ValueError("Le matériau n'a pas de baseColorTexture.")
texture_image = base_tex.convert("RGB")  # c'est déjà un PIL Image

tex_w, tex_h = texture_image.size
tex_pixels = np.array(texture_image)

# UV → pixels
uv = mesh.visual.uv  # Nx2
u = (uv[:,0] * (tex_w - 1)).astype(int)
v = ((1 - uv[:,1]) * (tex_h - 1)).astype(int)

vertex_colors = tex_pixels[v, u, :]

# Mesh Vedo
mesh_vedo = Mesh([vertices, faces])
mesh_vedo.pointcolors = vertex_colors


###################################
#######  orientation du mur #######
###################################

verts = np.array(vertices)
pca = PCA(n_components=3)
pca.fit(verts)

# axes principaux
axis1 = pca.components_[0]  # direction de la plus grande variance (longueur du mur)
axis2 = pca.components_[1]  # deuxième plus grande variance (largeur)
axis3 = pca.components_[2]  # plus petite variance (épaisseur, normal au mur)

# Mesh Vedo
mesh_vedo = Mesh([vertices, faces])
mesh_vedo.pointcolors = vertex_colors

# Faire face à moi : normal -> Z
target_normal = np.array([0, 0, 1])
axis = np.cross(axis3, target_normal)
angle = np.arccos(np.clip(np.dot(axis3, target_normal), -1, 1))
mesh_vedo.rotate(np.degrees(angle), axis)

# Redresser la voie : direction longue -> Y
# Calcul du vecteur actuel correspondant à axis1 après rotation
verts_rot = mesh_vedo.points  # vertices après rotation
center = np.mean(verts_rot, axis=0)
vec_long = axis1  # approximatif
# Ici tu peux calculer l'angle entre vec_long projeté sur X-Y et Y, puis faire rotate_z

# Exemple simple : rotation autour de Z pour aligner avec Y
# Supposons vec_long projeté sur XY
vec_proj = vec_long.copy()
vec_proj[2] = 0
angle_z = -np.arctan2(vec_proj[0], vec_proj[1])
mesh_vedo.rotate(-np.degrees(angle_z), [0, 0, 1])


###################################
###########  Affichage  ###########
###################################
show(mesh_vedo, axes=0)
