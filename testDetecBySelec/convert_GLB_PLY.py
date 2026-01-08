### Pour lancer le programme il faut être dans le dossier testDetecBySelec

import trimesh
import numpy as np
from vedo import Mesh, show, Plotter, load
from PIL import Image
import open3d as o3d

def bake_texture_to_vertex_colors(glb_file, ply_file):
    ###################################
    #####  creation du glb color ######
    ###################################
    tm = trimesh.load(glb_file)

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
    #########  Sauvegarde PLY  ########
    ###################################

    # Mesh Open3D et sauvegarde PLY
    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices)
    mesh_o3d.triangles = o3d.utility.Vector3iVector(faces)
    mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(vertex_colors / 255.0)
    mesh_o3d.compute_vertex_normals()
    o3d.io.write_triangle_mesh(ply_file, mesh_o3d)
    print(f"PLY coloré sauvegardé dans : {ply_file}")

    ###################################
    ###########  Affichage  ###########
    ###################################
    #show(mesh_vedo, axes=0)
    # Lecture du fichier (sans vérification)cd 
    pcd = o3d.io.read_point_cloud(ply_path)
    # Visualisation (bloquante jusqu'à la fermeture de la fenêtre)
    o3d.visualization.draw_geometries([pcd],
                                    window_name=f"Visualisation - {ply_path}",
                                    width=800,
                                    height=600)

# ----------------------
glb_path = "mur3\mur.glb"
ply_path = "mur3\mur_color.ply"

bake_texture_to_vertex_colors(glb_path, ply_path)
