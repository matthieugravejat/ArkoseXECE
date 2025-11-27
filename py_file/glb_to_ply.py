import open3d as o3d

fichier_glb = "modele.glb"      
fichier_ply = "modele_converti.ply"   

# --- 2. Chargement du fichier GLB ---
mesh = o3d.io.read_triangle_mesh(fichier_glb)

if not mesh.has_vertices():
    raise ValueError("Erreur : le fichier GLB ne contient pas de maillage lisible.")

print("GLB chargé :", mesh)

# --- 3. (Optionnel) calcul des normales si absentes ---
if not mesh.has_vertex_normals():
    mesh.compute_vertex_normals()

# --- 4. Export en .PLY ---
o3d.io.write_triangle_mesh(fichier_ply, mesh)
print(f"Conversion terminée : {fichier_ply}")