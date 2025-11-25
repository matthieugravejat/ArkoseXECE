import open3d as o3d
import os
import zipfile
import tempfile

# Dossier source (contenant les fichiers ZIP)
source_dir = "data_glb"  # Dossier contenant les archives ZIP
# Dossier de destination
output_dir = "data_PLY"  # Où les fichiers PLY seront sauvegardés

os.makedirs(output_dir, exist_ok=True)

# Parcourir tous les fichiers ZIP du dossier source
if os.path.exists(source_dir):
    for zip_name in os.listdir(source_dir):
        if zip_name.lower().endswith(".zip"):
            chemin_zip = os.path.join(source_dir, zip_name)
            
            try:
                print(f"Traitement de l'archive {zip_name}...")
                
                # Extraire le ZIP dans un dossier temporaire
                with tempfile.TemporaryDirectory() as temp_dir:
                    with zipfile.ZipFile(chemin_zip, 'r') as zip_ref:
                        zip_ref.extractall(temp_dir)
                    
                    # Parcourir les fichiers extraits
                    for root, dirs, files in os.walk(temp_dir):
                        for file in files:
                            if file.lower().endswith((".obj", ".glb", ".gltf", ".ply", ".stl")):
                                chemin_source = os.path.join(root, file)
                                
                                # Créer le nom du fichier PLY
                                nom_sans_extension = os.path.splitext(file)[0]
                                chemin_ply = os.path.join(output_dir, nom_sans_extension + ".ply")
                                
                                try:
                                    print(f"  Conversion de {file}...")
                                    
                                    # Lire le fichier 3D (Open3D détecte automatiquement le format)
                                    mesh = o3d.io.read_triangle_mesh(chemin_source)
                                    
                                    # Afficher des infos sur le maillage
                                    print(f"    Sommets: {len(mesh.vertices)}, Triangles: {len(mesh.triangles)}")
                                    
                                    # Sauvegarder en PLY
                                    o3d.io.write_triangle_mesh(chemin_ply, mesh)
                                    print(f"    ✓ Sauvegardé en {chemin_ply}\n")
                                    
                                except Exception as e:
                                    print(f"    ✗ Erreur: {e}\n")
                
                print(f"✓ Archive {zip_name} traitée\n")
                
            except zipfile.BadZipFile:
                print(f"✗ {zip_name} n'est pas un archive ZIP valide\n")
            except Exception as e:
                print(f"✗ Erreur lors du traitement de {zip_name}: {e}\n")
else:
    print(f"Le dossier '{source_dir}' n'existe pas!")
    print("Veuillez placer vos archives ZIP dans ce dossier et relancer le script.")
