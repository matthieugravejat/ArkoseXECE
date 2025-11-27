import trimesh
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import io
import colorsys

# 1. Charger le GLB et extraire texture
mesh = trimesh.load("Prise_sur_mur.glb", force='mesh')
material = mesh.visual.material

tex = material.baseColorTexture

# Polycam → tex est déjà une image PIL
if isinstance(tex, Image.Image):
    img = tex.convert("RGB")
elif hasattr(tex, "data"):
    img = Image.open(io.BytesIO(tex.data)).convert("RGB")
else:
    raise ValueError("Format inattendu pour la texture GLB")

# 1b. Afficher la texture
img.show(title="Texture de la prise")  # affichage de la texture

# 2. Extraire tous les pixels RGB
img_np = np.array(img)
pixels = img_np.reshape(-1, 3)

# 3. Trouver la couleur dominante via KMeans
kmeans = KMeans(n_clusters=3, random_state=0).fit(pixels)
labels, counts = np.unique(kmeans.labels_, return_counts=True)

dominant = kmeans.cluster_centers_[labels[np.argmax(counts)]].astype(int)
print("Couleur dominante brute (RGB) :", dominant)

# 4. Convertir RGB → HSV
r, g, b = dominant / 255.0
h, s, v = colorsys.rgb_to_hsv(r, g, b)
h_deg = h * 360
print(f"Hue : {h_deg:.1f}°, Saturation : {s:.2f}, Value : {v:.2f}")

# 5. Classification couleur via Hue
def classifier_couleur(h, s, v):
    # Noir
    if v < 0.15:
        return "noir"
    
    # Rouge
    if (h >= 345 or h <= 15) and s > 0.4 and v > 0.2:
        return "rouge"

    # Jaune
    if 40 <= h <= 70 and s > 0.4 and v > 0.3:
        return "jaune"

    # Vert
    if 90 <= h <= 160 and s > 0.3 and v > 0.2:
        return "vert"

    # Bleu
    if 200 <= h <= 240 and s > 0.35 and v > 0.2:
        return "bleu"

    # Violet
    if 260 <= h <= 295 and s > 0.35 and v > 0.2:
        return "violet"

    return "inconnu"

final_color = classifier_couleur(h_deg, s, v)
print("→ Couleur détectée :", final_color)