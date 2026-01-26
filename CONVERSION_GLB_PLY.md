# Conversion GLB vers PLY avec Couleurs

## Résumé des modifications

Ce document décrit les modifications apportées pour permettre le chargement de fichiers GLB, leur conversion automatique en PLY avec couleurs, et leur affichage dans l'application.

## Modifications apportées

### 1. Backend (`matching_api.py`)

#### Imports ajoutés
- `trimesh` : Pour charger et manipuler les fichiers GLB
- `PIL.Image` : Pour traiter les textures des modèles GLB

#### Nouvelle fonction : `convert_glb_to_colored_ply(glb_path, ply_path)`
Cette fonction convertit un fichier GLB en PLY avec les couleurs extraites de la texture :
- Charge le fichier GLB avec trimesh
- Fusionne les sous-meshes si nécessaire (scène GLB)
- Extrait la texture UV du matériau
- Bake les couleurs de la texture sur les vertices
- Sauvegarde le résultat en format PLY avec couleurs
- Retourne un objet `o3d.geometry.PointCloud`

**Inspiré de** : `ArkoseXECE/zip_mur_GLB/convert_glb_to_pcd_ply.py`

#### Nouvelle route API : `/api/convert_glb` (POST)
- Accepte un fichier GLB en upload
- Convertit le GLB en PLY avec couleurs
- **Détecte le plan du mur** (comme `/api/load_wall`)
- **Crée une session** pour permettre l'isolation des prises
- Retourne les données du point cloud au format JSON :
  - `session_id` : ID de session pour les interactions ultérieures
  - `total_points` : Nombre total de points
  - `wall_points` : Nombre de points sur le plan du mur
  - `non_wall_points` : Nombre de points hors du mur (prises)
  - `points` : Liste des coordonnées 3D (centrées)
  - `colors` : Liste des couleurs RGB (0-255)
  - `center` : Centre du point cloud
  - `message` : Message de confirmation

### 2. Frontend

#### Fichier `app.js`

**Nouvelle fonction : `convertAndLoadGLB(file)`**
- Envoie le fichier GLB à l'API `/api/convert_glb`
- Affiche un message de chargement pendant la conversion
- Récupère les données du point cloud converti
- Affiche le point cloud dans le viewer 3D

**Fonction modifiée : `handleFile(file)`**
- Accepte maintenant les fichiers `.ply` ET `.glb`
- Détecte automatiquement le type de fichier
- Appelle `convertAndLoadGLB()` pour les fichiers GLB
- Appelle `loadWallPLY()` pour les fichiers PLY (comportement existant)

#### Fichier `index.html`

**Modifications de l'interface utilisateur :**
- Texte de la dropzone : "Glissez-déposez votre fichier PLY ou GLB de mur ici"
- Attribut `accept` de l'input file : `.ply,.glb`
- Texte des formats acceptés : "Formats acceptés: .ply et .glb"

## Flux de traitement

### Pour un fichier PLY (comportement existant)
1. L'utilisateur sélectionne un fichier `.ply`
2. Le fichier est envoyé à `/api/load_wall`
3. Le point cloud est affiché directement

### Pour un fichier GLB (nouveau)
1. L'utilisateur sélectionne un fichier `.glb`
2. Le fichier est envoyé à `/api/convert_glb`
3. Le backend convertit le GLB en PLY avec couleurs :
   - Extraction de la géométrie (vertices, faces)
   - Extraction de la texture UV
   - Baking des couleurs sur les vertices
   - Conversion en point cloud
   - **Détection du plan du mur**
   - **Création d'une session**
4. Les données du point cloud et la session sont retournées au frontend
5. Le point cloud coloré est affiché dans le viewer 3D
6. **L'utilisateur peut maintenant cliquer sur les prises pour les isoler** (comme avec un fichier PLY)

## Dépendances requises

Assurez-vous que les bibliothèques suivantes sont installées :

```bash
pip install trimesh pillow open3d numpy
```

## Utilisation

1. Lancez le serveur API :
   ```bash
   cd frontend
   python matching_api.py
   ```

2. Ouvrez l'application dans votre navigateur

3. Sélectionnez un mur dans le plan

4. Glissez-déposez un fichier `.glb` ou `.ply` :
   - **Fichier PLY** : Chargement direct
   - **Fichier GLB** : Conversion automatique en PLY avec couleurs, puis affichage

5. Le point cloud s'affiche avec les couleurs préservées

## Avantages

- **Support multi-format** : Accepte maintenant les fichiers GLB texturés
- **Préservation des couleurs** : Les couleurs de la texture sont bakées sur les vertices
- **Transparence** : L'utilisateur n'a pas besoin de convertir manuellement
- **Compatibilité** : Le reste du pipeline (sélection de prises, matching) fonctionne de la même manière

## Notes techniques

- La conversion GLB → PLY se fait côté serveur pour de meilleures performances
- Les fichiers temporaires sont automatiquement nettoyés après conversion
- Le point cloud est centré automatiquement pour un affichage optimal
- Les couleurs sont normalisées (0-1 pour Open3D, 0-255 pour le frontend)
