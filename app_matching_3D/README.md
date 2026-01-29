# 🧗 HoldGen - Reconnaissance 3D de Prises d'Escalade

Système de matching automatique de prises d'escalade à partir de scans 3D avec base de données MongoDB, utilisant une approche en 3 phases : filtrage couleur, écrémage par eigenvalues, et matching RANSAC/ICP.

---

## ⚠️ IMPORTANT - Configuration Réseau

**⛔ NE PAS se connecter au WiFi public "Welcome Salon"** - Ce réseau bloque l'accès à MongoDB et empêchera le chargement de la base de données des prises.

Utilisez plutôt :
- Le WiFi privé de l'école (si disponible)
- Une connexion filaire (Ethernet)
- Votre point d'accès mobile personnel

---

## 📋 Prérequis

- **Python 3.8+**
- **pip** (gestionnaire de paquets Python)
- **Connexion Internet** (pour MongoDB - voir note ci-dessus)

---

## 🔧 Installation

### 1. Cloner le repository

```bash
git clone https://github.com/matthieugravejat/ArkoseXECE.git
cd ArkoseXECE
```

### 2. Installer les dépendances

```bash
pip install open3d numpy flask flask-cors pymongo gdown trimesh pillow scikit-learn
```

**Liste complète des dépendances :**
- `open3d` - Traitement de nuages de points 3D
- `numpy` - Calculs matriciels
- `flask` - Serveur API backend
- `flask-cors` - Support CORS pour l'API
- `pymongo` - Connexion à la base de données MongoDB
- `gdown` - Téléchargement depuis Google Drive
- `trimesh` - Traitement des maillages 3D (GLB/OBJ)
- `pillow` - Traitement d'images
- `scikit-learn` - Clustering K-means

---

## 🚀 Lancement de l'application

### **Étape 1 : Vérifier la connexion réseau**

✅ Assurez-vous d'être connecté à un réseau **autorisé** (pas Welcome Salon)

### **Étape 2 : Lancer le serveur backend (API Flask)**

Ouvrez un terminal dans le dossier `app_matching_3D` :

```bash
cd app_matching_3D
python matching_api_mongo.py
```

Le serveur démarrera sur `http://localhost:5001` et commencera à charger la base de données MongoDB en arrière-plan.

Vous verrez un message comme :
```
🚀 HoldGen API Server
📂 Base directory: ...
💾 Cache directory: ...
🔄 Lancement du chargement MongoDB en arrière-plan...
✅ Thread de chargement démarré!
📊 Progression disponible sur: http://localhost:5001/api/loading_status
```

### **Étape 3 : Lancer le serveur HTTP (Frontend) - Dans un NOUVEAU terminal**

```bash
cd app_matching_3D
python3 -m http.server 8080
```

Le serveur HTTP démarrera sur `http://localhost:8080`

### **Étape 4 : Accéder à l'application**

Ouvrez votre navigateur et accédez à :
```
http://localhost:8080
```

Vous verrez un écran de chargement avec une barre de progression de la base de données. **Attendez que le chargement soit terminé** (cela peut prendre quelques minutes lors du premier lancement).

---

## 📝 Résumé des ports

| Service | Port | URL |
|---------|------|-----|
| Frontend (Web UI) | 8080 | http://localhost:8080 |
| API Backend | 5001 | http://localhost:5001 |
| Statut MongoDB | 5001 | http://localhost:5001/api/loading_status |

---

## 📁 Structure du projet

```
ArkoseXECE/
├── app_matching_3D/                       # Application web
│   ├── index.html                         # Interface utilisateur
│   ├── style.css                          # Styles CSS
│   ├── app.js                             # Logique JavaScript (Three.js)
│   ├── matching_api_mongo.py              # API Flask avec MongoDB
│   ├── matching_api.py                    # API Flask (version locale)
│   └── cache_prises/                      # Cache des fichiers PLY téléchargés
│
├── Code_Matching_Complet/
│   └── Database_Prise_Couleurs/           # Base de données locale PLY (couleurs)
│       ├── Pusher_smoothie_2/
│       ├── SNAP_5_S/
│       ├── Teknik_NoShadowHands_1_5/
│       └── ... (9 familles de prises)
│
├── zip_mur_GLB/                           # Modèles 3D GLB pour visualisation
│   ├── SNAP_5_S_num1.glb
│   ├── Teknik_NoShadowHands_1_5_num1.glb
│   └── ... (modèles des prises)
│
├── Matching_zones/
│   ├── matching_fast.py                   # Algorithme de matching optimisé
│   ├── selecteur_zone_3d.py               # Sélection interactive de zones
│   └── resultats_matching/                # Résultats des matchings
│
├── convertir_zip-glb_en_ply/              # Outils de conversion GLB→PLY
├── lecteur_data/                          # Lecture et affichage de données PLY
├── test_glb_conversion.py                 # Tests de conversion
└── README.md                              # Ce fichier
```

---

## 🎯 Fonctionnalités

### **1. Mode Mur** 🖱️
Chargez un fichier PLY de mur et sélectionnez les prises interactivement :
- Clic sur les prises pour les isoler
- Mode automatique DBSCAN pour détection rapide
- Visualisation 3D temps réel

### **2. Matching en 3 phases** ⚡

Le système utilise un pipeline optimisé :

- **Phase 0 : Filtrage par couleur (HSV)**
  - Analyse des couleurs dominantes
  - Sépare couleurs saturées vs désaturées
  - Élimine les prises de couleurs incompatibles
  
- **Phase 1 : Écrémage par eigenvalues**
  - Compare la "silhouette" 3D des objets
  - Analyse PCA de la forme géométrique
  - Filtre rapide basé sur les dimensions principales
  
- **Phase 2 : RANSAC + ICP**
  - Matching précis sur les candidats restants
  - RANSAC pour l'alignment initial
  - ICP (Iterative Closest Point) pour affinage
  - Score final : eigenvalues (30%) + ICP fitness (70%)

### **3. Visualisation 3D** 🎨
- Affichage temps réel des nuages de points avec Three.js
- Modèles GLB haute qualité pour les résultats
- Isolation des prises avec couleurs distinctes
- Rotation/zoom interactif

---

## 🧪 Utilisation avancée

### Paramètres de matching

Éditez `matching_api_mongo.py` (classe `HoldMatcherAPI`, méthode `__init__`) :

```python
matcher = HoldMatcherAPI(
    source_pcd,
    eigen_threshold=0.12,      # Seuil eigenvalues (0.05 = strict, 0.2 = permissif)
    hue_threshold=40.0,        # Seuil teinte en degrés (30° = strict, 90° = permissif)
    value_threshold=40.0,      # Seuil luminosité pour couleurs désaturées
    use_color_filter=True      # Activer/désactiver le filtre couleur
)
```

### Tester localement sans MongoDB

Si MongoDB n'est pas accessible, utilisez `matching_api.py` qui charge les prises depuis le dossier local `Code_Matching_Complet/` :

```bash
python matching_api.py
```

---

## 📊 Base de données

### MongoDB (matching_api_mongo.py)
Contient **~60 prises** dans 9 familles, stockées dans les collections:
- `Color_Red`, `Color_Blue`, `Color_Green`, `Color_Yellow`, `Color_Purple`, `Color_Black`

Les fichiers PLY sont téléchargés depuis Google Drive au premier lancement et mis en cache localement.

**⚠️ Note sur le téléchargement Google Drive :**

Google Drive impose des limitations pour les téléchargements :
- Chaque fichier PLY (~10-20 MB) est stocké sur un lien Google Drive individuel
- **Le téléchargement ne peut pas se faire en parallèle** avec la même adresse IP (risque de 403 Forbidden)
- **Les téléchargements sont séquentiels** : un fichier à la fois
- Le système utilise `gdown` avec des **délais** entre chaque requête pour éviter les blocages
- **Premier lancement peut prendre 5-15 minutes** selon votre connexion (1 minute par prise environ)
- Les fichiers sont ensuite mis en cache local, les lancements suivants sont instantanés

C'est pourquoi l'écran de chargement est patient et pourquoi nous avons un système de barre de progression.

### Base locale (matching_api.py)
Stockée dans `Code_Matching_Complet/Database_Prise_Couleurs/` avec les familles :
- Pusher Smoothie (2 variantes)
- SNAP (5S, 10 LXL, 2 XXL)
- Teknik (BigBullies, BigCurledSlopers, FatSlopers, NoShadowHands)
- Expression (Pure Eggs)

Chaque prise est disponible en :
- **Format PLY** (nuage de points avec couleurs) → pour le matching
- **Format GLB** (maillage 3D) → pour la visualisation web

---

## 🐛 Dépannage

### L'API ne démarre pas
```bash
# Vérifier que Flask est installé
pip install flask flask-cors

# Vérifier que le port 5000 est libre
lsof -ti:5000 | xargs kill -9
```

### Le frontend ne charge pas
```bash
# Vérifier que le port 8080 est libre
lsof -ti:8080 | xargs kill -9
```

### Erreur "Too few correspondences"
C'est un avertissement normal d'Open3D pendant la phase RANSAC. Tant que le matching se termine et retourne des résultats, tout fonctionne correctement.

---

## 👨‍💻 Développement

### Architecture technique

**Backend (Python + Flask)**
- Open3D pour le traitement 3D
- NumPy pour les calculs matriciels
- Scikit-learn pour le clustering couleur

**Frontend (JavaScript + Three.js)**
- Three.js pour la visualisation 3D
- GLTFLoader pour charger les modèles
- API Fetch pour communiquer avec le backend

---

## 📝 Licence

Projet académique - ECE Paris - ING5 2026

---

## 🙏 Crédits

- **Prises d'escalade** : Scan 3D des familles Pusher, SNAP, Teknik, Expression
- **Algorithmes** : RANSAC/ICP (Open3D), PCA pour eigenvalues
- **Visualisation** : Three.js

---

## 📧 Contact

Pour toute question : [votre email]

**Enjoy climbing! 🧗‍♂️**
