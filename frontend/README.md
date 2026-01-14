# 🧗 HoldGen - Reconnaissance 3D de Prises d'Escalade

Système de matching automatique de prises d'escalade à partir de scans 3D, utilisant une approche en 3 phases : filtrage couleur, écrémage par eigenvalues, et matching RANSAC/ICP.

---

## 📋 Prérequis

- **Python 3.8+**
- **pip** (gestionnaire de paquets Python)

---

## 🔧 Installation

### 1. Cloner le repository

```bash
git clone https://github.com/matthieugravejat/ArkoseXECE.git
cd ArkoseXECE
```

### 2. Installer les dépendances

```bash
pip install open3d numpy flask flask-cors scikit-learn
```

**Liste complète des dépendances :**
- `open3d` - Traitement de nuages de points 3D
- `numpy` - Calculs matriciels
- `flask` - Serveur API backend
- `flask-cors` - Support CORS pour l'API
- `scikit-learn` - Clustering K-means (couleurs dominantes)

---

## 🚀 Lancement de l'application

### **Option 1 : Lancement automatique (recommandé)**

Utilisez le script de lancement :

```bash
python launch.py
```

Ce script lance automatiquement :
- Le serveur HTTP (port 8080) pour le frontend
- L'API Flask (port 5000) pour le backend

### **Option 2 : Lancement manuel**

Ouvrez **2 terminaux** :

#### Terminal 1 - Frontend (Serveur HTTP)
```bash
cd frontend
python3 -m http.server 8080
```

#### Terminal 2 - Backend (API Flask)
```bash
cd frontend
python matching_api.py
```

### **Accès à l'application**

Ouvrez votre navigateur et accédez à :
```
http://localhost:8080
```

---

## 📁 Structure du projet

```
Code_3D_NDP/
├── frontend/                              # Application web
│   ├── index.html                         # Interface utilisateur
│   ├── style.css                          # Styles CSS
│   ├── app.js                             # Logique JavaScript (Three.js)
│   └── matching_api.py                    # API Flask (backend)
│
├── Code_Matching_Complet/
│   └── Database_Prise_Couleurs/           # Base de données PLY (avec couleurs)
│       ├── Pusher_smoothie_2/
│       ├── SNAP_5_S/
│       ├── Teknik_NoShadowHands_1_5/
│       └── ... (9 familles de prises)
│
├── zip_mur_GLB/                           # Modèles 3D GLB pour visualisation
│   ├── SNAP_5_S_num1.glb
│   ├── Teknik_NoShadowHands_1_5_num1.glb
│   └── ... (62 fichiers .glb)
│
├── Matching_zones/
│   ├── matching_optimized.py              # Algorithme de matching 3 phases
│   ├── selecteur_zone_3d.py               # Sélection interactive de zones
│   └── zones_selectionnees/               # Zones scannées
│
├── select_and_isolate_holds.py            # Isolation automatique de prises
├── mur_color.ply                          # Scan du mur complet
└── README.md                              # Ce fichier
```

---

## 🎯 Fonctionnalités

### **1. Sélection interactive de zones** 🖱️
Utilisez l'interface web pour sélectionner des prises sur le mur 3D.

### **2. Matching en 3 phases** ⚡

Le système utilise un pipeline optimisé :

- **Phase 0 : Filtrage par couleur (HSV)**
  - Sépare couleurs saturées vs désaturées
  - Élimine les prises de couleurs incompatibles
  
- **Phase 1 : Écrémage par eigenvalues**
  - Compare la "silhouette" 3D des objets
  - Filtre rapide basé sur la forme géométrique
  
- **Phase 2 : RANSAC + ICP**
  - Matching précis sur les candidats restants
  - Score final combinant eigenvalues (30%) + ICP fitness (70%)

### **3. Visualisation 3D** 🎨
Affichage des résultats avec Three.js et modèles GLB haute qualité.

---

## 🧪 Utilisation avancée

### Tester le matching en ligne de commande

```bash
cd Matching_zones
python matching_optimized.py
```

Ce script analyse automatiquement la dernière zone sélectionnée et affiche :
- Les scores de matching
- Les métriques (fitness, RMSE, eigenvalues)
- Une visualisation 3D des top 3 résultats

### Ajuster les paramètres de matching

Éditez `matching_optimized.py` (ligne 876) :

```python
matcher = TwoPhaseHoldMatcher(
    source_path,
    eigen_threshold=0.1,       # Seuil eigenvalues (0.05 = strict, 0.2 = permissif)
    hue_threshold=60.0,        # Seuil teinte (35° = strict, 90° = permissif)
    value_threshold=40.0,      # Seuil luminosité pour couleurs désaturées
    use_color_filter=True      # Activer/désactiver le filtre couleur
)
```

---

## 📊 Base de données

La base contient **~60 prises** réparties en 9 familles :
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
