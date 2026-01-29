"""
HOLDGEN - API Backend Flask
Serveur pour le matching de prises d'escalade 3D
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import tempfile
import open3d as o3d
import numpy as np
from pathlib import Path
from colorsys import rgb_to_hsv
import traceback
import trimesh
from PIL import Image
import matplotlib.colors as mcolors
import gdown
import threading
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATCHING_ZONES_DIR = os.path.join(BASE_DIR, "Matching_zones")

"""
CONFIGURATION MONGODB ET CACHE
- Gère la connexion MongoDB et le cache des fichiers PLY
- Variables globales pour la progression du chargement
"""

# URI de connexion MongoDB
MONGO_URI = "mongodb+srv://dbPriseArkose_fake:maxmax@bddecexarkose.fegj7eg.mongodb.net/?appName=BddEceXArkose"
DB_NAME = "XCULT"

# Collections de couleurs à charger
COLOR_COLLECTIONS = [
    "Color_Red",
    "Color_Blue",
    "Color_Green",
    "Color_Yellow",
    "Color_Purple",
    "Color_Black"
]

# Répertoire de cache pour les fichiers PLY téléchargés
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache_prises")
os.makedirs(CACHE_DIR, exist_ok=True)

# Variables globales pour le cache
CACHE_LOADED = False
CACHE_PROGRESS = {
    'current': 0,
    'total': 0,
    'status': 'loading',  # 'loading', 'ready', 'error'
    'message': 'Initialisation...'
}
PLY_CACHE = []
CACHE_LOCK = threading.Lock()

"""
FONCTIONS COULEUR
- rgb_to_lab: Conversion RGB vers LAB pour comparaison perceptuelle
- get_color_info_hsv: Extraction couleur dominante d'un point cloud
- get_color_name: Classification des couleurs
- compute_color_match: Comparaison de deux couleurs
"""


def rgb_to_lab(rgb):
    """Convertit RGB (0-1) vers LAB pour comparaison perceptuelle."""
    rgb = np.clip(rgb, 0, 1)
    mask = rgb > 0.04045
    rgb_lin = np.where(mask, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    
    matrix = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ])
    
    xyz = rgb_lin @ matrix.T
    white = np.array([0.95047, 1.00000, 1.08883])
    xyz_norm = xyz / white
    
    mask = xyz_norm > 0.008856
    f_xyz = np.where(mask, xyz_norm ** (1/3), (7.787 * xyz_norm) + (16/116))
    
    L = (116 * f_xyz[..., 1]) - 16
    a = 500 * (f_xyz[..., 0] - f_xyz[..., 1])
    b = 200 * (f_xyz[..., 1] - f_xyz[..., 2])
    
    return np.stack([L, a, b], axis=-1)


def get_color_info_hsv(pcd):
    """Extrait les informations de couleur d'un point cloud en HSV."""
    if not pcd.has_colors():
        return None
    
    colors = np.asarray(pcd.colors)
    if len(colors) == 0:
        return None
    
    mean_rgb = colors.mean(axis=0)
    h, s, v = rgb_to_hsv(mean_rgb[0], mean_rgb[1], mean_rgb[2])
    
    return {
        'rgb': mean_rgb,
        'hsv': np.array([h * 360, s * 100, v * 100])
    }


def get_color_name(hsv):
    """Détermine le nom de la couleur à partir de la teinte (Hue)."""
    h, s, v = hsv[0], hsv[1], hsv[2]
    
    if s < 20:
        if v < 20:
            return "Noir"
        elif v > 80:
            return "Blanc"
        else:
            return "Gris"
    
    if h < 15 or h >= 345:
        return "Rouge"
    elif h < 45:
        return "Orange"
    elif h < 75:
        return "Jaune"
    elif h < 150:
        return "Vert"
    elif h < 210:
        return "Cyan"
    elif h < 270:
        return "Bleu"
    elif h < 330:
        return "Violet"
    else:
        return "Rouge"


def is_desaturated(hsv, threshold=20):
    """Vérifie si une couleur est désaturée."""
    return hsv[1] < threshold


def compute_color_match(source_hsv, target_hsv, hue_threshold=60.0, value_threshold=40.0):
    """Détermine si deux couleurs matchent."""
    source_desat = is_desaturated(source_hsv)
    target_desat = is_desaturated(target_hsv)
    
    if source_desat:
        if target_desat:
            value_diff = abs(source_hsv[2] - target_hsv[2])
            if value_diff < value_threshold:
                return True, f"Même type désaturé (ΔV={value_diff:.0f}%)"
            else:
                return False, f"Luminosité différente"
        else:
            return False, "Source désaturée, cible colorée"
    else:
        if target_desat:
            return False, "Source colorée, cible désaturée"
        else:
            h1, h2 = source_hsv[0], target_hsv[0]
            hue_diff = abs(h1 - h2)
            if hue_diff > 180:
                hue_diff = 360 - hue_diff
            
            if hue_diff < hue_threshold:
                return True, f"Même teinte (ΔH={hue_diff:.0f}°)"
            else:
                return False, f"Teinte différente"

# =====================================================================
# CHARGEMENT DES PRISES DEPUIS MONGODB
# =====================================================================

def load_prises_from_mongodb():
    """
    Charge toutes les prises depuis MongoDB et télécharge les fichiers PLY.
    Cette fonction s'exécute dans un thread séparé au démarrage.
    """
    global CACHE_LOADED, CACHE_PROGRESS, PLY_CACHE
    
    try:
        with CACHE_LOCK:
            CACHE_PROGRESS['status'] = 'loading'
            CACHE_PROGRESS['message'] = 'Connexion à MongoDB...'
        
        print(f"🔌 Connexion à MongoDB...")
        client = MongoClient(MONGO_URI, server_api=ServerApi('1'))
        
        client.admin.command('ping')
        print("✅ Connexion MongoDB réussie!")
        
        db = client[DB_NAME]
        
        total_prises = 0
        for collection_name in COLOR_COLLECTIONS:
            collection = db[collection_name]
            count = collection.count_documents({})
            total_prises += count
            print(f"📊 {collection_name}: {count} prises")
        
        with CACHE_LOCK:
            CACHE_PROGRESS['total'] = total_prises
            CACHE_PROGRESS['message'] = f'{total_prises} prises trouvées. Téléchargement en cours...'
        
        print(f"\n📥 Début du téléchargement de {total_prises} prises...")
        
        current_count = 0
        consecutive_errors = 0
        stop_downloading = False
        
        for collection_name in COLOR_COLLECTIONS:
            if stop_downloading:
                break
                
            collection = db[collection_name]
            
            prises = list(collection.find({}))
            
            for prise in prises:
                if stop_downloading:
                    break
                    
                try:
                    constructor_id = prise.get('constructor_ID', 'unknown')
                    set_name = prise.get('set_name', 'unknown')
                    google_link = prise.get('google_ply_link', '')
                    
                    if not google_link:
                        print(f"⚠️  Pas de lien Google Drive pour {constructor_id}")
                        current_count += 1
                        with CACHE_LOCK:
                            CACHE_PROGRESS['current'] = current_count
                        continue
                    
                    filename = f"{collection_name}_{set_name}_{constructor_id}.ply"
                    local_path = os.path.join(CACHE_DIR, filename)
                    
                    if not os.path.exists(local_path):
                        file_id = None
                        if '/file/d/' in google_link:
                            file_id = google_link.split('/file/d/')[1].split('/')[0]
                        elif 'id=' in google_link:
                            file_id = google_link.split('id=')[1].split('&')[0]
                        
                        if file_id:
                            url = f"https://drive.google.com/uc?id={file_id}"
                            try:
                                gdown.download(url, local_path, quiet=True)
                                print(f"✅ Téléchargé: {filename}")
                                consecutive_errors = 0
                            except Exception as e:
                                print(f"❌ Erreur téléchargement {filename}: {e}")
                                consecutive_errors += 1
                                
                                if consecutive_errors >= 3:
                                    print(f"\n⚠️ TROP D'ERREURS (Quota Google Drive probable). Arrêt des téléchargements.")
                                    print(f"ℹ️ Le système va continuer avec les {len(PLY_CACHE)} prises déjà disponibles.")
                                    stop_downloading = True
                                    break 
                                
                                current_count += 1
                                with CACHE_LOCK:
                                    CACHE_PROGRESS['current'] = current_count
                                continue
                        else:
                            print(f"⚠️  Lien invalide pour {constructor_id}")
                            current_count += 1
                            with CACHE_LOCK:
                                CACHE_PROGRESS['current'] = current_count
                            continue
                    else:
                        print(f"♻️  Déjà en cache: {filename}")
                        consecutive_errors = 0
                    
                    PLY_CACHE.append(local_path)
                    
                    current_count += 1
                    
                    with CACHE_LOCK:
                        CACHE_PROGRESS['current'] = current_count
                        percent = (current_count / total_prises) * 100
                        CACHE_PROGRESS['message'] = f'Chargement: {current_count}/{total_prises} ({percent:.1f}%)'
                    
                except Exception as e:
                    print(f"❌ Erreur pour prise {prise.get('_id')}: {e}")
                    current_count += 1
                    with CACHE_LOCK:
                        CACHE_PROGRESS['current'] = current_count

        
        client.close()
        
        with CACHE_LOCK:
            CACHE_LOADED = True
            CACHE_PROGRESS['status'] = 'ready'
            CACHE_PROGRESS['message'] = f'{len(PLY_CACHE)} prises chargées avec succès!'
        
        print(f"\n🎉 Chargement terminé! {len(PLY_CACHE)} prises disponibles.")
        
    except Exception as e:
        print(f"\n❌ ERREUR lors du chargement MongoDB: {e}")
        traceback.print_exc()
        with CACHE_LOCK:
            CACHE_PROGRESS['status'] = 'error'
            CACHE_PROGRESS['message'] = f'Erreur: {str(e)}'




"""
CONVERSION GLB VERS PLY AVEC COULEURS
Convertit un fichier GLB en PLY avec couleurs bakées depuis la texture
"""

def convert_glb_to_colored_ply(glb_path, ply_path):
    """
    Convertit un fichier GLB en PLY avec couleurs bakées depuis la texture.
    Inspiré de convert_glb_to_pcd_ply.py
    
    Args:
        glb_path: Chemin vers le fichier GLB source
        ply_path: Chemin où sauvegarder le fichier PLY
    
    Returns:
        o3d.geometry.PointCloud: Le point cloud coloré
    """
    tm = trimesh.load(glb_path)
    
    if isinstance(tm, trimesh.Scene):
        print("GLB détecté comme Scene → fusion des sous-mesh")
        mesh = trimesh.util.concatenate([m for m in tm.geometry.values()])
    else:
        mesh = tm
    
    vertices = mesh.vertices
    faces = mesh.faces
    
    if mesh.visual.kind != "texture":
        raise ValueError("Le GLB n'a pas de texture UV à baker.")
    
    base_tex = mesh.visual.material.baseColorTexture
    if base_tex is None:
        raise ValueError("Le matériau n'a pas de baseColorTexture.")
    texture_image = base_tex.convert("RGB")
    
    tex_w, tex_h = texture_image.size
    tex_pixels = np.array(texture_image)
    
    uv = mesh.visual.uv
    u = (uv[:,0] * (tex_w - 1)).astype(int)
    v = ((1 - uv[:,1]) * (tex_h - 1)).astype(int)
    
    vertex_colors = tex_pixels[v, u, :]
    
    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices)
    mesh_o3d.triangles = o3d.utility.Vector3iVector(faces)
    mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(vertex_colors / 255.0)
    mesh_o3d.compute_vertex_normals()
    
    o3d.io.write_triangle_mesh(ply_path, mesh_o3d)
    print(f"PLY coloré sauvegardé dans : {ply_path}")
    
    pcd = o3d.io.read_point_cloud(ply_path)
    return pcd


"""
CLASSE MATCHER POUR API
Implémente le pipeline de matching en 3 phases:
- Phase 0: Filtrage couleur
- Phase 1: Filtrage eigenvalues/forme
- Phase 2: Matching précis avec RANSAC/ICP
"""

class HoldMatcherAPI:
    """Version API du matcher pour le frontend."""
    
    def __init__(self, source_pcd, eigen_threshold=0.12, hue_threshold=60.0, 
                 value_threshold=40.0, use_color_filter=True):
        self.source_raw = source_pcd
        self.eigen_threshold = eigen_threshold
        self.hue_threshold = hue_threshold
        self.value_threshold = value_threshold
        self.use_color_filter = use_color_filter
        
        self.source_color_hsv = None
        self.source_color_name = None
        if self.source_raw.has_colors():
            color_info = get_color_info_hsv(self.source_raw)
            if color_info is not None:
                self.source_color_hsv = color_info['hsv']
                self.source_color_name = get_color_name(self.source_color_hsv)
                print(f"🎯 Prise source (HSV): {self.source_color_hsv.astype(int)}, Nom: {self.source_color_name}")
            else:
                print("⚠️ Prise source: couleurs non détectables")
        else:
            print("⚠️ Prise source: pas de couleurs dans le maillage")
            self.use_color_filter = False
        
        self.source_normalized, self.source_eigenvalues = self.normalize_and_get_eigenvalues(
            self.source_raw, "SOURCE"
        )
        
        self.phase0_results = []
        self.phase1_results = []
        self.phase2_results = []
    
    def normalize_and_get_eigenvalues(self, pcd, name="prise"):
        """Normalise une prise ET retourne ses eigenvalues normalisées."""
        points = np.asarray(pcd.points)
        center = points.mean(axis=0)
        centered_points = points - center
        
        cov_matrix = np.cov(centered_points.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        eigenvalues_normalized = eigenvalues / (eigenvalues.sum() + 1e-9)
        
        if np.linalg.det(eigenvectors) < 0:
            eigenvectors[:, 2] *= -1
        
        aligned_points = centered_points @ eigenvectors
        
        normalized_pcd = o3d.geometry.PointCloud()
        normalized_pcd.points = o3d.utility.Vector3dVector(aligned_points)
        
        if pcd.has_normals():
            normals = np.asarray(pcd.normals)
            normalized_normals = normals @ eigenvectors
            normalized_pcd.normals = o3d.utility.Vector3dVector(normalized_normals)
        
        if pcd.has_colors():
            normalized_pcd.colors = pcd.colors
        
        return normalized_pcd, eigenvalues_normalized
    
    def compute_eigenvalues_distance(self, eigen1, eigen2):
        """Distance euclidienne entre deux vecteurs d'eigenvalues."""
        return np.linalg.norm(eigen1 - eigen2)
    
    def find_targets(self, use_colored_database=True):
        """
        Trouve tous les fichiers PLY cibles.
        Utilise maintenant le cache MongoDB au lieu du système de fichiers local.
        """
        global CACHE_LOADED, PLY_CACHE
        
        if not CACHE_LOADED:
            print("⚠️ ATTENTION: Le cache MongoDB n'est pas encore chargé!")
            return []
        
        print(f"✅ Utilisation du cache: {len(PLY_CACHE)} prises disponibles")
        return PLY_CACHE.copy()


    
    def phase0_color_screening(self):
        """PHASE 0 : Pré-filtrage par couleur."""
        target_files = self.find_targets(use_colored_database=self.use_color_filter)
        
        if not target_files:
            return []
        
        if not self.use_color_filter or self.source_color_hsv is None:
            for target_path in target_files:
                self.phase0_results.append({
                    'path': target_path,
                    'name': os.path.basename(target_path),
                    'passed_phase0': True,
                    'reason': 'Filtre désactivé'
                })
            return target_files
        
        passed_files = []
        
        for target_path in target_files:
            target_name = os.path.basename(target_path)
            
            try:
                target_pcd = o3d.io.read_point_cloud(target_path)
                target_color_info = get_color_info_hsv(target_pcd)
                
                if target_color_info is None:
                    self.phase0_results.append({
                        'path': target_path,
                        'name': target_name,
                        'passed_phase0': False,
                        'reason': 'Pas de couleurs'
                    })
                    continue
                
                target_hsv = target_color_info['hsv']
                target_color_name = get_color_name(target_hsv)
                
                match, reason = compute_color_match(
                    self.source_color_hsv, target_hsv,
                    hue_threshold=self.hue_threshold,
                    value_threshold=self.value_threshold
                )
                
                if match:
                    passed_files.append(target_path)
                    print(f"✅ Prise compatible: {target_name} ({reason})")
                else:
                    # Log limité pour éviter le flood
                    if len(self.phase0_results) < 5:
                        print(f"❌ Prise rejetée: {target_name} ({reason})")
                
                self.phase0_results.append({
                    'path': target_path,
                    'name': target_name,
                    'passed_phase0': match,
                    'reason': reason,
                    'color_name': target_color_name,
                    'hsv': target_hsv.tolist() if isinstance(target_hsv, np.ndarray) else target_hsv
                })
                    
            except Exception as e:
                print(f"Erreur Phase 0 ({target_name}): {e}")
        
        print(f"🎨 Phase 0 terminée: {len(passed_files)}/{len(target_files)} prises retenues")
        return passed_files

    
    def phase1_eigenvalues_screening(self, target_files=None):
        """PHASE 1 : Écrémage par eigenvalues."""
        if target_files is None:
            target_files = self.find_targets(use_colored_database=self.use_color_filter)
        
        if not target_files:
            return
        
        for target_path in target_files:
            target_name = os.path.basename(target_path)
            
            try:
                target_raw = o3d.io.read_point_cloud(target_path)
                target_normalized, target_eigenvalues = self.normalize_and_get_eigenvalues(
                    target_raw, target_name
                )
                
                eigen_distance = self.compute_eigenvalues_distance(
                    self.source_eigenvalues, target_eigenvalues
                )
                
                source_bbox = self.source_normalized.get_axis_aligned_bounding_box()
                source_extent = source_bbox.get_extent()
                target_extent_norm = target_normalized.get_axis_aligned_bounding_box().get_extent()
                
                source_sorted = np.sort(source_extent)
                target_sorted = np.sort(target_extent_norm)
                dimension_ratios = target_sorted / (source_sorted + 1e-6)
                mean_ratio = np.mean(dimension_ratios)
                
                if eigen_distance < self.eigen_threshold:
                    self.phase1_results.append({
                        'name': target_name,
                        'path': target_path,
                        'normalized_pcd': target_normalized,
                        'eigenvalues': target_eigenvalues,
                        'eigen_distance': eigen_distance,
                        'dimension_ratio': mean_ratio,
                        'passed_phase1': True
                    })
                    print(f"✅ Forme acceptée: {target_name} (dist={eigen_distance:.3f})")
                else:
                    self.phase1_results.append({
                        'name': target_name,
                        'path': target_path,
                        'normalized_pcd': target_normalized,
                        'eigenvalues': target_eigenvalues,
                        'eigen_distance': eigen_distance,
                        'dimension_ratio': mean_ratio,
                        'passed_phase1': False
                    })
                    # Log limité
                    if len(self.phase1_results) < 5 or eigen_distance < self.eigen_threshold * 1.5:
                        print(f"❌ Forme rejetée: {target_name} (dist={eigen_distance:.3f} > {self.eigen_threshold})")

                
            except Exception as e:
                print(f"Erreur Phase 1: {e}")
    
    def phase2_detailed_matching(self, progress_callback=None):
        """PHASE 2 : RANSAC/ICP sur les candidats."""
        candidates = [r for r in self.phase1_results if r.get('passed_phase1', False)]
        
        if not candidates:
            return
        
        total = len(candidates)
        for i, candidate in enumerate(candidates):
            # Appeler le callback de progression
            if progress_callback:
                progress_callback(i + 1, total, candidate['name'])
            
            try:
                result = self.ransac_icp_matching(
                    self.source_normalized,
                    candidate['normalized_pcd'],
                    candidate['dimension_ratio']
                )
                
                if result:
                    eigen_score = 1.0 / (1.0 + candidate['eigen_distance'] * 5)
                    icp_fitness = result['fitness']
                    final_score = eigen_score * 0.3 + icp_fitness * 0.7
                    
                    self.phase2_results.append({
                        'name': candidate['name'],
                        'path': candidate['path'],
                        'eigen_distance': candidate['eigen_distance'],
                        'eigen_score': eigen_score,
                        'icp_fitness': icp_fitness,
                        'rmse': result['rmse'],
                        'scale': result['scale'],
                        'final_score': final_score,
                        'dimension_ratio': candidate['dimension_ratio']
                    })
                
            except Exception as e:
                print(f"Erreur Phase 2: {e}")
    
    def ransac_icp_matching(self, source_pcd, target_pcd, estimated_scale):
        """RANSAC + ICP pour matching précis."""
        source = source_pcd.voxel_down_sample(0.003)
        target = target_pcd.voxel_down_sample(0.005)
        
        if not source.has_normals():
            source.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
            )
        if not target.has_normals():
            target.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
            )
        
        source.translate(-source.get_center())
        target.translate(-target.get_center())
        
        scales_to_test = [
            estimated_scale * 0.95,
            estimated_scale, estimated_scale * 1.05
        ]
        
        best_result = None
        best_fitness = -1
        
        for scale in scales_to_test:
            source_scaled = o3d.geometry.PointCloud(source)
            if scale != 1.0:
                source_scaled.scale(scale, center=[0, 0, 0])
            
            source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                source_scaled,
                o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=100)
            )
            target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                target,
                o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=100)
            )
            
            result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                source_scaled, target,
                source_fpfh, target_fpfh,
                mutual_filter=True,
                max_correspondence_distance=0.05,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                ransac_n=3,
                checkers=[
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(0.05)
                ],
                criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 500)
            )
            
            result_icp = o3d.pipelines.registration.registration_icp(
                source_scaled, target,
                0.02,
                result_ransac.transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
            )
            
            if result_icp.fitness > best_fitness:
                best_fitness = result_icp.fitness
                best_result = {
                    'fitness': result_icp.fitness,
                    'rmse': result_icp.inlier_rmse,
                    'scale': scale
                }
        
        return best_result
    
    def run_pipeline(self, progress_callback=None):
        """Pipeline complet en 3 phases."""
        print(f"🚀 Démarrage du pipeline de matching...")
        
        # Phase 0
        passed_color = self.phase0_color_screening()
        print(f"📊 Phase 0 (Couleur): {len(passed_color)} candidats retenus")
        
        if not passed_color:
            print("❌ Aucun candidat après filtrage couleur.")
            return []
        
        # Phase 1
        self.phase1_eigenvalues_screening(target_files=passed_color)
        candidates_p1 = [r for r in self.phase1_results if r.get('passed_phase1', False)]
        print(f"📊 Phase 1 (Eigenvalues): {len(candidates_p1)} candidats retenus")
        
        if not candidates_p1:
            print("❌ Aucun candidat après filtrage eigenvalues.")
            return []
            
        # Phase 2 avec callback de progression
        print(f"🔄 Lancement Phase 2 (ICP/RANSAC) sur {len(candidates_p1)} candidats...")
        self.phase2_detailed_matching(progress_callback=progress_callback)
        print(f"✅ Phase 2 terminée. {len(self.phase2_results)} résultats trouvés.")
        
        return self.get_top_results(3)
    
    def get_top_results(self, n=3):
        """Retourne les N meilleurs résultats formatés pour le frontend."""
        if not self.phase2_results:
            return []
        
        sorted_results = sorted(self.phase2_results, key=lambda x: x['final_score'], reverse=True)
        top_n = sorted_results[:min(n, len(sorted_results))]
        
        results = []
        for i, result in enumerate(top_n, 1):
            # Créer une URL relative pour le fichier PLY
            ply_path = result.get('path', '')
            ply_url = f"/api/ply?path={ply_path}" if ply_path else None
            
            # Trouver le fichier GLB correspondant dans zip_mur_GLB/
            glb_url = None
            name_without_ext = result['name'].replace('.ply', '')
            glb_path = os.path.join(BASE_DIR, 'zip_mur_GLB', f'{name_without_ext}.glb')
            if os.path.exists(glb_path):
                glb_url = f"/api/glb?path={glb_path}"
            
            results.append({
                'rank': i,
                'name': name_without_ext,
                'score': round(result['final_score'] * 100, 1),
                'icp_fitness': round(result['icp_fitness'] * 100, 1),
                'eigen_score': round(result['eigen_score'] * 100, 1),
                'rmse_mm': round(result['rmse'] * 1000, 2),
                'scale': round(result['dimension_ratio'], 2),
                'ply_url': ply_url,
                'glb_url': glb_url
            })
        
        return results


# =====================================================================
# ROUTES API
# =====================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Vérification que l'API est en ligne."""
    return jsonify({'status': 'ok', 'message': 'HoldGen API is running'})


@app.route('/api/loading_status', methods=['GET'])
def loading_status():
    """
    Retourne le statut du chargement du cache MongoDB.
    Utilisé par le frontend pour afficher une barre de progression.
    """
    global CACHE_LOADED, PLY_CACHE
    
    with CACHE_LOCK:
        response = dict(CACHE_PROGRESS)
    
    if CACHE_LOADED and response.get('status') != 'ready':
        response['status'] = 'ready'
        response['message'] = f'{len(PLY_CACHE)} prises disponibles'
    
    if len(PLY_CACHE) > 0 and response.get('total', 0) == 0:
        response['total'] = len(PLY_CACHE)
        response['current'] = len(PLY_CACHE)
        
    return jsonify(response)



@app.route('/api/ply', methods=['GET'])
def serve_ply():
    """Sert un fichier PLY pour la visualisation 3D."""
    from flask import send_file
    
    ply_path = request.args.get('path', '')
    
    if not ply_path:
        return jsonify({'error': 'Chemin non spécifié'}), 400
    
    if not os.path.exists(ply_path):
        return jsonify({'error': 'Fichier non trouvé'}), 404
    
    if not ply_path.lower().endswith('.ply'):
        return jsonify({'error': 'Format non autorisé'}), 403
    
    return send_file(ply_path, mimetype='application/octet-stream')


@app.route('/api/glb', methods=['GET'])
def serve_glb():
    """Sert un fichier GLB pour la visualisation 3D."""
    from flask import send_file
    
    glb_path = request.args.get('path', '')
    
    if not glb_path:
        return jsonify({'error': 'Chemin non spécifié'}), 400
    
    if not os.path.exists(glb_path):
        return jsonify({'error': 'Fichier non trouvé'}), 404
    
    if not glb_path.lower().endswith('.glb'):
        return jsonify({'error': 'Format non autorisé'}), 403
    
    return send_file(glb_path, mimetype='model/gltf-binary')


@app.route('/api/preview_ply', methods=['POST'])
def preview_ply():
    """
    Charge un fichier PLY et retourne les points au format JSON.
    Résout le problème du format 'double' non supporté par Three.js PLYLoader.
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Aucun fichier reçu'}), 400
        
        file = request.files['file']
        
        if not file.filename.lower().endswith('.ply'):
            return jsonify({'error': 'Format non supporté'}), 400
        
        # Sauvegarder temporairement
        with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name
        
        try:
            # Charger avec Open3D
            pcd = o3d.io.read_point_cloud(tmp_path)
            points = np.asarray(pcd.points).astype(np.float32)
            
            # Centrer les points
            center = points.mean(axis=0)
            points = points - center
            
            # Récupérer les couleurs si présentes
            colors = None
            if pcd.has_colors():
                colors = (np.asarray(pcd.colors) * 255).astype(np.uint8).tolist()
            
            return jsonify({
                'success': True,
                'points': points.tolist(),
                'colors': colors,
                'count': len(points)
            })
            
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/ply_json', methods=['GET'])
def ply_to_json():
    """
    Convertit un fichier PLY en JSON pour la visualisation 3D.
    """
    ply_path = request.args.get('path', '')
    
    if not ply_path or not os.path.exists(ply_path):
        return jsonify({'error': 'Fichier non trouvé'}), 404
    
    try:
        pcd = o3d.io.read_point_cloud(ply_path)
        points = np.asarray(pcd.points).astype(np.float32)
        
        # Centrer
        center = points.mean(axis=0)
        points = points - center
        
        colors = None
        if pcd.has_colors():
            colors = (np.asarray(pcd.colors) * 255).astype(np.uint8).tolist()
        
        return jsonify({
            'success': True,
            'points': points.tolist(),
            'colors': colors,
            'count': len(points)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/convert_glb', methods=['POST'])
def convert_glb():
    """
    Convertit un fichier GLB en PLY avec couleurs et crée une session pour l'isolation des prises.
    Cette route fonctionne comme /api/load_wall mais accepte les fichiers GLB.
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Aucun fichier reçu'}), 400
        
        file = request.files['file']
        
        if not file.filename.lower().endswith('.glb'):
            return jsonify({'error': 'Format non supporté. Seuls les fichiers .glb sont acceptés'}), 400
        
        # Sauvegarder temporairement le GLB
        with tempfile.NamedTemporaryFile(suffix='.glb', delete=False) as tmp_glb:
            file.save(tmp_glb.name)
            tmp_glb_path = tmp_glb.name
        
        # Créer un fichier temporaire pour le PLY
        tmp_ply_path = tmp_glb_path.replace('.glb', '.ply')
        
        try:
            # Convertir GLB vers PLY avec couleurs
            pcd = convert_glb_to_colored_ply(tmp_glb_path, tmp_ply_path)
            
            if len(pcd.points) == 0:
                return jsonify({'error': 'Fichier GLB vide ou corrompu'}), 400
            
            # Détecter le plan du mur (comme dans load_wall)
            plane_model, wall_mask = detect_wall_plane_api(pcd)
            
            if plane_model is None:
                return jsonify({'error': 'Aucun plan de mur détecté'}), 400
            
            # Créer une session (comme dans load_wall)
            import uuid
            session_id = str(uuid.uuid4())
            
            # Préparer les points pour le frontend
            points = np.asarray(pcd.points).astype(np.float32)
            center = points.mean(axis=0)
            points_centered = points - center
            
            # MODIFICATION: Stocker le center dans la session
            wall_sessions[session_id] = {
                'pcd': pcd,
                'plane_model': list(plane_model),
                'wall_mask': wall_mask,
                'tmp_path': tmp_ply_path,  # Garder le PLY converti
                'center': center.tolist(),  # Centre pour recentrer les points reconstruits
                'isolated_holds': [],
                'hold_indices': []
            }
            
            # Récupérer les couleurs
            colors = None
            if pcd.has_colors():
                colors = (np.asarray(pcd.colors) * 255).astype(np.uint8).tolist()
            
            wall_points_count = int(np.sum(wall_mask))
            
            return jsonify({
                'success': True,
                'session_id': session_id,
                'total_points': len(points),
                'wall_points': wall_points_count,
                'non_wall_points': len(points) - wall_points_count,
                'points': points_centered.tolist(),
                'colors': colors,
                'center': center.tolist(),
                'message': f'GLB converti avec succès: {len(points)} points'
            })
            
        finally:
            # Nettoyer seulement le fichier GLB temporaire
            # Le PLY est gardé dans la session
            if os.path.exists(tmp_glb_path):
                os.remove(tmp_glb_path)
                
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/match', methods=['POST'])
def match_hold():
    """
    Endpoint principal pour le matching.
    Reçoit un fichier PLY et retourne les 3 meilleurs matches.
    """
    try:
        # Vérifier qu'un fichier a été envoyé
        if 'file' not in request.files:
            return jsonify({'error': 'Aucun fichier reçu'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'Nom de fichier vide'}), 400
        
        # Vérifier l'extension
        if not file.filename.lower().endswith('.ply'):
            return jsonify({'error': 'Format invalide. Seuls les fichiers .ply sont acceptés'}), 400
        
        # Sauvegarder temporairement le fichier
        with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name
        
        try:
            # Charger le point cloud
            source_pcd = o3d.io.read_point_cloud(tmp_path)
            
            if len(source_pcd.points) == 0:
                return jsonify({'error': 'Fichier PLY vide ou corrompu'}), 400
            
            # Extraire les infos de la source
            source_info = {
                'points': len(source_pcd.points),
                'has_colors': source_pcd.has_colors(),
                'color_name': None
            }
            
            if source_pcd.has_colors():
                color_info = get_color_info_hsv(source_pcd)
                if color_info:
                    source_info['color_name'] = get_color_name(color_info['hsv'])
            
            # Récupérer le paramètre wall (optionnel)
            wall_id = request.form.get('wall_id', 'unknown')
            
            # Lancer le matcher
            matcher = HoldMatcherAPI(
                source_pcd,
                eigen_threshold=0.12,
                hue_threshold=40.0,
                value_threshold=40.0,
                use_color_filter=source_pcd.has_colors()
            )
            
            results = matcher.run_pipeline()
            
            return jsonify({
                'success': True,
                'wall_id': wall_id,
                'source_info': source_info,
                'results': results,
                'message': f'{len(results)} correspondances trouvées'
            })
            
        finally:
            # Nettoyer le fichier temporaire
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# Variable globale pour stocker la progression
matching_progress = {}

@app.route('/api/match_stream', methods=['POST'])
def match_stream():
    """
    Endpoint pour le matching avec progression en temps réel via SSE.
    Retourne les événements de progression pendant la Phase 2.
    """
    from flask import Response
    import queue
    import threading
    
    # File d'attente pour les messages de progression
    progress_queue = queue.Queue()
    
    def run_matching(tmp_path, source_pcd, source_info, wall_id):
        """Exécute le matching dans un thread séparé."""
        try:
            def progress_callback(current, total, name):
                """Callback appelé à chaque fichier traité en Phase 2."""
                progress_queue.put({
                    'type': 'progress',
                    'current': current,
                    'total': total,
                    'name': name,
                    'percent': int((current / total) * 100)
                })
            
            # Lancer le matcher avec callback
            matcher = HoldMatcherAPI(
                source_pcd,
                eigen_threshold=0.12,
                hue_threshold=60.0,
                value_threshold=40.0,
                use_color_filter=source_pcd.has_colors()
            )
            
            results = matcher.run_pipeline(progress_callback=progress_callback)
            
            # Envoyer les résultats finaux
            progress_queue.put({
                'type': 'complete',
                'success': True,
                'wall_id': wall_id,
                'source_info': source_info,
                'results': results,
                'message': f'{len(results)} correspondances trouvées'
            })
            
        except Exception as e:
            traceback.print_exc()
            progress_queue.put({
                'type': 'error',
                'error': str(e)
            })
        finally:
            # Nettoyer le fichier temporaire
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            # Marquer la fin
            progress_queue.put(None)
    
    def generate():
        """Génère les événements SSE."""
        while True:
            msg = progress_queue.get()
            if msg is None:
                break
            yield f"data: {json.dumps(msg)}\n\n"
    
    try:
        # Vérifier le fichier
        if 'file' not in request.files:
            return jsonify({'error': 'Aucun fichier reçu'}), 400
        
        file = request.files['file']
        
        if not file.filename.lower().endswith('.ply'):
            return jsonify({'error': 'Format invalide'}), 400
        
        # Sauvegarder temporairement
        with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name
        
        # Charger le point cloud
        source_pcd = o3d.io.read_point_cloud(tmp_path)
        
        if len(source_pcd.points) == 0:
            os.remove(tmp_path)
            return jsonify({'error': 'Fichier PLY vide'}), 400
        
        # Infos source
        source_info = {
            'points': len(source_pcd.points),
            'has_colors': source_pcd.has_colors(),
            'color_name': None
        }
        
        if source_pcd.has_colors():
            color_info = get_color_info_hsv(source_pcd)
            if color_info:
                source_info['color_name'] = get_color_name(color_info['hsv'])
        
        wall_id = request.form.get('wall_id', 'unknown')
        
        # Lancer le matching dans un thread
        thread = threading.Thread(
            target=run_matching,
            args=(tmp_path, source_pcd, source_info, wall_id)
        )
        thread.start()
        
        # Retourner le stream SSE
        return Response(
            generate(),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'Access-Control-Allow-Origin': '*'
            }
        )
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# =====================================================================
# STOCKAGE SESSIONS ET CONSTANTES DBSCAN
# =====================================================================

"""
MODE MUR: Isolation manuelle et automatique des prises
- wall_sessions: Stockage temporaire des murs chargés
- DBSCAN_EPS, MIN_CLUSTER_SIZE: Paramètres de clustering
"""

wall_sessions = {}
SEUIL_RANSAC_CM = 0.09
NB_ITERATIONS_DECAPAGE = 3
MIN_POINTS_POUR_MUR = 5000

SEUIL_LUMINOSITE_NOIR = 0.45
SEUIL_SATURATION_MIN = 0.20
HUE_BOIS_MIN = 0.05
HUE_BOIS_MAX = 0.105
SEUIL_SATURATION_FLUO = 0.40

DBSCAN_EPS = 0.03
DBSCAN_MIN_POINTS = 10
MIN_CLUSTER_SIZE = 100

def get_dominant_color_name_dbscan(pcd):
    """Détermine la couleur dominante d'un cluster."""
    if not pcd.has_colors() or len(pcd.points) == 0: 
        return "inconnu"
    rgb = np.asarray(pcd.colors)
    avg_rgb = np.median(rgb, axis=0)
    h, s, v = mcolors.rgb_to_hsv(avg_rgb)
    
    if v < SEUIL_LUMINOSITE_NOIR: 
        return "noir"
    if s < 0.15: 
        return "gris"
    if h < 0.05 or h > 0.92: 
        return "rouge"
    if 0.05 <= h < 0.11: 
        return "orange"
    if 0.11 <= h < 0.28: 
        return "jaune"
    if 0.28 <= h < 0.48: 
        return "vert"
    if 0.48 <= h < 0.70: 
        return "bleu"
    if 0.70 <= h < 0.92: 
        return "violet"
    return "autre"

def auto_isolate_dbscan_api(pcd):
    """
    Isole les prises automatiquement via DBSCAN.
    Retourne: liste de clusters (indices), liste des points candidats (sans mur)
    """
    current_pcd = pcd
    current_original_indices = np.arange(len(pcd.points))
    
    print(f"Auto-Isolation: {len(current_pcd.points)} points initiaux")
    
    for i in range(NB_ITERATIONS_DECAPAGE):
        plane_model, inliers = current_pcd.segment_plane(distance_threshold=SEUIL_RANSAC_CM/100,
                                                         ransac_n=3,
                                                         num_iterations=1000)
        if len(inliers) < MIN_POINTS_POUR_MUR:
            break
            
        current_pcd = current_pcd.select_by_index(inliers, invert=True)
        current_original_indices = np.delete(current_original_indices, inliers)
        
    print(f"Après RANSAC: {len(current_pcd.points)} points")
    
    if current_pcd.has_colors():
        rgb = np.asarray(current_pcd.colors)
        hsv = mcolors.rgb_to_hsv(rgb)
        h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]

        mask_noir = (v < SEUIL_LUMINOSITE_NOIR)
        mask_bois = (h >= HUE_BOIS_MIN) & (h <= HUE_BOIS_MAX) & (s < SEUIL_SATURATION_FLUO)
        mask_gris = (s < SEUIL_SATURATION_MIN) & (v >= SEUIL_LUMINOSITE_NOIR)

        final_mask = mask_noir | (~mask_bois & ~mask_gris)
        ind_keep = np.where(final_mask)[0]
        
        current_pcd = current_pcd.select_by_index(ind_keep)
        current_original_indices = current_original_indices[ind_keep]
        
    print(f"Après Couleur: {len(current_pcd.points)} points")
    
    if len(current_pcd.points) == 0:
        return [], []

    try:
        current_pcd, ind_clean = current_pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=1.0)
        current_original_indices = current_original_indices[ind_clean]
    except:
        pass
        
    candidate_indices = current_original_indices.tolist()

    labels = np.array(current_pcd.cluster_dbscan(eps=DBSCAN_EPS, min_points=DBSCAN_MIN_POINTS))
    
    if len(labels) == 0:
        return [], candidate_indices
        
    unique_labels = np.unique(labels)
    clusters = []
    
    print(f"DBSCAN: {len(unique_labels)} clusters trouvés")
    
    for label in unique_labels:
        if label == -1: continue # Bruit
        
        idx_cluster_local = np.where(labels == label)[0]
        
        if len(idx_cluster_local) < MIN_CLUSTER_SIZE:
            continue
            
        # Récupérer les vrais indices
        cluster_original_indices = current_original_indices[idx_cluster_local].tolist()
        clusters.append(cluster_original_indices)
        
    print(f"Clusters valides conservés: {len(clusters)}")
    return clusters, candidate_indices
    

from collections import deque

"""
DÉTECTION ET ISOLATION DES PRISES
- detect_wall_plane_api: Détecte le plan principal du mur
- distance_to_plane_api: Calcule distance point-plan
- expand_from_seed_api: Expansion spatiale depuis seed point
- reconstruct_surface_using_alpha_shape: Reconstruction surface avec Alpha Shape
"""

def detect_wall_plane_api(pcd, config=None):
    """Détecte le plan principal du mur."""
    if config is None:
        config = {
            "ransac_distance_threshold": 0.02,
            "ransac_num_iterations": 2000,
            "min_plane_points": 10000,
            "min_plane_extent": 0.5,
        }
    
    points = np.asarray(pcd.points)
    n_points = len(points)
    
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=config["ransac_distance_threshold"],
        ransac_n=3,
        num_iterations=config["ransac_num_iterations"]
    )
    
    if len(inliers) < config["min_plane_points"]:
        return None, None
    
    plane_points = points[inliers]
    extent = np.max(plane_points, axis=0) - np.min(plane_points, axis=0)
    
    if np.sum(extent >= config["min_plane_extent"]) < 2:
        return None, None
    
    wall_mask = np.zeros(n_points, dtype=bool)
    wall_mask[inliers] = True
    
    return plane_model, wall_mask


def distance_to_plane_api(point, plane_model):
    """Calcule la distance d'un point au plan."""
    a, b, c, d = plane_model
    normal = np.array([a, b, c])
    norm = np.linalg.norm(normal)
    return abs((np.dot(point, normal) + d) / norm)


def reconstruct_surface_using_alpha_shape(pcd, alpha=0.08, density_multiplier=3.0):
    """
    Reconstruit une surface pleine à partir d'un nuage de points partiel (troué)
    en utilisant un maillage Alpha Shape, puis ré-échantillonne des points.
    
    Args:
        pcd: Point cloud source (avec trous)
        alpha: Paramètre Alpha Shape - plus grand = maillage moins serré
        density_multiplier: Multiplicateur pour le nombre de points générés
    """
    try:
        alphas_to_try = [alpha, alpha * 1.5, alpha * 2.0, alpha * 0.7]
        best_mesh = None
        best_triangle_count = 0
        best_alpha = alpha
        
        for test_alpha in alphas_to_try:
            try:
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, test_alpha)
                triangle_count = len(mesh.triangles)
                
                if triangle_count > best_triangle_count:
                    best_triangle_count = triangle_count
                    best_mesh = mesh
                    best_alpha = test_alpha
            except:
                continue
        
        if best_mesh is None or best_triangle_count == 0:
            print(f"⚠️ Alpha Shape échoué, retour au nuage original")
            return pcd
            
        mesh = best_mesh
        
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        
        # Tenter de remplir les trous dans le mesh (si Open3D le supporte)
        try:
            mesh = mesh.subdivide_midpoint(number_of_iterations=1)
        except:
            pass
            
        mesh.compute_vertex_normals()
        
        n_points = int(max(len(pcd.points) * density_multiplier, 1000))
        
        try:
            pcd_new = mesh.sample_points_poisson_disk(number_of_points=n_points)
        except:
            pcd_new = mesh.sample_points_uniformly(number_of_points=n_points)
        
        if pcd.has_colors():
            pcd_tree = o3d.geometry.KDTreeFlann(pcd)
            original_colors = np.asarray(pcd.colors)
            new_colors = []
            
            for pt in pcd_new.points:
                [_, idx, _] = pcd_tree.search_knn_vector_3d(pt, 1)
                new_colors.append(original_colors[idx[0]])
            
            pcd_new.colors = o3d.utility.Vector3dVector(new_colors)
            
        print(f"🔧 Reconstruction Alpha Shape (alpha={best_alpha:.3f}, x{density_multiplier}): {len(pcd.points)} -> {len(pcd_new.points)} pts")
        return pcd_new
        
    except Exception as e:
        print(f"⚠️ Erreur reconstruction Alpha Shape: {e}")
        return pcd


def expand_from_seed_api(pcd, seed_idx, plane_model, wall_mask, config=None):
    """Expansion spatiale depuis un point de départ jusqu'au mur."""
    if config is None:
        config = {
            "expansion_radius": 0.03,
            "stop_at_wall_distance": -1,
        }
    
    points = np.asarray(pcd.points)
    n_points = len(points)
    
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    
    visited = np.zeros(n_points, dtype=bool)
    cluster = []
    queue = deque([seed_idx])
    
    radius = config["expansion_radius"]
    wall_dist_threshold = config["stop_at_wall_distance"]
    
    while queue:
        current_idx = queue.popleft()
        
        if visited[current_idx]:
            continue
        
        visited[current_idx] = True
        cluster.append(current_idx)
        
        [k, idx, _] = pcd_tree.search_radius_vector_3d(points[current_idx], radius)
        
        for neighbor_idx in idx[1:]:
            if visited[neighbor_idx]:
                continue
            
            if wall_mask[neighbor_idx]:
                continue
            
            neighbor_point = points[neighbor_idx]
            dist_to_wall = distance_to_plane_api(neighbor_point, plane_model)
            
            if dist_to_wall <= wall_dist_threshold:
                continue
            
            queue.append(neighbor_idx)
    
    return cluster


@app.route('/api/load_wall', methods=['POST'])
def load_wall():
    """
    Charge un fichier PLY de mur, détecte le plan et retourne les points.
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Aucun fichier reçu'}), 400
        
        file = request.files['file']
        
        if not file.filename.lower().endswith('.ply'):
            return jsonify({'error': 'Format invalide. Seuls les fichiers .ply sont acceptés'}), 400
        
        # Sauvegarder le fichier
        with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name
        
        # Charger avec Open3D
        pcd = o3d.io.read_point_cloud(tmp_path)
        
        if len(pcd.points) == 0:
            os.remove(tmp_path)
            return jsonify({'error': 'Fichier PLY vide'}), 400
        
        # Détecter le plan du mur
        plane_model, wall_mask = detect_wall_plane_api(pcd)
        
        if plane_model is None:
            os.remove(tmp_path)
            return jsonify({'error': 'Aucun plan de mur détecté'}), 400
        
        # Créer une session
        import uuid
        session_id = str(uuid.uuid4())
        
        # Préparer les points pour le frontend
        points = np.asarray(pcd.points).astype(np.float32)
        center = points.mean(axis=0)
        points_centered = points - center
        
        # MODIFICATION: Stocker le center dans la session
        wall_sessions[session_id] = {
            'pcd': pcd,
            'plane_model': list(plane_model),
            'wall_mask': wall_mask,
            'tmp_path': tmp_path,
            'center': center.tolist(),  # Centre pour recentrer les points reconstruits
            'isolated_holds': [],
            'hold_indices': []
        }
        
        colors = None
        if pcd.has_colors():
            colors = (np.asarray(pcd.colors) * 255).astype(np.uint8).tolist()
        
        wall_points_count = int(np.sum(wall_mask))
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'total_points': len(points),
            'wall_points': wall_points_count,
            'non_wall_points': len(points) - wall_points_count,
            'points': points_centered.tolist(),
            'colors': colors,
            'center': center.tolist()
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/isolate_hold', methods=['POST'])
def isolate_hold():
    """
    Isole une prise à partir d'un point cliqué.
    Reçoit: session_id, point_index
    Retourne: les indices des points de la prise isolée
    """
    try:
        data = request.get_json()
        
        session_id = data.get('session_id')
        point_index = data.get('point_index')
        algo = data.get('algo', 'manual')  # 'manual' ou 'dbscan'
        
        if not session_id or session_id not in wall_sessions:
            return jsonify({'error': 'Session invalide'}), 400
        
        if point_index is None:
            return jsonify({'error': 'Index de point manquant'}), 400
        
        session = wall_sessions[session_id]
        pcd = session['pcd']
        plane_model = session['plane_model']
        wall_mask = session['wall_mask']
        
        # Vérifier que le point n'est pas sur le mur (sauf en mode DBSCAN où on fait confiance au mapping)
        if algo != 'dbscan' and wall_mask[point_index]:
            return jsonify({'error': 'Point sur le mur, veuillez sélectionner une prise'}), 400
        
        # Vérifier que le point n'est pas déjà dans une prise isolée
        for i, indices in enumerate(session['hold_indices']):
            if point_index in indices:
                return jsonify({
                    'success': True,
                    'already_isolated': True,
                    'hold_id': i,
                    'message': f'Point déjà dans la prise {i+1}'
                })
        
        # Expansion spatiale
        cluster = []
        
        if algo == 'dbscan':
            # Mode DBSCAN: Utiliser les clusters pré-calculés
            if 'dbscan_map' in session and point_index in session['dbscan_map']:
                cluster_idx = session['dbscan_map'][point_index]
                cluster = session['dbscan_clusters'][cluster_idx]
            else:
                return jsonify({'error': 'Ce point ne fait pas partie d\'une prise détectée (bruit ou mur).'}), 400
        else:
            # Mode Manuel: Region Growing
            cluster = expand_from_seed_api(pcd, point_index, plane_model, wall_mask)
        
        if len(cluster) < 10:
            return jsonify({'error': 'Prise trop petite (moins de 10 points)'}), 400
        
        # Ajouter à la session
        hold_id = len(session['isolated_holds'])
        session['hold_indices'].append(set(cluster))
        
        # Créer le point cloud de la prise
        points = np.asarray(pcd.points)[cluster]
        colors = None
        if pcd.has_colors():
            colors = np.asarray(pcd.colors)[cluster]
        
        hold_pcd = o3d.geometry.PointCloud()
        hold_pcd.points = o3d.utility.Vector3dVector(points)
        if colors is not None:
            hold_pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # RECONSTRUCTION DE SURFACE (Bouchage de trous)
        # Alpha plus grand = maillage moins serré = moins de trous
        # density_multiplier = plus de points générés pour mieux couvrir
        hold_pcd = reconstruct_surface_using_alpha_shape(hold_pcd, alpha=0.10, density_multiplier=4.0)
        
        session['isolated_holds'].append(hold_pcd)
        
        # MODIFICATION: Centrer les points reconstruits avec le même centre que le mur
        wall_center = np.array(session.get('center', [0, 0, 0]))
        rec_points = (np.asarray(hold_pcd.points) - wall_center).tolist()
        rec_colors = []
        if hold_pcd.has_colors():
            rec_colors = np.asarray(hold_pcd.colors).tolist()
            
        return jsonify({
            'success': True,
            'hold_id': hold_id,
            'point_count': len(rec_points), 
            'indices': cluster,
            'reconstructed_points': rec_points,
            'reconstructed_colors': rec_colors
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/extract_all_holds', methods=['POST'])
def extract_all_holds():
    """
    Isole automatiquement TOUTES les prises détectées par DBSCAN.
    Reçoit: session_id
    Retourne: liste des prises isolées (id, indices, point_count)
    """
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        
        if not session_id or session_id not in wall_sessions:
            return jsonify({'error': 'Session invalide'}), 400
            
        session = wall_sessions[session_id]
        pcd = session['pcd']
        
        # 1. Lancer DBSCAN (ou récupérer si déjà fait ?)
        # Pour l'instant, on relance pour être sûr d'avoir les clusters propres
        clusters_indices, _ = auto_isolate_dbscan_api(pcd)
        
        if not clusters_indices:
             return jsonify({'success': True, 'count': 0, 'holds': []})

        extracted_holds = []
        
        # 2. Convertir chaque cluster en prise isolée
        for cluster in clusters_indices:
            # Vérifier unicité (facultatif si on fait un reset violent avant)
            # Ici on ajoute tout
            
            hold_id = len(session['isolated_holds'])
            session['hold_indices'].append(set(cluster))
            
            # Créer le point cloud de la prise
            points = np.asarray(pcd.points)[cluster]
            colors = None
            if pcd.has_colors():
                colors = np.asarray(pcd.colors)[cluster]
            
            hold_pcd = o3d.geometry.PointCloud()
            hold_pcd.points = o3d.utility.Vector3dVector(points)
            if colors is not None:
                hold_pcd.colors = o3d.utility.Vector3dVector(colors)
            
            session['isolated_holds'].append(hold_pcd)
            
            extracted_holds.append({
                'hold_id': hold_id,
                'indices': cluster,
                'point_count': len(cluster)
            })
            
        return jsonify({
            'success': True,
            'count': len(extracted_holds),
            'holds': extracted_holds,
            'message': f'{len(extracted_holds)} prises extraites automatiquement.'
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/match_isolated_holds', methods=['POST'])
def match_isolated_holds():
    """
    Lance le matching pour toutes les prises isolées.
    Retourne les top 3 matches pour chaque prise.
    """
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        
        if not session_id or session_id not in wall_sessions:
            return jsonify({'error': 'Session invalide'}), 400
        
        session = wall_sessions[session_id]
        isolated_holds = session['isolated_holds']
        
        # Support pour analyse par lot (progress bar)
        hold_index = data.get('hold_index') # Optionnel
        
        target_holds = []
        target_indices = []
        
        if hold_index is not None:
            if 0 <= hold_index < len(isolated_holds):
                target_holds = [isolated_holds[hold_index]]
                target_indices = [hold_index]
            else:
                 return jsonify({'error': 'Index de prise invalide'}), 400
        else:
            target_holds = isolated_holds
            target_indices = range(len(isolated_holds))

        if len(target_holds) == 0:
            return jsonify({'error': 'Aucune prise à analyser'}), 400
        
        all_results = []
        
        for i, hold_pcd in zip(target_indices, target_holds):
            print(f"\n[Matching] Prise {i+1}/{len(isolated_holds)}...")
            
            # Lancer le matcher
            matcher = HoldMatcherAPI(
                hold_pcd,

                eigen_threshold=0.12,
                hue_threshold=40.0,
                value_threshold=40.0,
                use_color_filter=hold_pcd.has_colors()
            )
            
            results = matcher.run_pipeline()
            
            # Infos sur la prise source
            hold_info = {
                'hold_id': i,
                'point_count': len(hold_pcd.points),
                'has_colors': hold_pcd.has_colors(),
                'color_name': None
            }
            
            if hold_pcd.has_colors():
                color_info = get_color_info_hsv(hold_pcd)
                if color_info:
                    hold_info['color_name'] = get_color_name(color_info['hsv'])
            
            all_results.append({
                'hold_info': hold_info,
                'matches': results
            })
        
        return jsonify({
            'success': True,
            'total_holds': len(isolated_holds),
            'results': all_results
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/get_hold_matches', methods=['POST'])
def get_hold_matches():
    """
    Lance le matching pour une seule prise isolée.
    """
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        hold_id = data.get('hold_id')
        
        if not session_id or session_id not in wall_sessions:
            return jsonify({'error': 'Session invalide'}), 400
        
        session = wall_sessions[session_id]
        isolated_holds = session['isolated_holds']
        
        if hold_id is None or hold_id >= len(isolated_holds):
            return jsonify({'error': 'ID de prise invalide'}), 400
        
        hold_pcd = isolated_holds[hold_id]
        
        print(f"\n[Matching] Prise {hold_id+1}...")
        
        # Lancer le matcher
        matcher = HoldMatcherAPI(
            hold_pcd,
            eigen_threshold=0.12,
            hue_threshold=40.0,
            value_threshold=40.0,
            use_color_filter=hold_pcd.has_colors()
        )
        
        results = matcher.run_pipeline()
        
        # Infos sur la prise source
        hold_info = {
            'hold_id': hold_id,
            'point_count': len(hold_pcd.points),
            'has_colors': hold_pcd.has_colors(),
            'color_name': None
        }
        
        if hold_pcd.has_colors():
            color_info = get_color_info_hsv(hold_pcd)
            if color_info:
                hold_info['color_name'] = get_color_name(color_info['hsv'])
        
        return jsonify({
            'success': True,
            'hold_info': hold_info,
            'matches': results
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/auto_isolate', methods=['POST'])
def auto_isolate():
    """Endpoint pour l'isolation automatique (DBSCAN)."""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        
        if not session_id or session_id not in wall_sessions:
            return jsonify({'error': 'Session invalide'}), 400
            
        session = wall_sessions[session_id]
        pcd = session['pcd']
        
        # Lancer l'algo
        clusters_indices, candidate_indices = auto_isolate_dbscan_api(pcd)
        
        if not clusters_indices:
            return jsonify({
                'success': True,
                'count': 0,
                'candidate_indices': candidate_indices,
                'message': 'Aucun cluster détecté. Essayez de changer les paramètres.'
            })
            
        # Stocker les clusters pour le mode interactif
        session['dbscan_clusters'] = clusters_indices
        session['dbscan_map'] = {}  # point_index -> cluster_index_in_list
        
        for i, indices in enumerate(clusters_indices):
            for idx in indices:
                session['dbscan_map'][idx] = i
                
        return jsonify({
            'success': True,
            'count': len(clusters_indices),
            'candidate_indices': candidate_indices,
            'message': f'Mode DBSCAN activé : {len(clusters_indices)} clusters pré-calculés. Cliquez sur une prise pour la sélectionner.'
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/clear_session', methods=['POST'])
def clear_session():
    """Nettoie une session."""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        
        if session_id and session_id in wall_sessions:
            session = wall_sessions[session_id]
            # Supprimer le fichier temporaire
            if 'tmp_path' in session and os.path.exists(session['tmp_path']):
                os.remove(session['tmp_path'])
            del wall_sessions[session_id]
        
        return jsonify({'success': True})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("=" * 60)
    print("🧗 HoldGen API Server")
    print("=" * 60)
    print(f"📂 Base directory: {BASE_DIR}")
    print(f"💾 Cache directory: {CACHE_DIR}")
    print(f"🚀 Starting server on http://localhost:5001")
    print("=" * 60)
    
    # Démarrer le thread de chargement MongoDB
    print("\n🔄 Lancement du chargement MongoDB en arrière-plan...")
    loading_thread = threading.Thread(target=load_prises_from_mongodb, daemon=True)
    loading_thread.start()
    print("✅ Thread de chargement démarré!")
    print("\n📊 Progression disponible sur: http://localhost:5001/api/loading_status\n")
    
    app.run(host='0.0.0.0', port=5001, debug=True)