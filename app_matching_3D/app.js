/**
 * HoldGen - Gestionnaire de Salle d'Escalade
 * Application Frontend avec visualisation 3D Three.js
 * 
 * Mode MUR: Charge un PLY de mur, sélection des prises par clic, matching
 */

/*
CONFIGURATION ET VARIABLES GLOBALES
- API_URL: Endpoint du serveur Flask
- wallScene, wallCamera, wallRenderer: Contexte Three.js pour le mur
- isolatedHolds: Stockage des prises isolées
- HOLD_COLORS: Palette de couleurs pour les prises
*/

const API_URL = 'http://localhost:5001';

// Variables globales Three.js
let wallScene, wallCamera, wallRenderer, wallControls;
let wallPointCloud = null;
let wallAnimationId = null;

let activeViewers = new Map();
let viewerObserver = null;

// État de l'application
let currentSessionId = null;
let wallCenter = { x: 0, y: 0, z: 0 };
let isolatedHolds = [];
let holdMatchResults = {};
let resultViewers = [];
let raycaster = null;
let mouse = null;
let isolationMode = 'manual';
let dbscanPointCloud = null;
let dbscanCandidateIndices = null;

let originalWallPointCount = 0;

const HOLD_COLORS = [
    0xff0000, 0x00ff00, 0x0000ff, 0xffff00, 0xff00ff,
    0x00ffff, 0xff8000, 0x8000ff, 0x00ff80, 0xff0080,
];

document.addEventListener('DOMContentLoaded', () => {
    /*
    CHARGEMENT INITIAL
    - Suivi de la progression du chargement MongoDB
    - Affichage barre de progression
    */
    
    async function checkDatabaseLoading() {
        const loadingScreen = document.getElementById('initialLoadingScreen');
        const progressBar = document.getElementById('dbProgressBar');
        const loadingText = document.getElementById('dbLoadingText');
        const loadingDetail = document.getElementById('dbLoadingDetail');

        if (!loadingScreen) return;

        console.log("🚀 Démarrage du suivi de chargement MongoDB...");

        let retryCount = 0;
        const maxRetries = 5;
        let stuckCount = 0; // Compteur pour détecter si bloqué à 0/0

        while (true) {
            try {
                const response = await fetch(`${API_URL}/api/loading_status`);

                if (!response.ok) {
                    throw new Error(`Erreur HTTP: ${response.status}`);
                }

                const data = await response.json();
                
                console.log("Loading status:", data); // Debug

                if (data.status === 'ready') {
                    // Chargement terminé
                    if (progressBar) progressBar.style.width = '100%';
                    if (loadingText) loadingText.textContent = data.message || 'Prêt!';
                    if (loadingDetail) loadingDetail.textContent = 'Prêt!';

                    console.log("✅ Chargement MongoDB terminé!");

                    // Attendre un peu pour montrer le 100%
                    await new Promise(resolve => setTimeout(resolve, 800));

                    // Masquer l'écran
                    loadingScreen.classList.add('hidden');
                    setTimeout(() => {
                        loadingScreen.style.display = 'none';
                    }, 500);

                    break;
                } else if (data.status === 'error') {
                    // Erreur
                    if (loadingDetail) {
                        loadingDetail.textContent = data.message;
                        loadingDetail.style.color = '#ff4d4f';
                    }
                    console.error("❌ Erreur chargement MongoDB:", data.message);
                    // On ne break pas pour permettre un retry éventuel côté serveur ou restart
                    await new Promise(resolve => setTimeout(resolve, 5000));
                } else {
                    // En cours
                    const current = data.current || 0;
                    const total = data.total || 1; // Éviter division par 0
                    const percent = Math.min(100, Math.round((current / total) * 100));

                    if (progressBar) progressBar.style.width = `${percent}%`;
                    if (loadingText) loadingText.textContent = `${current} / ${total} prises chargées`;
                    if (loadingDetail) loadingDetail.textContent = data.message || 'Chargement en cours...';
                    
                    // Détecter si bloqué à 0/0 pendant trop longtemps
                    if (current === 0 && total <= 1) {
                        stuckCount++;
                        console.log(`Stuck count: ${stuckCount}`);
                        
                        // Après 5 secondes bloqué à 0/0, vérifier si l'API répond bien
                        if (stuckCount >= 5) {
                            console.log("⚠️ Semble bloqué, vérification health check...");
                            try {
                                const healthResponse = await fetch(`${API_URL}/api/health`);
                                if (healthResponse.ok) {
                                    // L'API répond, peut-être le cache est déjà chargé
                                    console.log("API répond, forçage passage...");
                                    if (progressBar) progressBar.style.width = '100%';
                                    if (loadingText) loadingText.textContent = 'Base de données prête';
                                    if (loadingDetail) loadingDetail.textContent = 'Prêt!';
                                    
                                    await new Promise(resolve => setTimeout(resolve, 500));
                                    loadingScreen.classList.add('hidden');
                                    setTimeout(() => {
                                        loadingScreen.style.display = 'none';
                                    }, 500);
                                    break;
                                }
                            } catch (e) {
                                console.log("Health check failed:", e);
                            }
                        }
                    } else {
                        stuckCount = 0; // Reset si on a du progrès
                    }
                }

                // Attendre avant la prochaine vérification
                await new Promise(resolve => setTimeout(resolve, 1000));
                retryCount = 0; // Reset retry count on success

            } catch (error) {
                console.error('Erreur connexion API loading:', error);
                retryCount++;

                if (loadingDetail) loadingDetail.textContent = `Erreur de connexion (tentative ${retryCount}/${maxRetries})...`;

                if (retryCount >= maxRetries) {
                    if (loadingDetail) {
                        loadingDetail.textContent = "Impossible de joindre le serveur. Vérifiez qu'il est lancé.";
                        loadingDetail.style.color = '#ff4d4f';
                    }
                    // Continue trying but slower
                    await new Promise(resolve => setTimeout(resolve, 5000));
                } else {
                    await new Promise(resolve => setTimeout(resolve, 2000));
                }
            }
        }
    }

    // Lancer la vérification immédiatement
    checkDatabaseLoading();

    // Éléments DOM
    const wallGroups = document.querySelectorAll('.wall-group');
    const dropzoneContainer = document.getElementById('dropzoneContainer');
    const dropzone = document.getElementById('dropzone');
    const selectedWallInfo = document.getElementById('selectedWallInfo');
    const wallNameSpan = selectedWallInfo.querySelector('.wall-name');
    const closeDropzoneBtn = document.getElementById('closeDropzone');
    const fileInput = document.getElementById('fileInput');
    const previewSection = document.getElementById('previewSection');
    const previewViewer = document.getElementById('previewViewer');
    const previewCanvas = document.getElementById('previewCanvas');
    const fileName = document.getElementById('fileName');
    const fileSize = document.getElementById('fileSize');
    const removeFileBtn = document.getElementById('removeFile');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const loadingInline = document.getElementById('loadingInline');
    const loadingText = document.getElementById('loadingText');
    const resultsContainer = document.getElementById('resultsContainer');
    const resultsGrid = document.getElementById('resultsGrid');
    const sourceInfo = document.getElementById('sourceInfo');
    const noResults = document.getElementById('noResults');
    const newAnalysisBtn = document.getElementById('newAnalysisBtn');

    // Nouveaux éléments pour le mode mur
    const holdCountBadge = document.getElementById('holdCountBadge');
    const selectionInstructions = document.getElementById('selectionInstructions');
    const autoDetectBtn = document.getElementById('autoDetectBtn');
    const clearSelectionBtn = document.getElementById('clearSelectionBtn');

    /**
     * Lance l'extraction complète automatique
     */
    async function autoExtractAllHolds() {
        if (!currentSessionId) {
            alert("Veuillez d'abord charger un mur.");
            return;
        }

        try {
            showLoading('Extraction automatique de TOUTES les prises...');

            const response = await fetch(`${API_URL}/api/extract_all_holds`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: currentSessionId })
            });

            const data = await response.json();
            hideLoading();

            if (!data.success) {
                alert(data.message || 'Erreur lors de l\'extraction');
                return;
            }

            if (data.count === 0) {
                alert("Aucune prise détectée.");
                return;
            }

            // Ajouter toutes les prises
            data.holds.forEach((hold, index) => {
                const holdColor = HOLD_COLORS[isolatedHolds.length % HOLD_COLORS.length];

                isolatedHolds.push({
                    holdId: hold.hold_id,
                    indices: hold.indices,
                    color: holdColor,
                    pointCount: hold.point_count,
                    matches: null
                });

                // Visualisation immédiate
                highlightIsolatedHold(hold.indices, holdColor);
            });

            updateHoldCountBadge();
            alert(`${data.count} prises ont été extraites et ajoutées !`);

        } catch (error) {
            hideLoading();
            console.error(error);
            alert('Erreur: ' + error.message);
        }
    }

    // --- Event Listeners ---
    if (autoDetectBtn) {
        autoDetectBtn.addEventListener('click', () => {
            if (isolationMode === 'dbscan') {
                deactivateDBSCANMode();
            } else {
                activateDBSCANMode();
            }
        });
    }

    // Bouton Auto-Extraction Complète
    const autoExtractAllBtn = document.getElementById('autoExtractAllBtn');
    if (autoExtractAllBtn) {
        autoExtractAllBtn.addEventListener('click', autoExtractAllHolds);
    }

    if (clearSelectionBtn) {
        clearSelectionBtn.addEventListener('click', clearIsolatedHolds);
    }

    // État de l'application
    let selectedWall = null;
    let currentFile = null;
    let currentGLBUrl = null; // Pour le modal de comparaison

    /**
     * Initialise le viewer Three.js pour le mur
     */
    function initWallViewer() {
        const container = previewViewer;
        const width = container.clientWidth || 600;
        const height = container.clientHeight || 400;

        // Scene
        wallScene = new THREE.Scene();
        wallScene.background = new THREE.Color(0x1a1a2e);

        // Camera
        wallCamera = new THREE.PerspectiveCamera(60, width / height, 0.01, 1000);
        wallCamera.position.set(0, 0, 3);

        // Renderer
        wallRenderer = new THREE.WebGLRenderer({
            canvas: previewCanvas,
            antialias: true
        });
        wallRenderer.setSize(width, height);
        wallRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

        // Controls
        wallControls = new THREE.OrbitControls(wallCamera, wallRenderer.domElement);
        wallControls.enableDamping = true;
        wallControls.dampingFactor = 0.05;
        wallControls.autoRotate = false;

        // Lumières
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        wallScene.add(ambientLight);

        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(1, 1, 1);
        wallScene.add(directionalLight);

        // Raycaster pour la sélection
        raycaster = new THREE.Raycaster();
        raycaster.params.Points.threshold = 0.02;
        mouse = new THREE.Vector2();

        // Événement clic pour sélection
        previewCanvas.addEventListener('click', onWallClick);

        // Animation loop
        function animate() {
            wallAnimationId = requestAnimationFrame(animate);
            wallControls.update();
            wallRenderer.render(wallScene, wallCamera);
        }
        animate();

        // Resize handler
        window.addEventListener('resize', resizeWallViewer);
    }

    /**
     * Redimensionne le viewer
     */
    function resizeWallViewer() {
        if (wallRenderer && previewViewer && previewViewer.offsetWidth > 0) {
            const w = previewViewer.clientWidth;
            const h = previewViewer.clientHeight;
            wallCamera.aspect = w / h;
            wallCamera.updateProjectionMatrix();
            wallRenderer.setSize(w, h);
        }
    }

    /**
     * Gestion du clic sur le mur pour sélectionner une prise
     */
    async function onWallClick(event) {
        if (!currentSessionId) return;

        // Choix du point cloud à raycaste (Mur normal OU Nuage DBSCAN)
        let targetPointCloud = wallPointCloud;
        if (isolationMode === 'dbscan' && dbscanPointCloud) {
            targetPointCloud = dbscanPointCloud;
        }

        if (!targetPointCloud) return;

        const rect = previewCanvas.getBoundingClientRect();
        mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

        raycaster.setFromCamera(mouse, wallCamera);
        const intersects = raycaster.intersectObject(targetPointCloud);

        if (intersects.length > 0) {
            let pointIndex = intersects[0].index;

            // Si mode DBSCAN, mapper vers l'index global
            if (isolationMode === 'dbscan' && dbscanCandidateIndices) {
                pointIndex = dbscanCandidateIndices[pointIndex];
            }

            console.log('Point cliqué:', pointIndex, '(Mode:', isolationMode, ')');

            await isolateHoldFromPoint(pointIndex);
        }
    }

    /**
     * Isole une prise à partir d'un point cliqué
     */
    async function isolateHoldFromPoint(pointIndex) {
        try {
            showLoading('Isolation de la prise...');

            const response = await fetch(`${API_URL}/api/isolate_hold`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: currentSessionId,
                    point_index: pointIndex,
                    algo: isolationMode
                })
            });

            const data = await response.json();
            hideLoading();

            if (!data.success) {
                if (data.already_isolated) {
                    console.log('Point déjà isolé:', data.message);
                    return;
                }
                throw new Error(data.error || 'Erreur inconnue');
            }

            console.log('Prise isolée:', data);

            // Ajouter à la liste des prises isolées
            const holdColor = HOLD_COLORS[isolatedHolds.length % HOLD_COLORS.length];

            // MODIFICATION: Intégrer les points reconstruits directement dans le nuage de points
            let allIndices = [...data.indices]; // Indices originaux
            
            if (data.reconstructed_points && data.reconstructed_points.length > 0) {
                console.log(`Intégration de ${data.reconstructed_points.length} points reconstruits au mur`);
                const newIndices = addReconstructedPointsToWall(
                    data.reconstructed_points, 
                    data.reconstructed_colors
                );
                allIndices = [...data.indices, ...newIndices]; // Fusionner tous les indices
            }

            isolatedHolds.push({
                holdId: data.hold_id,
                indices: allIndices, // TOUS les indices (originaux + reconstruits)
                color: holdColor,
                pointCount: allIndices.length,
                matches: null
            });

            // Colorer TOUS les points de la prise (originaux + reconstruits)
            highlightIsolatedHold(allIndices, holdColor);
            updateHoldCountBadge();

        } catch (error) {
            hideLoading();
            console.error('Erreur isolation:', error);
            alert('Erreur: ' + error.message);
        }
    }

    /**
     * Ajoute les points reconstruits directement dans le wallPointCloud
     * @returns {Array} Les nouveaux indices ajoutés
     */
    function addReconstructedPointsToWall(reconstructedPoints, reconstructedColors) {
        if (!wallPointCloud || !reconstructedPoints || reconstructedPoints.length === 0) {
            return [];
        }

        const currentPos = wallPointCloud.geometry.attributes.position.array;
        const currentCol = wallPointCloud.geometry.attributes.color.array;
        const currentOriginal = wallPointCloud.userData.originalColors;
        
        const numNewPoints = reconstructedPoints.length;
        const currentNumPoints = currentPos.length / 3;
        
        console.log(`Adding ${numNewPoints} reconstructed points to wall (current: ${currentNumPoints})`);
        
        // Créer de nouveaux arrays plus grands
        const newPos = new Float32Array(currentPos.length + numNewPoints * 3);
        const newCol = new Float32Array(currentCol.length + numNewPoints * 3);
        const newOriginal = new Float32Array(currentOriginal.length + numNewPoints * 3);
        
        // Copier les anciens
        newPos.set(currentPos);
        newCol.set(currentCol);
        newOriginal.set(currentOriginal);
        
        // Ajouter les nouveaux points
        const newIndices = [];
        for (let i = 0; i < numNewPoints; i++) {
            const pt = reconstructedPoints[i];
            const col = reconstructedColors[i] || [0.5, 0.5, 0.5]; // Gris par défaut
            
            const idx = currentNumPoints + i;
            newIndices.push(idx);
            
            newPos[idx * 3] = pt[0];
            newPos[idx * 3 + 1] = pt[1];
            newPos[idx * 3 + 2] = pt[2];
            
            // Couleurs (0-1 float)
            newCol[idx * 3] = col[0];
            newCol[idx * 3 + 1] = col[1];
            newCol[idx * 3 + 2] = col[2];
            
            newOriginal[idx * 3] = col[0];
            newOriginal[idx * 3 + 1] = col[1];
            newOriginal[idx * 3 + 2] = col[2];
        }
        
        // Créer un nouveau geometry (on ne peut pas juste resize en Three.js)
        const newGeometry = new THREE.BufferGeometry();
        newGeometry.setAttribute('position', new THREE.BufferAttribute(newPos, 3));
        newGeometry.setAttribute('color', new THREE.BufferAttribute(newCol, 3));
        
        // Remplacer le geometry
        wallPointCloud.geometry.dispose();
        wallPointCloud.geometry = newGeometry;
        wallPointCloud.userData.originalColors = newOriginal;
        
        // Si en mode DBSCAN, ajouter aussi au nuage DBSCAN
        if (isolationMode === 'dbscan' && dbscanPointCloud) {
            addReconstructedPointsToDBSCAN(reconstructedPoints, reconstructedColors, newIndices);
        }
        
        return newIndices;
    }

    /**
     * Ajoute les points reconstruits au dbscanPointCloud (si actif)
     */
    function addReconstructedPointsToDBSCAN(reconstructedPoints, reconstructedColors, globalIndices) {
        if (!dbscanPointCloud) return;
        
        const currentPos = dbscanPointCloud.geometry.attributes.position.array;
        const currentCol = dbscanPointCloud.geometry.attributes.color.array;
        
        const numNewPoints = reconstructedPoints.length;
        const currentNumPoints = currentPos.length / 3;
        
        // Créer de nouveaux arrays
        const newPos = new Float32Array(currentPos.length + numNewPoints * 3);
        const newCol = new Float32Array(currentCol.length + numNewPoints * 3);
        
        newPos.set(currentPos);
        newCol.set(currentCol);
        
        for (let i = 0; i < numNewPoints; i++) {
            const pt = reconstructedPoints[i];
            const col = reconstructedColors[i] || [0.5, 0.5, 0.5];
            
            const idx = currentNumPoints + i;
            
            newPos[idx * 3] = pt[0];
            newPos[idx * 3 + 1] = pt[1];
            newPos[idx * 3 + 2] = pt[2];
            
            newCol[idx * 3] = col[0];
            newCol[idx * 3 + 1] = col[1];
            newCol[idx * 3 + 2] = col[2];
        }
        
        // Mettre à jour le mapping global
        if (dbscanCandidateIndices) {
            const newMapping = [...dbscanCandidateIndices];
            globalIndices.forEach(gi => newMapping.push(gi));
            dbscanCandidateIndices = newMapping;
        }
        
        // Remplacer le geometry
        const newGeometry = new THREE.BufferGeometry();
        newGeometry.setAttribute('position', new THREE.BufferAttribute(newPos, 3));
        newGeometry.setAttribute('color', new THREE.BufferAttribute(newCol, 3));
        
        dbscanPointCloud.geometry.dispose();
        dbscanPointCloud.geometry = newGeometry;
    }

    /**
     * Met en surbrillance une prise isolée
     */
    function highlightIsolatedHold(indices, color) {
        // 1. Mise à jour du mur principal (toujours, pour qu'il soit à jour quand on le réaffiche)
        if (wallPointCloud) {
            const colors = wallPointCloud.geometry.attributes.color.array;
            const r = ((color >> 16) & 255) / 255;
            const g = ((color >> 8) & 255) / 255;
            const b = (color & 255) / 255;

            indices.forEach(idx => {
                if (idx * 3 + 2 < colors.length) {
                    colors[idx * 3] = r;
                    colors[idx * 3 + 1] = g;
                    colors[idx * 3 + 2] = b;
                }
            });

            wallPointCloud.geometry.attributes.color.needsUpdate = true;
        }

        // 2. Mise à jour du nuage DBSCAN (s'il est visible)
        if (dbscanPointCloud && dbscanCandidateIndices) {
            console.log("Mise à jour visualisation DBSCAN...");
            const dbscanColors = dbscanPointCloud.geometry.attributes.color.array;
            const r = ((color >> 16) & 255) / 255;
            const g = ((color >> 8) & 255) / 255;
            const b = (color & 255) / 255;

            // Optimisation : créer un Set pour recherche rapide
            const indicesSet = new Set(indices);
            let matchCount = 0;

            // Parcourir tous les candidats DBSCAN pour voir s'ils correspondent à la prise isolée
            for (let localIdx = 0; localIdx < dbscanCandidateIndices.length; localIdx++) {
                const globalIdx = dbscanCandidateIndices[localIdx];
                if (indicesSet.has(globalIdx)) {
                    dbscanColors[localIdx * 3] = r;
                    dbscanColors[localIdx * 3 + 1] = g;
                    dbscanColors[localIdx * 3 + 2] = b;
                    matchCount++;
                }
            }
            console.log(`Updated ${matchCount} points in DBSCAN cloud.`);

            dbscanPointCloud.geometry.attributes.color.needsUpdate = true;
        } else {
            console.log("DBSCAN cloud not active or indices missing.");
        }
    }

    /**
     * Met à jour le badge du nombre de prises
     */
    function updateHoldCountBadge() {
        if (holdCountBadge) {
            holdCountBadge.textContent = `${isolatedHolds.length} prise(s) sélectionnée(s)`;
            holdCountBadge.style.display = isolatedHolds.length > 0 ? 'block' : 'none';
        }

        // Activer le bouton d'analyse si des prises sont sélectionnées
        if (analyzeBtn) {
            analyzeBtn.disabled = isolatedHolds.length === 0;
        }
    }

    /**
     * Charge un fichier PLY de mur via l'API
     */
    async function loadWallPLY(file) {
        console.log('📦 Chargement du mur:', file.name);

        const formData = new FormData();
        formData.append('file', file);

        try {
            showLoading('Chargement du mur...');

            const response = await fetch(`${API_URL}/api/load_wall`, {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!data.success) {
                throw new Error(data.error || 'Erreur inconnue');
            }

            hideLoading();
            console.log('✅ Mur chargé:', data.total_points, 'points');
            console.log('   Plan détecté:', data.wall_points, 'points sur le mur');

            // Sauvegarder la session
            currentSessionId = data.session_id;
            wallCenter = data.center;
            isolatedHolds = [];
            holdMatchResults = {};

            // Créer le point cloud
            displayWallPointCloud(data);

            // Afficher les instructions
            if (selectionInstructions) {
                selectionInstructions.style.display = 'block';
            }

            // Forcer le resize
            setTimeout(resizeWallViewer, 100);

        } catch (error) {
            hideLoading();
            console.error('❌ Erreur:', error);
            alert('Erreur lors du chargement du mur.\n' + error.message);
        }
    }

    /**
     * Affiche le point cloud du mur
     */
    function displayWallPointCloud(data) {
        // Supprimer l'ancien point cloud
        if (wallPointCloud) {
            wallScene.remove(wallPointCloud);
            wallPointCloud.geometry.dispose();
            wallPointCloud.material.dispose();
        }

        // Créer la géométrie
        const geometry = new THREE.BufferGeometry();
        const positions = new Float32Array(data.points.flat());
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

        // Calculer le bounding box
        geometry.computeBoundingBox();
        const size = new THREE.Vector3();
        geometry.boundingBox.getSize(size);
        const maxDim = Math.max(size.x, size.y, size.z);

        // Taille des points
        const pointSize = Math.max(0.003, maxDim * 0.015);

        // Créer les couleurs
        let colorArray;
        if (data.colors && data.colors.length > 0) {
            colorArray = new Float32Array(data.colors.flat().map(c => c / 255));
        } else {
            // Couleur par défaut (gris)
            colorArray = new Float32Array(data.points.length * 3);
            for (let i = 0; i < colorArray.length; i += 3) {
                colorArray[i] = 0.5;
                colorArray[i + 1] = 0.5;
                colorArray[i + 2] = 0.5;
            }
        }
        geometry.setAttribute('color', new THREE.BufferAttribute(colorArray, 3));

        // Matériel
        const material = new THREE.PointsMaterial({
            size: pointSize,
            vertexColors: true,
            sizeAttenuation: true
        });

        // Créer le point cloud
        wallPointCloud = new THREE.Points(geometry, material);
        wallPointCloud.userData.originalColors = new Float32Array(colorArray); // Sauvegarder couleurs originales
        wallScene.add(wallPointCloud);

        // NOUVEAU: Stocker le nombre de points originaux
        originalWallPointCount = data.points.length;
        console.log(`Original wall point count stored: ${originalWallPointCount}`);

        // Positionner la caméra
        const cameraDistance = maxDim * 2;
        wallCamera.position.set(0, 0, cameraDistance);
        wallCamera.lookAt(0, 0, 0);
        wallControls.target.set(0, 0, 0);
        wallControls.update();
    }

    async function analyzeHolds() {
        if (!currentSessionId || isolatedHolds.length === 0) {
            alert("Aucune prise sélectionnée à analyser.");
            return;
        }

        try {
            // Initialisation UI Loading
            showLoading('Analyse détaillée en cours...');
            if (progressContainer) progressContainer.style.display = 'block';
            if (progressBar) progressBar.style.width = '0%';
            if (progressPercent) progressPercent.textContent = '0%';
            if (progressDetail) progressDetail.textContent = `0 / ${isolatedHolds.length} prises`;

            // Reset des résultats
            holdMatchResults = {};
            cleanupResultViewers();
            resultsGrid.innerHTML = ''; // On vide la grille
            resultsContainer.classList.add('visible'); // On affiche le conteneur vide

            // Info source initiale
            sourceInfo.innerHTML = `
                <div class="source-info-item">
                    <span class="source-info-label">Prises analysées:</span>
                    <span class="source-info-value" id="analyzedCount">0</span> / <span class="source-info-value">${isolatedHolds.length}</span>
                </div>
                <div class="source-info-item">
                    <span class="source-info-label">Mur:</span>
                    <span class="source-info-value">${selectedWall?.dataset.name || 'Non spécifié'}</span>
                </div>
            `;

            // Boucle séquentielle pour la barre de progression
            for (let i = 0; i < isolatedHolds.length; i++) {
                const hold = isolatedHolds[i];

                // Update UI avant l'appel
                const percent = Math.round((i / isolatedHolds.length) * 100);
                if (progressBar) progressBar.style.width = `${percent}%`;
                if (progressPercent) progressPercent.textContent = `${percent}%`;
                if (progressDetail) progressDetail.textContent = `Analyse prise ${i + 1} / ${isolatedHolds.length}`;

                // Appel API pour UNE prise
                const response = await fetch(`${API_URL}/api/match_isolated_holds`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        session_id: currentSessionId,
                        hold_index: i // On demande juste cet index
                    })
                });

                const data = await response.json();

                if (!data.success) {
                    console.error(`Erreur matching prise ${i}:`, data.error);
                    continue; // On continue avec les autres
                }

                // Traitement du résultat (liste de 1 élément normally)
                if (data.results && data.results.length > 0) {
                    const result = data.results[0];
                    holdMatchResults[result.hold_info.hold_id] = result;
                    isolatedHolds[i].matches = result.matches;
                    isolatedHolds[i].color_name = result.hold_info.color_name;

                    // Ajout progressif à l'affichage
                    appendMatchResult(result, i);
                }

                // Update compteur
                const countElem = document.getElementById('analyzedCount');
                if (countElem) countElem.textContent = i + 1;
            }

            // Fin
            if (progressBar) progressBar.style.width = '100%';
            if (progressPercent) progressPercent.textContent = '100%';
            if (progressDetail) progressDetail.textContent = 'Terminé !';

            setTimeout(() => {
                hideLoading();
                if (progressContainer) progressContainer.style.display = 'none';
            }, 500);

            resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });

        } catch (error) {
            hideLoading();
            if (progressContainer) progressContainer.style.display = 'none';
            console.error('Erreur matching:', error);
            alert('Erreur: ' + error.message);
        }
    }

    /**
     * Ajoute UN résultat à la grille (pour l'affichage progressif)
     */
    function appendMatchResult(result, index) {
        const hold = isolatedHolds[index];
        const colorHex = hold ? '#' + hold.color.toString(16).padStart(6, '0') : '#ff0000';
        const matches = result.matches || [];

        const html = `
            <div class="hold-result-row" data-hold-id="${result.hold_info.hold_id}">
                <div class="hold-label" style="background: ${colorHex}">
                    <span class="hold-number">Prise ${index + 1}</span>
                    <button class="hold-meta" onclick="openComparisonModal(${index})" title="Comparer en 3D">➕</button>
                </div>
                <div class="matches-row">
                    ${matches.length > 0 ? matches.slice(0, 3).map((match, mi) => `
                        <div class="result-card rank-${mi + 1}">
                            <div class="result-rank">${mi + 1}</div>
                            <div class="result-viewer" id="matchViewer${index}_${mi}"></div>
                            <div class="result-name">${match.name}</div>
                            <div class="result-score">
                                <div class="score-value">${match.score}%</div>
                                <div class="score-label">Score de correspondance</div>
                            </div>
                            <div class="result-details">
                                <div class="detail-item">
                                    <div class="detail-label-container">
                                        <span class="detail-label">Précision</span>
                                        <span class="info-icon" data-tooltip="% de superposition entre le scan et le modèle">ℹ︎</span>
                                    </div>
                                    <span class="detail-value">${match.icp_fitness}%</span>
                                </div>
                                <div class="detail-item">
                                    <div class="detail-label-container">
                                        <span class="detail-label">Forme</span>
                                        <span class="info-icon" data-tooltip="Similitude de la courbure et géométrie">ℹ︎</span>
                                    </div>
                                    <span class="detail-value">${match.eigen_score}%</span>
                                </div>
                                <div class="detail-item">
                                    <div class="detail-label-container">
                                        <span class="detail-label">Distance</span>
                                        <span class="info-icon" data-tooltip="Écart moyen en mm entre les points">ℹ︎</span>
                                    </div>
                                    <span class="detail-value">${match.rmse_mm}mm</span>
                                </div>
                                <div class="detail-item">
                                    <div class="detail-label-container">
                                        <span class="detail-label">Taille</span>
                                        <span class="info-icon" data-tooltip="Facteur d'échelle (1.0 = taille réelle)">ℹ︎</span>
                                    </div>
                                    <span class="detail-value">${match.scale}×</span>
                                </div>
                            </div>
                        </div>
                    `).join('') : '<div class="no-match">Aucune correspondance trouvée</div>'}
                </div>
            </div >
            `;

        // Insertion HTML
        resultsGrid.insertAdjacentHTML('beforeend', html);

        // Lazy loading: créer des placeholders au lieu des viewers immédiats
        setTimeout(() => {
            matches.slice(0, 3).forEach((match, mi) => {
                const viewerId = `matchViewer${index}_${mi}`;
                const container = document.getElementById(viewerId);

                if (container && (match.glb_url || match.ply_url)) {
                    const modelUrl = match.glb_url ? `${API_URL}${match.glb_url}` : `${API_URL}${match.ply_url}`;
                    const isGLB = !!match.glb_url;

                    // Ajouter les data attributes pour lazy loading
                    container.dataset.viewerId = viewerId;
                    container.dataset.modelUrl = modelUrl;
                    container.dataset.isGlb = isGLB;

                    // Observer le container
                    if (viewerObserver) {
                        viewerObserver.observe(container);
                    }
                }
            });
        }, 100);

        noResults.classList.remove('visible');
    }

    // Ancienne fonction showMatchResults (gardée pour compatibilité ou reset, mais vidée)
    function showMatchResults(results) {
        // Obsolète avec l'affichage progressif, mais peut servir de structure
    }

    /**
     * Crée un viewer 3D optimisé pour lazy loading
     */
    function createResultViewerOptimized(container, modelUrl, isGLB = false) {
        if (!container) return null;

        const width = container.clientWidth || 150;
        const height = container.clientHeight || 120;

        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0xffffff);

        const camera = new THREE.PerspectiveCamera(60, width / height, 0.01, 1000);
        camera.position.set(0, 0, 0.3);

        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(width, height);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        container.appendChild(renderer.domElement);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.autoRotate = true;
        controls.autoRotateSpeed = 2;

        scene.add(new THREE.AmbientLight(0xffffff, 0.8));
        const light = new THREE.DirectionalLight(0xffffff, 1);
        light.position.set(1, 1, 1);
        scene.add(light);

        if (isGLB) {
            const loader = new THREE.GLTFLoader();
            loader.load(modelUrl, function (gltf) {
                const model = gltf.scene;
                const box = new THREE.Box3().setFromObject(model);
                const center = box.getCenter(new THREE.Vector3());
                model.position.sub(center);
                const size = box.getSize(new THREE.Vector3());
                const maxDim = Math.max(size.x, size.y, size.z);
                scene.add(model);
                camera.position.set(maxDim * 1.5, maxDim * 1.5, maxDim * 1.5);
                camera.lookAt(0, 0, 0);
                controls.update();
            });
        } else {
            const loader = new THREE.PLYLoader();
            loader.load(modelUrl, function (geometry) {
                geometry.computeBoundingBox();
                const center = new THREE.Vector3();
                geometry.boundingBox.getCenter(center);
                geometry.translate(-center.x, -center.y, -center.z);
                const size = new THREE.Vector3();
                geometry.boundingBox.getSize(size);
                const maxDim = Math.max(size.x, size.y, size.z);
                const material = geometry.hasAttribute('color')
                    ? new THREE.PointsMaterial({ size: 0.003, vertexColors: true })
                    : new THREE.PointsMaterial({ size: 0.003, color: 0x6610f2 });
                const points = new THREE.Points(geometry, material);
                scene.add(points);
                camera.position.set(0, 0, maxDim * 2);
                controls.update();
            });
        }

        function animate() {
            const animId = requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
            return animId;
        }
        const animId = animate();

        return { scene, camera, renderer, controls, animId };
    }

    /**
     * Nettoie les viewers de résultats
     */
    function cleanupResultViewers() {
        resultViewers.forEach(viewer => {
            if (viewer.animId) cancelAnimationFrame(viewer.animId);
            if (viewer.renderer) viewer.renderer.dispose();
        });
        resultViewers = [];
        activeViewers.clear();
    }

    /**
     * Initialise l'Intersection Observer pour le lazy loading
     */
    function initViewerObserver() {
        if (viewerObserver) return;

        viewerObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                const viewerId = entry.target.dataset.viewerId;
                if (!viewerId) return;

                if (entry.isIntersecting) {
                    // Viewer visible: créer ou reprendre l'animation
                    const viewerData = activeViewers.get(viewerId);
                    if (viewerData && viewerData.isPaused) {
                        resumeViewer(viewerId);
                    } else if (!viewerData) {
                        // Créer le viewer lazy
                        createLazyViewer(entry.target);
                    }
                } else {
                    // Viewer invisible: mettre en pause
                    pauseViewer(viewerId);
                }
            });
        }, {
            threshold: 0.1, // 10% visible
            rootMargin: '50px' // Charger un peu avant d'être visible
        });
    }

    /**
     * Met en pause un viewer
     */
    function pauseViewer(viewerId) {
        const viewerData = activeViewers.get(viewerId);
        if (!viewerData || viewerData.isPaused) return;

        if (viewerData.animationId) {
            cancelAnimationFrame(viewerData.animationId);
            viewerData.animationId = null;
        }
        viewerData.isPaused = true;
    }

    /**
     * Reprend l'animation d'un viewer
     */
    function resumeViewer(viewerId) {
        const viewerData = activeViewers.get(viewerId);
        if (!viewerData || !viewerData.isPaused) return;

        const { viewer } = viewerData;

        function animate() {
            const animId = requestAnimationFrame(animate);
            viewerData.animationId = animId;
            if (viewer.controls) viewer.controls.update();
            if (viewer.renderer && viewer.scene && viewer.camera) {
                viewer.renderer.render(viewer.scene, viewer.camera);
            }
        }

        viewerData.isPaused = false;
        animate();
    }

    /**
     * Met en pause tous les viewers (pour le modal)
     */
    function pauseAllViewers() {
        activeViewers.forEach((viewerData, viewerId) => {
            pauseViewer(viewerId);
        });
    }

    /**
     * Reprend tous les viewers visibles
     */
    function resumeAllViewers() {
        // Ne reprendre que les viewers visibles dans le viewport
        activeViewers.forEach((viewerData, viewerId) => {
            const container = document.querySelector(`[data-viewer-id="${viewerId}"]`);
            if (container) {
                const rect = container.getBoundingClientRect();
                const isVisible = rect.top < window.innerHeight && rect.bottom > 0;
                if (isVisible && viewerData.isPaused) {
                    resumeViewer(viewerId);
                }
            }
        });
    }

    /**
     * Crée un viewer lazy à partir d'un container
     */
    function createLazyViewer(container) {
        const viewerId = container.dataset.viewerId;
        const modelUrl = container.dataset.modelUrl;
        const isGLB = container.dataset.isGlb === 'true';

        if (!modelUrl || activeViewers.has(viewerId)) return;

        const viewer = createResultViewerOptimized(container, modelUrl, isGLB);
        if (viewer) {
            activeViewers.set(viewerId, {
                viewer: viewer,
                animationId: viewer.animId,
                isPaused: false
            });
        }
    }

    /**
     * Gère la sélection d'un mur
     */
    function selectWall(wallGroup) {
        if (selectedWall) {
            selectedWall.classList.remove('selected');
        }

        if (selectedWall === wallGroup) {
            selectedWall = null;
            hideDropzone();
            return;
        }

        selectedWall = wallGroup;
        selectedWall.classList.add('selected');
        showDropzone(wallGroup);
    }

    /**
     * Affiche la zone de drop
     */
    function showDropzone(wallGroup) {
        const wallName = wallGroup.dataset.name || 'Mur sélectionné';
        wallNameSpan.textContent = wallName;
        dropzoneContainer.classList.add('visible');

        // Initialiser le viewer si pas encore fait
        if (!wallRenderer) {
            setTimeout(initWallViewer, 100);
        }

        setTimeout(() => {
            dropzoneContainer.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }, 100);
    }

    /**
     * Masque la zone de drop
     */
    function hideDropzone() {
        dropzoneContainer.classList.remove('visible');
        dropzoneContainer.classList.remove('has-preview');
        if (selectedWall) {
            selectedWall.classList.remove('selected');
            selectedWall = null;
        }
        resetDropzone();
    }

    /**
     * Réinitialise la dropzone
     */
    function resetDropzone() {
        previewSection.classList.remove('visible');
        dropzone.style.display = '';
        dropzoneContainer.classList.remove('has-preview');
        currentFile = null;
        fileInput.value = '';
        analyzeBtn.disabled = true;
        loadingInline.classList.remove('visible');
        isolatedHolds = [];
        holdMatchResults = {};
        updateHoldCountBadge();

        // Nettoyer le point cloud
        if (wallPointCloud && wallScene) {
            wallScene.remove(wallPointCloud);
            wallPointCloud.geometry.dispose();
            wallPointCloud.material.dispose();
            wallPointCloud = null;
        }

        // Reset le compteur de points originaux
        originalWallPointCount = 0;

        // Nettoyer la session
        if (currentSessionId) {
            fetch(`${API_URL}/api/clear_session`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: currentSessionId })
            }).catch(console.error);
            currentSessionId = null;
        }

        deactivateDBSCANMode();
    }

    /**
     * Formate la taille d'un fichier
     */
    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    /**
     * Convertit un fichier GLB en PLY et le charge
     */
    async function convertAndLoadGLB(file) {
        console.log('🔄 Conversion GLB vers PLY:', file.name);

        const formData = new FormData();
        formData.append('file', file);

        try {
            showLoading('Conversion du GLB en PLY avec couleurs...');

            const response = await fetch(`${API_URL}/api/convert_glb`, {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!data.success) {
                throw new Error(data.error || 'Erreur de conversion');
            }

            hideLoading();
            console.log('✅ GLB converti:', data.total_points, 'points');
            console.log('   Plan détecté:', data.wall_points, 'points sur le mur');
            console.log('   Message:', data.message);

            // Sauvegarder la session (comme dans loadWallPLY)
            currentSessionId = data.session_id;
            wallCenter = data.center;
            isolatedHolds = [];
            holdMatchResults = {};

            // Sauvegarder l'URL du GLB pour le modal
            currentGLBUrl = URL.createObjectURL(file);

            // Afficher le point cloud converti
            displayWallPointCloud(data);

            // Afficher les instructions
            if (selectionInstructions) {
                selectionInstructions.style.display = 'block';
            }

            // Forcer le resize
            setTimeout(resizeWallViewer, 100);

        } catch (error) {
            hideLoading();
            console.error('❌ Erreur conversion:', error);
            alert('Erreur lors de la conversion du GLB.\\n' + error.message);
        }
    }

    /**
     * Gère le fichier uploadé
     */
    function handleFile(file) {
        if (!file) return;

        const fileExtension = '.' + file.name.split('.').pop().toLowerCase();

        if (fileExtension !== '.ply' && fileExtension !== '.glb') {
            alert('Format de fichier non supporté. Seuls les fichiers .ply et .glb sont acceptés.');
            return;
        }

        currentFile = file;
        fileName.textContent = file.name;
        fileSize.textContent = formatFileSize(file.size);

        // Afficher la section preview
        dropzone.style.display = 'none';
        previewSection.classList.add('visible');
        dropzoneContainer.classList.add('has-preview');

        // Si c'est un GLB, le convertir d'abord en PLY
        if (fileExtension === '.glb') {
            convertAndLoadGLB(file);
        } else {
            // Charger le mur directement
            loadWallPLY(file);
        }
    }

    // Éléments de progression
    const progressContainer = document.getElementById('progressContainer');
    const progressBar = document.getElementById('progressBar');
    const progressPercent = document.getElementById('progressPercent');
    const progressDetail = document.getElementById('progressDetail');

    /**
     * Affiche le chargement
     */
    function showLoading(text = 'Chargement...') {
        loadingText.textContent = text;
        loadingInline.classList.add('visible');
        analyzeBtn.classList.add('loading');
    }

    /**
     * Cache le chargement
     */
    function hideLoading() {
        loadingInline.classList.remove('visible');
        analyzeBtn.classList.remove('loading');
        if (progressContainer) progressContainer.classList.remove('visible');
    }

    /**
     * Réinitialise pour une nouvelle analyse
     */
    function resetForNewAnalysis() {
        resultsContainer.classList.remove('visible');
        noResults.classList.remove('visible');
        cleanupResultViewers();
        // Ne pas resetDropzone ici, sinon on perd le mur
        // resetDropzone(); 

        // Juste vider les résultats, garder les prises isolées
        holdMatchResults = {};
    }

    /**
     * Active le mode DBSCAN (calcul des clusters)
     */
    async function activateDBSCANMode() {
        if (!currentSessionId) {
            alert("Veuillez d'abord charger un mur.");
            return;
        }

        try {
            showLoading('Activation Mode DBSCAN...');

            const response = await fetch(`${API_URL}/api/auto_isolate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: currentSessionId })
            });

            const data = await response.json();
            hideLoading();

            if (!data.success) {
                alert(data.message || 'Erreur inconnue');
                return;
            }

            if (data.count === 0) {
                alert("Aucun cluster détecté avec les paramètres actuels.");
                return;
            }

            // Activation du mode
            isolationMode = 'dbscan';

            // === VISUALISATION SANS MUR ===
            // Créer un nuage temporaire avec SEULEMENT les candidats
            if (data.candidate_indices && wallPointCloud) {
                const candidates = data.candidate_indices;
                dbscanCandidateIndices = candidates; // Stocker map

                // Extraire les positions/couleurs du mur original
                const originalPos = wallPointCloud.geometry.attributes.position.array;
                const originalCol = wallPointCloud.userData.originalColors || wallPointCloud.geometry.attributes.color.array;

                const newPos = new Float32Array(candidates.length * 3);
                const newCol = new Float32Array(candidates.length * 3);

                for (let i = 0; i < candidates.length; i++) {
                    const idx = candidates[i];

                    newPos[i * 3] = originalPos[idx * 3];
                    newPos[i * 3 + 1] = originalPos[idx * 3 + 1];
                    newPos[i * 3 + 2] = originalPos[idx * 3 + 2];

                    newCol[i * 3] = originalCol[idx * 3];
                    newCol[i * 3 + 1] = originalCol[idx * 3 + 1];
                    newCol[i * 3 + 2] = originalCol[idx * 3 + 2];
                }

                const geo = new THREE.BufferGeometry();
                geo.setAttribute('position', new THREE.BufferAttribute(newPos, 3));
                geo.setAttribute('color', new THREE.BufferAttribute(newCol, 3));
                geo.computeBoundingBox();

                const mat = new THREE.PointsMaterial({
                    size: wallPointCloud.material.size,
                    vertexColors: true,
                    sizeAttenuation: true
                });

                dbscanPointCloud = new THREE.Points(geo, mat);
                wallScene.add(dbscanPointCloud);

                // Cacher le mur original
                wallPointCloud.visible = false;
            }

            // Feedback visuel sur le bouton
            if (autoDetectBtn) {
                autoDetectBtn.style.background = "#d1fae5";
                autoDetectBtn.style.color = "#047857";
                autoDetectBtn.innerHTML = "<span>✅</span> Mode DBSCAN Actif";
            }

            alert(`${data.count} clusters identifiés.\nLe mur a été masqué pour faciliter la sélection.`);

        } catch (error) {
            hideLoading();
            console.error(error);
            alert('Erreur: ' + error.message);
        }
    }

    function deactivateDBSCANMode() {
        isolationMode = 'manual';

        // Restaurer bouton
        if (autoDetectBtn) {
            autoDetectBtn.style.background = "#e0e7ff";
            autoDetectBtn.style.color = "#4338ca";
            autoDetectBtn.innerHTML = "<span>✨</span> Mode Assistant (DBSCAN)";
        }

        // Supprimer nuage DBSCAN
        if (dbscanPointCloud) {
            wallScene.remove(dbscanPointCloud);
            dbscanPointCloud.geometry.dispose();
            dbscanPointCloud.material.dispose();
            dbscanPointCloud = null;
        }
        dbscanCandidateIndices = null;

        // Réafficher mur
        if (wallPointCloud) {
            wallPointCloud.visible = true;
        }
    }

    /**
     * Efface toutes les sélections
     */
    function clearIsolatedHolds() {
        if (!wallPointCloud || isolatedHolds.length === 0) return;

        if (!confirm("Voulez-vous vraiment effacer toutes les sélections ?")) return;

        // Vider la liste
        isolatedHolds = [];
        updateHoldCountBadge();

        // Tronquer les points ajoutés et restaurer les couleurs originales
        if (wallPointCloud.userData.originalColors && originalWallPointCount > 0) {
            const positions = wallPointCloud.geometry.attributes.position.array;
            const colors = wallPointCloud.userData.originalColors;
            
            // Si des points ont été ajoutés, recréer le geometry original
            const currentCount = positions.length / 3;
            if (currentCount > originalWallPointCount) {
                console.log(`Restoring wall: ${currentCount} -> ${originalWallPointCount} points`);
                
                const newPos = new Float32Array(originalWallPointCount * 3);
                const newCol = new Float32Array(originalWallPointCount * 3);
                const newOriginal = new Float32Array(originalWallPointCount * 3);
                
                for (let i = 0; i < originalWallPointCount * 3; i++) {
                    newPos[i] = positions[i];
                    newCol[i] = colors[i];
                    newOriginal[i] = colors[i];
                }
                
                const newGeometry = new THREE.BufferGeometry();
                newGeometry.setAttribute('position', new THREE.BufferAttribute(newPos, 3));
                newGeometry.setAttribute('color', new THREE.BufferAttribute(newCol, 3));
                
                wallPointCloud.geometry.dispose();
                wallPointCloud.geometry = newGeometry;
                wallPointCloud.userData.originalColors = newOriginal;
            } else {
                // Juste restaurer les couleurs
                const colArray = wallPointCloud.geometry.attributes.color.array;
                for (let i = 0; i < colors.length; i++) {
                    colArray[i] = colors[i];
                }
                wallPointCloud.geometry.attributes.color.needsUpdate = true;
            }
        }

        // Reset mode
        deactivateDBSCANMode();
    }

    // ===== Event Listeners =====

    wallGroups.forEach(wallGroup => {
        wallGroup.addEventListener('click', () => selectWall(wallGroup));
    });

    closeDropzoneBtn.addEventListener('click', hideDropzone);

    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('drag-over');
    });

    dropzone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        dropzone.classList.remove('drag-over');
    });

    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('drag-over');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    removeFileBtn.addEventListener('click', () => {
        resetDropzone();
        dropzone.style.display = '';
    });

    // Bouton d'analyse = lancer le matching
    analyzeBtn.addEventListener('click', analyzeHolds);
    newAnalysisBtn.addEventListener('click', resetForNewAnalysis);

    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            if (resultsContainer.classList.contains('visible')) {
                resetForNewAnalysis();
            } else if (dropzoneContainer.classList.contains('visible')) {
                hideDropzone();
            }
        }
    });

    // ===== Comparison Modal =====
    const comparisonModal = document.getElementById('comparisonModal');
    const modalCloseBtn = document.getElementById('modalCloseBtn');
    const prevMatchBtn = document.getElementById('prevMatchBtn');
    const nextMatchBtn = document.getElementById('nextMatchBtn');

    let currentModalHoldIndex = 0;
    let currentMatchIndex = 0;
    let modalViewers = { zone: null, wall: null, match: null };
    let syncedControls = [];

    window.openComparisonModal = function (holdIndex) {
        currentModalHoldIndex = holdIndex;
        currentMatchIndex = 0;

        const hold = isolatedHolds[holdIndex];
        if (!hold || !hold.matches || hold.matches.length === 0) {
            alert("Aucun résultat disponible pour cette prise.");
            return;
        }

        // Update header
        document.getElementById('modalHoldNumber').textContent = holdIndex + 1;

        // Update color indicator
        const colorIndicator = document.getElementById('modalColorIndicator');
        if (colorIndicator && hold.color !== undefined) {
            // hold.color est un nombre hexadécimal (ex: 0xff0000)
            const r = (hold.color >> 16) & 0xFF;
            const g = (hold.color >> 8) & 0xFF;
            const b = hold.color & 0xFF;
            const hexColor = `rgb(${r}, ${g}, ${b})`;
            colorIndicator.style.backgroundColor = hexColor;
        }

        // Show modal
        comparisonModal.classList.add('visible');

        // Pause tous les viewers en arrière-plan
        pauseAllViewers();

        // Initialize viewers
        setTimeout(() => {
            cleanupModalViewers();
            initializeModalViewers(holdIndex);
        }, 100);
    };

    function cleanupModalViewers() {
        Object.values(modalViewers).forEach(viewer => {
            if (viewer && viewer.renderer) {
                viewer.renderer.dispose();
                viewer.renderer.forceContextLoss();
            }
        });
        modalViewers = { zone: null, wall: null, match: null };
        syncedControls = [];
    }

    function initializeModalViewers(holdIndex) {
        const hold = isolatedHolds[holdIndex];

        // 1. Wall PLY with colored holds (clone from main viewer)
        if (wallPointCloud) {
            modalViewers.zone = cloneWallToModal('modalZonePLY');
        }

        // 2. Wall GLB (complete wall)
        if (currentGLBUrl) {
            modalViewers.wall = createModalViewer('modalWallGLB', currentGLBUrl, true);
        }

        // 3. Match GLB (first match)
        loadMatchInModal(0);

        // Note: Synchronisation désactivée pour éviter les bugs de mouvement
        // Chaque viewer peut maintenant être manipulé indépendamment
    }

    function cloneWallToModal(containerId) {
        const container = document.getElementById(containerId);
        if (!container || !wallPointCloud) return null;

        // Clear container
        container.innerHTML = '';

        const width = container.clientWidth;
        const height = container.clientHeight;

        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0xf5f5f5);

        const camera = new THREE.PerspectiveCamera(60, width / height, 0.01, 1000);

        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(width, height);
        container.appendChild(renderer.domElement);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;

        // Lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        scene.add(ambientLight);
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(1, 1, 1);
        scene.add(directionalLight);

        // Clone the wall point cloud
        const clonedGeometry = wallPointCloud.geometry.clone();
        const clonedMaterial = wallPointCloud.material.clone();
        const clonedMesh = new THREE.Points(clonedGeometry, clonedMaterial);
        scene.add(clonedMesh);

        // Set camera position
        clonedGeometry.computeBoundingBox();
        const center = clonedGeometry.boundingBox.getCenter(new THREE.Vector3());
        const size = clonedGeometry.boundingBox.getSize(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z);
        camera.position.set(center.x, center.y, center.z + maxDim * 1.5);
        controls.target.copy(center);
        controls.update();

        function animate() {
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }
        animate();

        return { scene, camera, renderer, controls };
    }

    function createModalViewer(containerId, modelUrl, isGLB) {
        const container = document.getElementById(containerId);
        if (!container) return null;

        // Clear container
        container.innerHTML = '';

        const width = container.clientWidth;
        const height = container.clientHeight;

        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0xf5f5f5);

        const camera = new THREE.PerspectiveCamera(60, width / height, 0.01, 1000);
        camera.position.set(0, 0, 0.3);

        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(width, height);
        container.appendChild(renderer.domElement);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;

        // Lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        scene.add(ambientLight);
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(1, 1, 1);
        scene.add(directionalLight);

        // Load model
        if (isGLB) {
            const loader = new THREE.GLTFLoader();
            loader.load(modelUrl, (gltf) => {
                scene.add(gltf.scene);
                const box = new THREE.Box3().setFromObject(gltf.scene);
                const center = box.getCenter(new THREE.Vector3());
                const size = box.getSize(new THREE.Vector3());
                const maxDim = Math.max(size.x, size.y, size.z);
                camera.position.set(center.x, center.y, center.z + maxDim * 1.5);
                controls.target.copy(center);
                controls.update();
            });
        } else {
            const loader = new THREE.PLYLoader();
            loader.load(modelUrl, (geometry) => {
                geometry.computeVertexNormals();
                const material = new THREE.PointsMaterial({ size: 0.002, vertexColors: true });
                const mesh = new THREE.Points(geometry, material);
                scene.add(mesh);
                geometry.computeBoundingBox();
                const center = geometry.boundingBox.getCenter(new THREE.Vector3());
                const size = geometry.boundingBox.getSize(new THREE.Vector3());
                const maxDim = Math.max(size.x, size.y, size.z);
                camera.position.set(center.x, center.y, center.z + maxDim * 1.5);
                controls.target.copy(center);
                controls.update();
            });
        }

        function animate() {
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }
        animate();

        return { scene, camera, renderer, controls };
    }

    function setupSynchronizedControls() {
        // Include all three viewers in synchronization
        syncedControls = [
            modalViewers.zone?.controls,
            modalViewers.wall?.controls,
            modalViewers.match?.controls
        ].filter(c => c);

        syncedControls.forEach((controls, index) => {
            if (controls) {
                controls.addEventListener('change', () => {
                    // Get the rotation from the current control
                    const rotation = controls.object.rotation.clone();

                    // Apply to all other controls
                    syncedControls.forEach((otherControls, otherIndex) => {
                        if (otherControls && otherIndex !== index) {
                            otherControls.object.rotation.copy(rotation);
                            otherControls.update();
                        }
                    });
                });
            }
        });
    }

    function loadMatchInModal(matchIndex) {
        const hold = isolatedHolds[currentModalHoldIndex];
        const matches = hold.matches || [];

        if (matchIndex < 0 || matchIndex >= matches.length) return;

        currentMatchIndex = matchIndex;
        const match = matches[matchIndex];

        // Update UI
        document.getElementById('modalMatchRank').textContent = matchIndex + 1;
        document.getElementById('matchInfo').textContent = match.name;

        // Update navigation buttons
        prevMatchBtn.disabled = matchIndex === 0;
        nextMatchBtn.disabled = matchIndex === matches.length - 1;

        // Load match GLB
        if (modalViewers.match && modalViewers.match.renderer) {
            modalViewers.match.renderer.dispose();
        }

        const matchUrl = match.glb_url ? `${API_URL}${match.glb_url}` : (match.ply_url ? `${API_URL}${match.ply_url}` : null);
        if (matchUrl) {
            modalViewers.match = createModalViewer('modalMatchGLB', matchUrl, match.glb_url ? true : false);
        }
    }

    modalCloseBtn.addEventListener('click', () => {
        comparisonModal.classList.remove('visible');
        cleanupModalViewers();

        // Reprendre les viewers visibles en arrière-plan
        resumeAllViewers();
    });

    prevMatchBtn.addEventListener('click', () => {
        if (currentMatchIndex > 0) {
            loadMatchInModal(currentMatchIndex - 1);
        }
    });

    nextMatchBtn.addEventListener('click', () => {
        const hold = isolatedHolds[currentModalHoldIndex];
        if (hold && hold.matches && currentMatchIndex < hold.matches.length - 1) {
            loadMatchInModal(currentMatchIndex + 1);
        }
    });

    // ===== About Modal =====
    const aboutBtn = document.getElementById('aboutBtn');
    const aboutModal = document.getElementById('aboutModal');
    const aboutCloseBtn = document.getElementById('aboutCloseBtn');

    if (aboutBtn && aboutModal && aboutCloseBtn) {
        aboutBtn.addEventListener('click', () => {
            aboutModal.classList.add('visible');
        });

        aboutCloseBtn.addEventListener('click', () => {
            aboutModal.classList.remove('visible');
        });

        // Fermer en cliquant en dehors
        aboutModal.addEventListener('click', (e) => {
            if (e.target === aboutModal) {
                aboutModal.classList.remove('visible');
            }
        });
    }

    // Initialiser l'Intersection Observer pour le lazy loading
    initViewerObserver();

    console.log('HoldGen - Mode Mur avec Sélection de Prises initialisé');
    console.log(`API configurée sur: ${API_URL}`);
});