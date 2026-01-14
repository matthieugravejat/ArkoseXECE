/**
 * HoldGen - Gestionnaire de Salle d'Escalade
 * Application Frontend avec visualisation 3D Three.js
 * 
 * Mode MUR: Charge un PLY de mur, sélection des prises par clic, matching
 */

// Configuration API
const API_URL = 'http://localhost:5001';

// Variables globales Three.js
let wallScene, wallCamera, wallRenderer, wallControls;
let wallPointCloud = null;
let wallAnimationId = null;
let resultViewers = [];

// État de l'application
let currentSessionId = null;
let wallCenter = [0, 0, 0];
let isolatedHolds = [];  // {holdId, indices, color}
let holdMatchResults = {};  // {holdId: [matches]}
let raycaster = null;
let mouse = null;

// Couleurs pour les prises isolées
const HOLD_COLORS = [
    0xff0000, // Rouge
    0x00ff00, // Vert
    0x0000ff, // Bleu
    0xffff00, // Jaune
    0xff00ff, // Magenta
    0x00ffff, // Cyan
    0xff8000, // Orange
    0x8000ff, // Violet
    0x00ff80, // Vert menthe
    0xff0080, // Rose
];

document.addEventListener('DOMContentLoaded', () => {
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

    // État de l'application
    let selectedWall = null;
    let currentFile = null;

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
        if (!wallPointCloud || !currentSessionId) return;

        const rect = previewCanvas.getBoundingClientRect();
        mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

        raycaster.setFromCamera(mouse, wallCamera);
        const intersects = raycaster.intersectObject(wallPointCloud);

        if (intersects.length > 0) {
            const pointIndex = intersects[0].index;
            console.log('Point cliqué:', pointIndex);

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
                    point_index: pointIndex
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
            isolatedHolds.push({
                holdId: data.hold_id,
                indices: data.indices,
                color: holdColor,
                pointCount: data.point_count,
                matches: null
            });

            // Mettre à jour la visualisation
            highlightIsolatedHold(data.indices, holdColor);
            updateHoldCountBadge();

        } catch (error) {
            hideLoading();
            console.error('Erreur isolation:', error);
            alert('Erreur: ' + error.message);
        }
    }

    /**
     * Met en surbrillance une prise isolée
     */
    function highlightIsolatedHold(indices, color) {
        if (!wallPointCloud) return;

        const colors = wallPointCloud.geometry.attributes.color.array;
        const r = ((color >> 16) & 255) / 255;
        const g = ((color >> 8) & 255) / 255;
        const b = (color & 255) / 255;

        indices.forEach(idx => {
            colors[idx * 3] = r;
            colors[idx * 3 + 1] = g;
            colors[idx * 3 + 2] = b;
        });

        wallPointCloud.geometry.attributes.color.needsUpdate = true;
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
        wallScene.add(wallPointCloud);

        // Positionner la caméra
        const cameraDistance = maxDim * 2;
        wallCamera.position.set(0, 0, cameraDistance);
        wallCamera.lookAt(0, 0, 0);
        wallControls.target.set(0, 0, 0);
        wallControls.update();
    }

    /**
     * Lance le matching pour toutes les prises isolées
     */
    async function matchAllIsolatedHolds() {
        if (isolatedHolds.length === 0) {
            alert('Veuillez d\'abord sélectionner des prises sur le mur');
            return;
        }

        showLoading('Matching des prises...');

        try {
            const response = await fetch(`${API_URL}/api/match_isolated_holds`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: currentSessionId
                })
            });

            const data = await response.json();
            hideLoading();

            if (!data.success) {
                throw new Error(data.error || 'Erreur inconnue');
            }

            console.log('Résultats matching:', data);

            // Stocker les résultats
            data.results.forEach((result, i) => {
                holdMatchResults[result.hold_info.hold_id] = result;
                if (isolatedHolds[i]) {
                    isolatedHolds[i].matches = result.matches;
                    isolatedHolds[i].color_name = result.hold_info.color_name;
                }
            });

            // Afficher les résultats
            showMatchResults(data.results);

        } catch (error) {
            hideLoading();
            console.error('Erreur matching:', error);
            alert('Erreur: ' + error.message);
        }
    }

    /**
     * Affiche les résultats du matching
     */
    function showMatchResults(results) {
        cleanupResultViewers();

        // GARDER le mur visible - ne pas cacher dropzoneContainer
        // dropzoneContainer.classList.remove('visible');

        // Info source (sans emojis)
        sourceInfo.innerHTML = `
            <div class="source-info-item">
                <span class="source-info-label">Prises analysées:</span>
                <span class="source-info-value">${results.length}</span>
            </div>
            <div class="source-info-item">
                <span class="source-info-label">Mur:</span>
                <span class="source-info-value">${selectedWall?.dataset.name || 'Non spécifié'}</span>
            </div>
        `;

        // Créer les cartes de résultats EN LIGNE (horizontal)
        if (results.length > 0) {
            resultsGrid.innerHTML = results.map((result, index) => {
                const hold = isolatedHolds[index];
                const colorHex = hold ? '#' + hold.color.toString(16).padStart(6, '0') : '#ff0000';
                const matches = result.matches || [];

                return `
                    <div class="hold-result-row" data-hold-id="${result.hold_info.hold_id}">
                        <div class="hold-label" style="background: ${colorHex}">
                            <span class="hold-number">Prise ${index + 1}</span>
                            <span class="hold-meta">${result.hold_info.point_count} pts${result.hold_info.color_name ? ' • ' + result.hold_info.color_name : ''}</span>
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
                                            <span class="detail-label">ICP</span>
                                            <span class="detail-value">${match.icp_fitness}%</span>
                                        </div>
                                        <div class="detail-item">
                                            <span class="detail-label">Eigen</span>
                                            <span class="detail-value">${match.eigen_score}%</span>
                                        </div>
                                        <div class="detail-item">
                                            <span class="detail-label">RMSE</span>
                                            <span class="detail-value">${match.rmse_mm}mm</span>
                                        </div>
                                        <div class="detail-item">
                                            <span class="detail-label">Échelle</span>
                                            <span class="detail-value">${match.scale}×</span>
                                        </div>
                                    </div>
                                </div>
                            `).join('') : '<div class="no-match">Aucune correspondance trouvée</div>'}
                        </div>
                    </div>
                `;
            }).join('');

            noResults.classList.remove('visible');

            // Créer les viewers 3D pour chaque match
            setTimeout(() => {
                results.forEach((result, index) => {
                    const matches = result.matches || [];
                    matches.slice(0, 3).forEach((match, mi) => {
                        if (match.glb_url) {
                            const viewer = createResultViewer(`matchViewer${index}_${mi}`, `${API_URL}${match.glb_url}`, true);
                            if (viewer) resultViewers.push(viewer);
                        } else if (match.ply_url) {
                            const viewer = createResultViewer(`matchViewer${index}_${mi}`, `${API_URL}${match.ply_url}`, false);
                            if (viewer) resultViewers.push(viewer);
                        }
                    });
                });
            }, 100);

        } else {
            resultsGrid.innerHTML = '';
            noResults.classList.add('visible');
        }

        resultsContainer.classList.add('visible');
        setTimeout(() => {
            resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 100);
    }

    /**
     * Crée un viewer 3D pour un résultat
     */
    function createResultViewer(containerId, modelUrl, isGLB = false) {
        const container = document.getElementById(containerId);
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

        // Nettoyer la session
        if (currentSessionId) {
            fetch(`${API_URL}/api/clear_session`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: currentSessionId })
            }).catch(console.error);
            currentSessionId = null;
        }
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
     * Gère le fichier uploadé
     */
    function handleFile(file) {
        if (!file) return;

        const fileExtension = '.' + file.name.split('.').pop().toLowerCase();

        if (fileExtension !== '.ply') {
            alert('Format de fichier non supporté. Seuls les fichiers .ply sont acceptés.');
            return;
        }

        currentFile = file;
        fileName.textContent = file.name;
        fileSize.textContent = formatFileSize(file.size);

        // Afficher la section preview
        dropzone.style.display = 'none';
        previewSection.classList.add('visible');
        dropzoneContainer.classList.add('has-preview');

        // Charger le mur
        loadWallPLY(file);
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
        resetDropzone();

        window.scrollTo({ top: 0, behavior: 'smooth' });

        if (selectedWall) {
            selectedWall.classList.remove('selected');
            selectedWall = null;
        }
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
    analyzeBtn.addEventListener('click', matchAllIsolatedHolds);
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

    console.log('HoldGen - Mode Mur avec Sélection de Prises initialisé');
    console.log(`API configurée sur: ${API_URL}`);
});
