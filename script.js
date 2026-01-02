const DB_NAME = 'plant_ai_db';
const DB_VERSION = 1;
const STORE_NAME = 'training_data';

// --- IndexedDB Helpers ---
const openDB = () => {
    return new Promise((resolve, reject) => {
        const request = indexedDB.open(DB_NAME, DB_VERSION);
        request.onerror = (event) => reject("Database error: " + event.target.error);
        request.onupgradeneeded = (event) => {
            const db = event.target.result;
            if (!db.objectStoreNames.contains(STORE_NAME)) {
                db.createObjectStore(STORE_NAME);
            }
        };
        request.onsuccess = (event) => resolve(event.target.result);
    });
};

const saveToDB = async (key, value) => {
    const db = await openDB();
    return new Promise((resolve, reject) => {
        const transaction = db.transaction([STORE_NAME], "readwrite");
        const store = transaction.objectStore(STORE_NAME);
        const request = store.put(value, key);
        request.onsuccess = () => resolve();
        request.onerror = (e) => reject(e);
    });
};

const loadFromDB = async (key) => {
    const db = await openDB();
    return new Promise((resolve, reject) => {
        const transaction = db.transaction([STORE_NAME], "readonly");
        const store = transaction.objectStore(STORE_NAME);
        const request = store.get(key);
        request.onsuccess = () => resolve(request.result);
        request.onerror = (e) => reject(e);
    });
};

const clearDB = async () => {
    const db = await openDB();
    return new Promise((resolve, reject) => {
        const transaction = db.transaction([STORE_NAME], "readwrite");
        const store = transaction.objectStore(STORE_NAME);
        const request = store.clear();
        request.onsuccess = () => resolve();
        request.onerror = (e) => reject(e);
    });
};

// --- Main App Logic ---

async function run() {
    const classifier = knnClassifier.create();
    let net;
    
    const statusElement = document.getElementById('model-status');
    statusElement.innerText = 'Loading Model...';

    // Load the model.
    try {
        net = await mobilenet.load();
        console.log('Successfully loaded model');
        statusElement.innerText = 'Model Loaded. Restoring data...';
        statusElement.style.color = 'var(--primary-color)';
    } catch (e) {
        console.error("Error loading model:", e);
        statusElement.innerText = 'Error loading model. Check console.';
        statusElement.style.color = 'red';
        return;
    }

    // State for UI
    let counts = { 0: 0, 1: 0, 2: 0 };
    let previewImages = { 0: [], 1: [], 2: [] };

    // --- Persistence Functions ---

    const saveState = async () => {
        // 1. Save Classifier Dataset
        const dataset = classifier.getClassifierDataset();
        const datasetObj = {};
        Object.keys(dataset).forEach((key) => {
            const data = dataset[key].dataSync();
            // Convert TypedArray to normal Array for JSON serialization if needed, 
            // but IDB can store TypedArrays directly.
            datasetObj[key] = {
                data: data,
                shape: dataset[key].shape
            };
        });

        // 2. Save Images and Counts
        await saveToDB('classifier_dataset', datasetObj);
        await saveToDB('preview_images', previewImages);
        await saveToDB('counts', counts);
        console.log("Saved state to DB");
    };

    const loadState = async () => {
        try {
            // 1. Load Classifier
            const datasetObj = await loadFromDB('classifier_dataset');
            if (datasetObj) {
                const dataset = {};
                Object.keys(datasetObj).forEach((key) => {
                    const { data, shape } = datasetObj[key];
                    dataset[key] = tf.tensor(data, shape);
                });
                classifier.setClassifierDataset(dataset);
            }

            // 2. Load UI State
            const savedCounts = await loadFromDB('counts');
            const savedPreviews = await loadFromDB('preview_images');

            if (savedCounts) counts = savedCounts;
            if (savedPreviews) previewImages = savedPreviews;

            // Update DOM
            updateUI();
            
            statusElement.innerText = 'Model Ready (Data Restored)';
        } catch (e) {
            console.error("Error loading state", e);
            statusElement.innerText = 'Model Ready (New Session)';
        }
    };

    const updateUI = () => {
        // Update counts
        document.getElementById('count-a').innerText = counts[0];
        document.getElementById('count-b').innerText = counts[1];
        document.getElementById('count-c').innerText = counts[2];

        // Update Previews
        ['preview-a', 'preview-b', 'preview-c'].forEach((id, index) => {
            const container = document.getElementById(id);
            container.innerHTML = ''; // clear
            previewImages[index].forEach(src => {
                const img = document.createElement('img');
                img.src = src;
                container.appendChild(img);
            });
        });
    };

    // --- Helper Functions ---

    const readImage = (file) => {
        return new Promise((resolve) => {
            const reader = new FileReader();
            reader.onload = (e) => {
                const img = new Image();
                img.src = e.target.result;
                img.onload = () => resolve(img);
            };
            reader.readAsDataURL(file);
        });
    };

    const handleTrainUpload = async (event, classId) => {
        const files = event.target.files;
        if (!files.length) return;

        statusElement.innerText = 'Processing images...';

        for (let i = 0; i < files.length; i++) {
            const file = files[i];
            const img = await readImage(file);
            
            // Add to classifier
            const activation = net.infer(img, true);
            classifier.addExample(activation, classId);
            
            // Update State
            counts[classId]++;
            previewImages[classId].push(img.src);
            
            // Cleanup tensor? 
            // Note: In a loop, it's good practice to dispose intermediate tensors if not needed.
            // classifier.addExample keeps what it needs.
            // But we need to make sure we don't leak memory if we run this a lot.
            // `net.infer` returns a tensor. 
            // For now, relying on TFJS automatic management or manual disposal if performance drops.
        }
        
        updateUI();
        await saveState(); // Save after batch
        
        statusElement.innerText = 'Model Updated & Saved!';
        setTimeout(() => {
            statusElement.innerText = 'Ready to train or predict.';
        }, 2000);
        
        event.target.value = '';
    };

    // --- Event Listeners ---

    document.getElementById('file-a').addEventListener('change', (e) => handleTrainUpload(e, 0));
    document.getElementById('file-b').addEventListener('change', (e) => handleTrainUpload(e, 1));
    document.getElementById('file-c').addEventListener('change', (e) => handleTrainUpload(e, 2));

    document.getElementById('clear-data-btn').addEventListener('click', async () => {
        if(confirm("Are you sure you want to delete all training data?")) {
            classifier.clearAllClasses();
            counts = { 0: 0, 1: 0, 2: 0 };
            previewImages = { 0: [], 1: [], 2: [] };
            await clearDB();
            updateUI();
            statusElement.innerText = 'Training data cleared.';
        }
    });

    document.getElementById('file-test').addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        if (classifier.getNumClasses() === 0) {
            alert("Please train the AI with at least one class first!");
            return;
        }

        const resultBox = document.getElementById('prediction-result');
        const testImage = document.getElementById('test-image');
        
        resultBox.innerText = "Analyzing...";
        
        const img = await readImage(file);
        testImage.src = img.src;
        testImage.style.display = 'block';

        try {
            const activation = net.infer(img, 'conv_preds');
            const result = await classifier.predictClass(activation);

            const classes = ['Healthy Plant', 'Yellow Leaf Disease', 'Brown Spot Disease'];
            const label = classes[result.label];
            const probability = (result.confidences[result.label] * 100).toFixed(2);

            resultBox.innerText = `${label} (${probability}%)`;
            resultBox.style.color = result.label === 0 ? 'var(--primary-color)' : 'red';
            
            activation.dispose();
        } catch (error) {
            console.error(error);
            resultBox.innerText = "Error during prediction.";
        }
        
        e.target.value = '';
    });

    // --- Initialize ---
    await loadState();
}

run();
