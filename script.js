async function run() {
    const classifier = knnClassifier.create();
    let net;
    
    const statusElement = document.getElementById('model-status');
    statusElement.innerText = 'Loading Model...';

    // Load the model.
    try {
        net = await mobilenet.load();
        console.log('Successfully loaded model');
        statusElement.innerText = 'Model Loaded. Ready to train.';
        statusElement.style.color = 'var(--primary-color)';
    } catch (e) {
        console.error("Error loading model:", e);
        statusElement.innerText = 'Error loading model. Check console.';
        statusElement.style.color = 'red';
        return;
    }

    // Counts for UI
    const counts = { 0: 0, 1: 0, 2: 0 };

    // Helper to read file as image element
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

    // Helper to add image to preview
    const addToPreview = (imgSrc, previewId) => {
        const previewContainer = document.getElementById(previewId);
        const img = document.createElement('img');
        img.src = imgSrc;
        previewContainer.appendChild(img);
    };

    // Handle training uploads
    const handleTrainUpload = async (event, classId, countId, previewId) => {
        const files = event.target.files;
        if (!files.length) return;

        statusElement.innerText = 'Processing images...';

        for (let i = 0; i < files.length; i++) {
            const file = files[i];
            const img = await readImage(file);
            
            // Add to classifier
            const activation = net.infer(img, true);
            classifier.addExample(activation, classId);
            
            // Update UI
            counts[classId]++;
            document.getElementById(countId).innerText = counts[classId];
            addToPreview(img.src, previewId);
            
            // Clean up tensor
            // activation.dispose(); // Note: KNN Classifier might need the tensor, but usually it stores what it needs. 
            // Actually, in TFJS KNN, addExample keeps a reference. We should NOT dispose activation immediately if we passed it?
            // Checking docs: "It keeps a reference to the activation..." -> Wait, if we pass a tensor, we usually shouldn't dispose it IF the classifier doesn't make a copy.
            // However, typical usage examples often don't dispose explicitly or rely on tf.tidy. 
            // For safety in this loop, let's rely on garbage collection for the JS objects but we should be careful with GPU memory.
            // `net.infer` returns a tensor. `classifier.addExample` stores it. 
            // So we do NOT dispose `activation` here.
        }
        
        statusElement.innerText = 'Model Updated!';
        setTimeout(() => {
            statusElement.innerText = 'Ready to train or predict.';
        }, 2000);
        
        // Clear input
        event.target.value = '';
    };

    // Event Listeners for Training
    document.getElementById('file-a').addEventListener('change', (e) => handleTrainUpload(e, 0, 'count-a', 'preview-a'));
    document.getElementById('file-b').addEventListener('change', (e) => handleTrainUpload(e, 1, 'count-b', 'preview-b'));
    document.getElementById('file-c').addEventListener('change', (e) => handleTrainUpload(e, 2, 'count-c', 'preview-c'));

    // Handle Prediction
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
        
        // Clear input so same file can be selected again if needed
        e.target.value = '';
    });
}

run();
