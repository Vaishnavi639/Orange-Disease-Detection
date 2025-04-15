function predictDisease() {
    const input = document.getElementById('imageInput');
    const file = input.files[0];
    
    if (!file) {
        alert('Please select an image first');
        return;
    }

    // Preview the image
    const reader = new FileReader();
    reader.onload = function(e) {
        const preview = document.getElementById('preview');
        preview.innerHTML = `<img src="${e.target.result}" alt="Preview">`;
    };
    reader.readAsDataURL(file);

    // Prepare form data
    const formData = new FormData();
    formData.append('image', file);

    // Make prediction request
    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        const resultDiv = document.getElementById('result');
        if (data.error) {
            resultDiv.innerHTML = `Error: ${data.error}`;
        } else {
            resultDiv.innerHTML = `
                <h3>Result:</h3>
                <p>Disease: ${data.disease}</p>
                <p>Confidence: ${(data.confidence * 100).toFixed(2)}%</p>
            `;
        }
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('result').innerHTML = 'An error occurred during prediction';
    });
}