document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('prediction-form');
    const modal = document.getElementById('result-modal');
    const cloaseBtn = document.querySelector('.close-modal');
    const closeBtnBtn = document.getElementById('close-btn');
    const predictionText = document.getElementById('prediction-text');
    const predictionDesc = document.getElementById('prediction-desc');
    const resultIcon = document.getElementById('result-icon');
    const loader = document.getElementById('loader');

    const closeModal = () => {
        modal.classList.remove('visible');
        setTimeout(() => {
             modal.classList.add('hidden');
        }, 400); // Wait for transition
    };

    cloaseBtn.addEventListener('click', closeModal);
    closeBtnBtn.addEventListener('click', closeModal);
    window.addEventListener('click', (e) => {
        if (e.target === modal) closeModal();
    });

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Show loader
        loader.classList.remove('hidden');

        // Gather data
        const formData = new FormData(form);
        const data = {};
        formData.forEach((value, key) => {
            data[key] = parseFloat(value);
        });

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data),
            });

            const result = await response.json();

            // Hide loader
            loader.classList.add('hidden');

            if (response.ok) {
                showResult(result);
            } else {
                alert('Error: ' + (result.error || 'Something went wrong'));
            }
        } catch (error) {
            loader.classList.add('hidden');
            console.error('Error:', error);
            alert('Failed to connect to the server.');
        }
    });

    function showResult(data) {
        // data.prediction is "Low", "Average", "High"
        predictionText.textContent = data.prediction + " Quality";
        
        let iconHtml = '';
        let desc = '';

        if (data.prediction === 'High') {
            iconHtml = '<i class="fa-solid fa-trophy icon-high"></i>';
            desc = "Exceptional! This wine exhibits superior characteristics.";
        } else if (data.prediction === 'Average') {
            iconHtml = '<i class="fa-solid fa-wine-glass icon-avg"></i>';
            desc = "A balanced wine, perfect for casual dining.";
        } else {
            iconHtml = '<i class="fa-solid fa-thumbs-down icon-low"></i>';
            desc = "Below standard. May have some defects.";
        }

        resultIcon.innerHTML = iconHtml;
        predictionDesc.textContent = desc;
        
        modal.classList.remove('hidden');
        // Small delay to allow display:block to apply before opacity transition
        requestAnimationFrame(() => {
            modal.classList.add('visible');
        });
    }
});
