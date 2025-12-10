document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("predictForm");
    const predictBtn = document.getElementById("predictBtn");
    const emptyState = document.getElementById("emptyState");
    const predictionContent = document.getElementById("predictionContent");
    
    // UI Elements for result
    const resultHeader = document.getElementById("predictionResult");
    const resultBadge = document.getElementById("predictionBadge");
    const probBarsContainer = document.getElementById("probBars");

    const CLASS_MAP = {
        0: { label: "Low Quality", desc: "Rating 3-4" },
        1: { label: "Medium Quality", desc: "Rating 5-6" },
        2: { label: "High Quality", desc: "Rating 7-8" }
    };

    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        
        // 1. UI Loading State
        const originalBtnContent = predictBtn.innerHTML;
        predictBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Analyzing...';
        predictBtn.disabled = true;

        // 2. Data Gathering
        const formData = new FormData(form);
        const jsonData = {};
        formData.forEach((value, key) => {
            jsonData[key] = parseFloat(value);
        });

        try {
            // 3. API Call
            const response = await fetch("/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(jsonData)
            });

            if (!response.ok) {
                throw new Error("Analysis Failed");
            }

            const data = await response.json();
            
            // 4. Update UI with Results
            updateResults(data);

        } catch (error) {
            console.error(error);
            alert("An error occurred while analyzing the wine data. Please try again.");
        } finally {
            // 5. Reset Button
            predictBtn.innerHTML = originalBtnContent;
            predictBtn.disabled = false;
        }
    });

    function updateResults(data) {
        // Hide empty state, show content
        emptyState.classList.add("hidden");
        predictionContent.classList.remove("hidden");

        const classIdx = data.predicted_class;
        const info = CLASS_MAP[classIdx];

        // Update Verdict Card
        resultHeader.textContent = info.label;
        resultBadge.textContent = info.desc;
        
        // Color coding based on quality
        resultHeader.style.color = classIdx === 2 ? "#d4af37" : (classIdx === 0 ? "#ff4d4d" : "#f5f5f5");

        // Render Probability Bars
        renderBars(data.probabilities);
    }

    function renderBars(probs) {
        probBarsContainer.innerHTML = "";
        
        probs.forEach((prob, idx) => {
            const row = document.createElement("div");
            row.className = "bar-row";
            
            const label = CLASS_MAP[idx].label.split(" ")[0]; // Just "Low", "Medium", "High"
            
            const percent = (prob * 100).toFixed(1);

            row.innerHTML = `
                <span class="bar-label">${label}</span>
                <div class="bar-track">
                    <div class="bar-fill" style="width: 0%"></div>
                </div>
                <span class="bar-percent">${percent}%</span>
            `;
            
            probBarsContainer.appendChild(row);

            // Animate after a brief delay for layout reflow
            setTimeout(() => {
                row.querySelector(".bar-fill").style.width = `${percent}%`;
            }, 50);
        });
    }
});