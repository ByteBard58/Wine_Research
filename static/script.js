document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("predictForm");
  const inputs = form.querySelectorAll("input");
  const resultDiv = document.getElementById("result");
  const predLabel = document.getElementById("pred-label");
  const probBars = document.getElementById("prob-bars");
  const predictBtn = document.getElementById("predictBtn");

  // Handle Enter key navigation
  inputs.forEach((input, index) => {
    input.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        if (index < inputs.length - 1) {
          inputs[index + 1].focus();
        } else {
          predictBtn.click();
        }
      }
    });
  });

  // Handle form submit
  form.addEventListener("submit", async (event) => {
    event.preventDefault();

    const formData = new FormData(form);
    const values = {};
    formData.forEach((v, k) => (values[k] = parseFloat(v)));

    predLabel.textContent = "Predicting...";
    probBars.innerHTML = "";
    resultDiv.style.display = "block";

    // Optional: Add a placeholder while predicting
    probBars.innerHTML = `
        <div class="bar-container"><span class="bar-label">Low quality (3-4)</span><div class="bar-wrapper"><div class="bar" style="width: 100%; opacity: 0.5;"></div></div><span class="bar-value">...</span></div>
        <div class="bar-container"><span class="bar-label">Medium quality (5-6)</span><div class="bar-wrapper"><div class="bar" style="width: 100%; opacity: 0.5;"></div></div><span class="bar-value">...</span></div>
        <div class="bar-container"><span class="bar-label">High quality (7-8)</span><div class="bar-wrapper"><div class="bar" style="width: 100%; opacity: 0.5;"></div></div><span class="bar-value">...</span></div>
    `;

    try {
      const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(values),
      });

      if (!response.ok) throw new Error("Server error");

      const data = await response.json();
      const { predicted_label, probabilities } = data;

      predLabel.textContent = `Predicted Quality: ${predicted_label}`;
      probBars.innerHTML = ""; // Clear the temporary content

      const classMap = {
        0: 'Low quality (3-4)',
        1: 'Medium quality (5-6)',
        2: 'High quality (7-8)'
      };

      // The original `probabilities` array is sorted by index (0, 1, 2)
      probabilities.forEach((prob, idx) => {
        const barContainer = document.createElement("div");
        barContainer.className = "bar-container";

        const label = document.createElement("span");
        label.textContent = classMap[idx] || `Class ${idx}`;
        label.className = "bar-label";

        // --- START FIX: Create wrapper and inner bar ---
        const barWrapper = document.createElement("div");
        barWrapper.className = "bar-wrapper";

        const bar = document.createElement("div");
        bar.className = "bar";
        
        barWrapper.appendChild(bar); 
        // --- END FIX ---

        const value = document.createElement("span");
        value.textContent = `${(prob * 100).toFixed(1)}%`;
        value.className = "bar-value";

        barContainer.appendChild(label);
        barContainer.appendChild(barWrapper); // Append the new wrapper
        barContainer.appendChild(value);
        probBars.appendChild(barContainer);

        // --- START FIX: Use requestAnimationFrame for smooth transition ---
        // The double requestAnimationFrame forces the browser to render the element 
        // with the initial 'width: 0' before applying the final width, 
        // which triggers the CSS transition (animation).
        requestAnimationFrame(() => {
             requestAnimationFrame(() => {
                 bar.style.width = `${prob * 100}%`;
             });
        });
        // --- END FIX ---
      });
    } catch (err) {
      console.error(err);
      predLabel.textContent = "Error: Could not get prediction.";
      probBars.innerHTML = "";
    }
  });
});