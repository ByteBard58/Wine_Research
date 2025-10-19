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
      probBars.innerHTML = "";

      const classMap = {
        0: 'Low quality (3-4)',
        1: 'Medium quality (5-6)',
        2: 'High quality (7-8)'
      };

      probabilities.forEach((prob, idx) => {
        const barContainer = document.createElement("div");
        barContainer.className = "bar-container";

        const label = document.createElement("span");
        label.textContent = classMap[idx] || `Class ${idx}`;
        label.className = "bar-label";

        const bar = document.createElement("div");
        bar.className = "bar";
        bar.style.width = `${prob * 100}%`;

        const value = document.createElement("span");
        value.textContent = `${(prob * 100).toFixed(1)}%`;
        value.className = "bar-value";

        barContainer.appendChild(label);
        barContainer.appendChild(bar);
        barContainer.appendChild(value);
        probBars.appendChild(barContainer);
      });
    } catch (err) {
      console.error(err);
      predLabel.textContent = "Error: Could not get prediction.";
    }
  });
});