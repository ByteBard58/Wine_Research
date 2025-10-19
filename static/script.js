document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("predictForm");
  const inputs = document.querySelectorAll("input[type='number']");
  const predictBtn = document.getElementById("predictBtn");
  const resultDiv = document.getElementById("result");
  const probsDiv = document.getElementById("probabilities");

  // === 1. Enter key navigation ===
  inputs.forEach((input, idx) => {
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        if (idx < inputs.length - 1) {
          inputs[idx + 1].focus();
        } else {
          predictBtn.click();
        }
      }
    });
  });

  // === 2. Predict button click ===
  predictBtn.addEventListener("click", async () => {
    const formData = new FormData(form);
    const values = {};
    formData.forEach((v, k) => (values[k] = v));

    resultDiv.innerHTML = "Predicting...";
    probsDiv.innerHTML = "";

    try {
      const res = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(values),
      });

      const data = await res.json();

      if (data.error) {
        resultDiv.innerHTML = `<span style="color:#ff6666">${data.error}</span>`;
        return;
      }

      resultDiv.innerHTML = `<strong>Prediction:</strong> <span>${data.prediction}</span>`;

      // === 3. Show probabilities with animated bars ===
      probsDiv.innerHTML = "<h3>Class Probabilities:</h3>";

      for (const [cls, prob] of Object.entries(data.probabilities)) {
        const bar = document.createElement("div");
        bar.className = "progress-bar";

        const barInner = document.createElement("div");
        barInner.className = "progress-bar-inner";
        barInner.style.width = `${prob * 100}%`;
        barInner.textContent = `${cls.toUpperCase()}: ${(prob * 100).toFixed(1)}%`;

        bar.appendChild(barInner);
        probsDiv.appendChild(bar);
      }
    } catch (err) {
      console.error(err);
      resultDiv.innerHTML = `<span style="color:#ff6666">Server error. Check console.</span>`;
    }
  });
});
