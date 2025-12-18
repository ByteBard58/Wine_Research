# 🍷 Wine Quality Classifier

A machine learning–powered web application that assesses the quality of wine based on its physicochemical properties.
Built with **Flask**, **Scikit-Learn**, and **Pandas**, this project blends oenology with machine learning to create an interactive tool for wine quality analysis.
This project demonstrates how data science can be applied to the food and beverage industry to standardize quality assessment.

___

## 🚀 Overview

The **Wine Quality Classifier** uses a trained **Machine Learning Model** (optimized via **GridSearchCV**) to predict whether a given wine sample is of **Low**, **Average**, or **High** quality.
The model is hosted via a **Flask** web app where users can input parameters (like acidity, sugar, and alcohol content) and get instant quality assessments.

---

## 🧠 Motivation

Wine tasting is traditionally a subjective art, reliant on the refined palates of sommeliers. However, the chemical composition of wine plays a decisive role in its quality.
This project aims to bridge the gap between chemistry and sensory experience, providing an objective, data-driven approach to classifying wine quality.

---

## 📊 Dataset

- **Source:** [Wine Quality Dataset (WineQT)](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset)
- **Classes:**
  - `Low → 0` (Quality 3, 4)
  - `Average → 1` (Quality 5, 6)
  - `High → 2` (Quality 7, 8)
- **Features:**
  - Fixed Acidity, Volatile Acidity, Citric Acid
  - Residual Sugar, Chlorides
  - Free & Total Sulfur Dioxide
  - Density, pH, Sulphates, Alcohol

---

## ⚙️ Model Architecture

The model generation pipeline (`fit.py`) includes the following steps:

| Step | Description |
| :--- | :--- |
| **Imputation** | Missing values handled using median strategy |
| **Scaling** | Standardized with `StandardScaler` |
| **Augmentation** | SMOTE (Synthetic Minority Over-sampling Technique) for class balance |
| **Dimensionality** | PCA (Principal Component Analysis) for feature reduction |
| **Classifier** | Optimized Estimator (Random Forest / XGBoost / KNN) found via GridSearchCV |

Final model artifacts are serialized with `joblib` as:
```
models/
├── model.pkl
└── columns.pkl
```

---

## 🧩 Project Structure
```
Wine_Research/
├── Datasets/
│   └── WineQT.csv           # Primary dataset
├── models/
│   ├── columns.pkl          # List of feature names
│   └── model.pkl            # Serialized trained model
├── reports/
│   └── research.html        # Static HTML report of the research notebook
├── static/                  # Static assets
│   ├── script.js            # Frontend interaction logic
│   └── style.css            # Premium styling
├── templates/
│   └── index.html           # Main web interface
|
├── .gitignore               # gitignore file
├── app.py                   # Flask application entry point
├── fit.py                   # Machine learning pipeline script
├── LICENSE                  # Licensing information
├── research.py              # Research and analysis script
└── requirements.txt         # Dependencies
```

---

## 💻 Installation & Usage

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/ByteBard58/Wine_Research.git
cd Wine_Research
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the App
```bash
python app.py
```
The app will start at `http://127.0.0.1:5000`.

---

## 🍷 Web Interface

Users can input **physicochemical parameters** such as:

- Acidity levels (Fixed, Volatile, Citric)
- Sugar & Salt (Residual Sugar, Chlorides)
- Sulfur Dioxide levels
- Density & pH
- Alcohol content

The app returns the predicted quality tier (**Low**, **Average**, **High**) with a corresponding visual indicator.

**Note:** Due to the absence of front-end web development expertise and the lack of other contributors for the project, I had to resort to using AI tools (such as LLM services like ChatGPT™, Grok™, and GitHub Copilot™) to create a sophisticated front-end for the web app.

---

## 🧰 Tech Stack

- **Languages**: Python, HTML, CSS, JavaScript
- **Libraries**: Flask, Scikit-Learn, Pandas, NumPy, Joblib, XGBoost, Imbalanced-Learn
- **Dataset Source**: WineQT (Kaggle)

---

## 🪐 Author

Sakib ( ByteBard58 )

> Student | Aspiring Computer Engineer | AI & ML Enthusiast

📍 GitHub Profile: [ByteBard58](http://www.github.com/ByteBard58)

---

## 😃 Appreciation

Thank You for taking the time to review my work. I hope you enjoyed it and found it interesting. It would mean a lot to me if you could star it on GitHub 🌟

If you have any questions, suggestions, or anything you’d like to discuss, please don’t hesitate to reach out. You can find my contact information on my [GitHub profile page](http://www.github.com/ByteBard58). I’m all ears! 😊

Have a great day !
