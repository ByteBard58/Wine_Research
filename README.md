## 🍷 Wine Quality Prediction

### 📘 Overview
This project focuses on predicting the **quality of wines** based on their physicochemical attributes. The dataset comes from [Kaggle’s Wine Quality Dataset](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset), containing various chemical properties like acidity, sugar content, pH, and alcohol percentage.  
The goal is to classify wines into different quality categories using supervised machine learning models.

---

### ⚙️ Workflow Summary
The project follows a complete end-to-end machine learning pipeline:

1. **Data Preprocessing**
   - Missing value imputation  
   - Feature scaling (StandardScaler)  
   - Principle Component Analysis (PCA)
   - Pipelines for consistent preprocessing  

2. **Model Training & Evaluation**
   Tested and compared multiple ML algorithms:
   | Model | Key Parameters | Cross-Validation Accuracy |
   |:------|:---------------|:--------------------------|
   | **K-Nearest Neighbors** | Euclidean distance | **≈ 0.8390** |
   | **Support Vector Classifier** | RBF kernel | **≈ 0.8110** |
   | **Random Forest Classifier** | 1000 trees | **≈ 0.8731** |
   | **Logistic Regression** | lbfgs solver, L2 penalty | **≈ 0.5616** |

   Random Forest achieved the best overall performance and was chosen as the final model.

3. **Model Deployment (CLI Tool)**
   A simple command-line tool allows users to **input wine properties manually** and get a **predicted wine quality** instantly.  
   The tool loads the trained Random Forest model (saved via `joblib`) and performs preprocessing before prediction.

---

### 🧠 Tech Stack
- **Python 3.13.7**  
- **Libraries:** `pandas`, `numpy`, `scikit-learn`, `joblib`, `kagglehub`  

---

### 🚀 How to Run
```bash
# Clone the repo
git clone https://github.com/ByteBard58/Wine_Research
cd Wine_Research

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook research.ipynb

# OR run the CLI predictor
python wine_cli.py
```

### 📊 Results Summary

- **Best Model**: RandomForestClassifier (1000 trees)

- **Cross-Validation Accuracy**: ~0.8696

- **Test Accuracy**: ~0.86

Model performance was limited by class imbalance and overlapping feature distributions, which are inherent to the dataset.

### 💡 Future Improvements

- Handle class imbalance with SMOTE or weighted models

- Tune hyperparameters using GridSearchCV

- Experiment with gradient boosting (XGBoost, LightGBM)

- Build a lightweight Streamlit app version

### 📁 Project Structure
The repository layout as it appears in this workspace:

```
Wine_Research/
│
├── Datasets/
│ └── WineQT.csv # Original dataset
│
├── models/                     # Serialized ML models
│ ├── wine_all_features.joblib
│ ├── wine_features.joblib
│ └── wine_pipeline.joblib
│
├── research.ipynb            # Main analysis & model comparison notebook
├── wine_fit.py # Script for training & saving models
├── wine_cli.py # CLI tool for user input predictions
│
├── requirements.txt          # Project dependencies
├── README.md                 # Documentation
├── LICENSE                   # License info
└── .gitignore                # Ignored files/folders
```

### 😃 Appreciation
Thank you for giving my work a look. I hope you liked it. If you have any queries, I would love to here those. Check my [GitHub profile](https://github.com/ByteBard58/) for contact info.

**Have a great day!**