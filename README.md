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

3. **Model Deployment (CLI Tool & Web App)**
   A simple command-line tool allows users to **input wine properties manually** and get a **predicted wine quality** instantly.  
   The tool loads the trained Random Forest model (saved via `joblib`) and performs preprocessing before prediction.

   In newer commits, a fully functional **Web App** has been added, built with **Flask** and a **Vanilla frontend**.  
   This web app offers a clean UI and robust functionality. Users can input data through the web form and receive predictions with smooth visuals.

---

### 🧠 Tech Stack
- **Python 3.13.7**  
- **Libraries:** `pandas`, `numpy`, `scikit-learn`, `joblib`, `kagglehub`, `Flask`
- **HTML, CSS and JavaScript** for Frontend

---

### 🚀 How to Run
Clone the repo
```bash
git clone https://github.com/ByteBard58/Wine_Research
cd Wine_Research
```
Install dependencies
```bash
pip install -r requirements.txt
```
After that, you can do any of these 3 things:

Run the notebook
```bash
jupyter notebook research.ipynb
```

Run the CLI predictor
```bash
python wine_cli.py
```

Run the Web App
```bash
python app.py
```

---

### 📊 Results Summary

- **Best Model**: RandomForestClassifier (1000 trees)

- **Cross-Validation Accuracy**: ~0.8696

- **Test Accuracy**: ~0.86

Model performance was limited by class imbalance and overlapping feature distributions, which are inherent to the dataset.

### 💡 Future Improvements

- Handle class imbalance with SMOTE or weighted models

- Tune hyperparameters using GridSearchCV

- Experiment with gradient boosting (XGBoost, LightGBM)

  ---

### 📁 Project Structure
The repository layout as it appears in this workspace:
```
Wine_Research/
│
├── Datasets/
│   └── WineQT.csv                 # Original dataset
│
├── models/                        # Serialized ML models
│   ├── wine_all_features.joblib
│   ├── wine_features.joblib
│   └── wine_pipeline.joblib
│
├── static/                        # Frontend static files
│   ├── style.css
│   └── script.js
│
├── templates/                     # HTML templates for Flask
│   └── index.html
│
├── research.ipynb                 # Main analysis & model comparison notebook
├── wine_fit.py                    # Script for training & saving models
├── wine_cli.py                    # CLI tool for user input predictions
│
├── app.py                         # Flask web app entry point
│
├── requirements.txt               # Project dependencies
├── README.md                      # Documentation
├── LICENSE                        # License info
└── .gitignore                     # Ignored files/folders
```

---

### 👨‍💻 About the Web App
The web app is implemented in `app.py` and uses the `.joblib` files to access the ML pipeline. Flask serves as the backend framework, and the site runs on localhost:5000

It offers a user-friendly UI and elegant visuals.

Here are some things you should keep in mind:
- In order to run the Flask app as expected, you need the `.joblib` files which contain the ML pipeline and the list of all features. I have included the files with the repository (in the models subdirectory). But if the files are somehow missing, you can easily get those by running the `wine_fit.py` file. To do that, open terminal and run:
```bash
cd Wine_Research
python wine_fit.py
```
- After running app.py, open your browser and go to [localhost:5000](http://127.0.0.1:5000/)
- Both the CLI tool and web app handle empty inputs gracefully. Empty fields are treated as NaN and imputed by the model’s preprocessor, but predictions may be less accurate — so it’s best to provide all inputs.
- I am unable to code the frontend for the web app since I don't have any expertise with front end. I had to use AI tools (mostly LLMs like ChatGPT™ and Gemini™) in order to pull the front end. So, the files inside the `templates` and `static` subdirectory are completely AI generated (the code).

---

### 😃 Appreciation
Thank you for giving my work a look. I hope you liked it. If you have any queries, I would love to here those. Check my [GitHub profile](https://github.com/ByteBard58/) for contact info.

**Have a great day!**
