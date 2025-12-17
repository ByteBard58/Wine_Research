import numpy as np
import pandas as pd
import time

from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

from xgboost import XGBClassifier


df_raw = pd.read_csv("Datasets/WineQT.csv")
df_1 = df_raw.drop(columns=["Id"],axis=0)

def mapper(x):
    if x in [3,4]:
        return 0
    elif x in [5,6]:
        return 1
    elif x in [7,8]:
        return 2

df_1["label_quality"] = df_1["quality"].apply(mapper)
df_2 = df_1.drop(["quality"],axis=1)

df = df_2.copy()

feat_columns = df.iloc[:,:-1].columns
all_columns = df.columns

x = df.iloc[:,:-1].to_numpy()
y = df.iloc[:,-1].to_numpy()


x_train,x_test,y_train,y_test = train_test_split(
  x,y,test_size=1/4,random_state=32,shuffle=True,stratify=y
)

pipe = Pipeline([
    ("imputation",SimpleImputer(strategy="median")),
    ("scale",StandardScaler()),
    ("smote",SMOTE(random_state=21,k_neighbors=3)),
    ("dimen",PCA(n_components=5)),
    ("model",RandomForestClassifier(class_weight="balanced"))
])

rf = RandomForestClassifier(class_weight="balanced",random_state=32)
xgb = XGBClassifier(random_state=902)
knn = KNN()

param_grid = [

    # =========================
    # RANDOM FOREST
    # =========================
    {
        "model": [rf],
        "model__n_estimators": [1100,1300,1500],
        "model__max_depth": [7,10,None],
        "dimen": ["passthrough"]
    },
    {
        "model": [rf],
        "model__n_estimators": [1100, 1300, 1500],
        "model__max_depth": [7,10,None],
        "dimen": [LDA()]
    },
    {
        "model": [rf],
        "model__n_estimators":[1100,1300,1500],
        "model__max_depth": [7,10,None],
        "dimen": [PCA(random_state=9011)],
        "dimen__n_components": [3, 6, 9,None]
    },
    {
        "model": [rf],
        "model__n_estimators":[1100,1300,1500],
        "model__max_depth": [10,None],
        "dimen": [PCA(random_state=9011)],
        "dimen__n_components": [9,None],
        "smote":["passthrough"]
    },

    # =========================
    # XGBOOST
    # =========================
    {
        "model": [xgb],
        "model__n_estimators": [500, 700, 1000],
        "model__learning_rate": [0.01, 0.1],
        "model__max_depth": [7,10,None],
        "dimen": ["passthrough"]
    },
    {
        "model": [xgb],
        "model__n_estimators": [500, 700, 1000],
        "model__learning_rate": [0.01, 0.1],
        "model__max_depth": [7,10,None],
        "dimen": [LDA()]
    },
    {
        "model": [xgb],
        "model__n_estimators": [500, 700, 1000],
        "model__learning_rate": [0.01, 0.1],
        "model__max_depth": [7,10,None],
        "dimen": [PCA(random_state=401)],
        "dimen__n_components": [6, 9,None]
    },
    {
        "model": [xgb],
        "model__n_estimators": [500, 700, 1000],
        "model__learning_rate": [0.01, 0.1],
        "model__max_depth": [10,None],
        "dimen": [PCA(random_state=401)],
        "dimen__n_components": [9,None],
        "smote":["passthrough"]
    },
    
    # =========================
    # KNN
    # =========================
    {
        "model": [knn],
        "dimen": ["passthrough"],
        "smote": ["passthrough",SMOTE(k_neighbors=3,random_state=7)]
    }
]

rscv = GridSearchCV(pipe,param_grid=param_grid,n_jobs=-1,refit=True,cv=3,verbose=1)
t1 = time.time()
rscv.fit(x_train,y_train)
t2 = time.time()
min,sec = np.divmod((t2-t1),60)
print(f"Time Elapsed: {min} Minute {sec:.2f} Seconds")

est = rscv.best_estimator_
scr = rscv.best_score_
config = rscv.best_params_
print(f"Best Score = {scr}")
print(f"Best Configuration;\n{config}")
