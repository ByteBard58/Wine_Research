import numpy as np
import pandas as pd
import joblib
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

from sklearn.base import BaseEstimator
from sklearn.metrics import classification_report

from xgboost import XGBClassifier

def mapper(x):
    if x in [3,4]:
        return 0
    elif x in [5,6]:
        return 1
    elif x in [7,8]:
        return 2

def get_process_data(path="Datasets/WineQT.csv") -> tuple[np.ndarray,np.ndarray,pd.Index]:
  df_raw: pd.DataFrame = pd.read_csv(path)
  df_1: pd.DataFrame = df_raw.drop(columns=["Id"],axis=0)

  df_1["label_quality"]: pd.Series = df_1["quality"].apply(mapper)
  df_2: pd.DataFrame = df_1.drop(["quality"],axis=1)

  df: pd.DataFrame = df_2.copy()

  all_columns: pd.Index = df.columns

  x: np.ndarray = df.iloc[:,:-1].to_numpy()
  y: np.ndarray = df.iloc[:,-1].to_numpy()

  return x,y,all_columns


def get_pipe() -> Pipeline:
  pipe: Pipeline = Pipeline([
      ("imputation",SimpleImputer(strategy="median")),
      ("scale",StandardScaler()),
      ("smote",SMOTE(random_state=21,k_neighbors=3)),
      ("dimen",PCA(n_components=5)),
      ("model",RandomForestClassifier(class_weight="balanced"))
  ])
  return pipe



def evaluate(est:BaseEstimator,x_test:np.ndarray,y_test:np.ndarray) -> None:
  y_true = y_test
  y_pred = est.predict(x_test)
  print(classification_report(y_true=y_true,y_pred=y_pred))

def main() -> None:
  x,y,all_columns = get_process_data()

  x_train,x_test,y_train,y_test = train_test_split(
    x,y,test_size=1/4,random_state=32,shuffle=True,stratify=y
  )
  pipe = get_pipe()

  rf: BaseEstimator = RandomForestClassifier(class_weight="balanced",random_state=32)
  xgb: BaseEstimator = XGBClassifier(random_state=902)
  knn: BaseEstimator = KNN()

  param_grid: list = [

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

  rscv:BaseEstimator = GridSearchCV(pipe,param_grid=param_grid,n_jobs=-1,refit=True,cv=3,verbose=1)
  print("Starting model fitting...")
  print("It may take a while, please sit tight...")
  t1:float = time.time()
  rscv.fit(x_train,y_train)
  t2:float = time.time()
  print("Model fitting completed successfully ✅")
  min,sec = np.divmod((t2-t1),60)
  print(f"Time Elapsed: {min} Minute {sec:.2f} Seconds")

  est:BaseEstimator = rscv.best_estimator_
  scr:float = rscv.best_score_
  config:dict = rscv.best_params_
  print(f"Best Score = {scr}")
  print(f"Best Configuration;\n{config}")

  evaluate(est,x_test,y_test)
  joblib.dump(est,"models/model.pkl")
  joblib.dump(all_columns,"models/columns.pkl")
  print("Models saved successfully ✅")


if __name__ == "__main__":
  main()
