from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from pathlib import Path
from sklearn.decomposition import PCA
import kagglehub
import pandas as pd
import joblib

def download():
  path = kagglehub.dataset_download("yasserh/wine-quality-dataset")
  df = pd.read_csv(f"{path}/WineQT.csv") 
  return df

def modifcation(x):
  if x in [3,4]:
    return 0
  elif x in [5,6]:
    return 1
  elif x in [7,8]:
    return 2

def split(df=download()):
  df["label_quality"] = df["quality"].apply(modifcation)

  X = df.iloc[:,:-1].drop(["quality","Id"],axis=1)
  names = X.columns
  X = X.to_numpy()
  y = df.iloc[:,-1].to_numpy()

  x_train,x_test,y_train,y_test = train_test_split(
    X,y,test_size=1/3,stratify=y,random_state=91,shuffle=True)
  return x_train,x_test,y_train,y_test,names

def model(df = download()):
  x_train,x_test,y_train,y_test,root_columns = split(df=df)
  model = RandomForestClassifier(
    n_estimators=1500,max_depth=None,random_state=391,class_weight="balanced")
  pca = PCA(n_components=None)

  preprocessor = Pipeline([
    ("imputation",SimpleImputer(strategy="median")),
    ("scaling",StandardScaler())
  ])
  pipe = Pipeline([
    ("preprocessing",preprocessor),
    ("pca",pca),
    ("model",model)
  ])

  pipe.fit(x_train,y_train)

  y_true = y_test
  y_pred = pipe.predict(x_test)
  print(classification_report(y_true=y_true,y_pred=y_pred))
  return pipe,root_columns

# Output:
#            0       0.00      0.00      0.00        13
#            1       0.88      1.00      0.94       315
#            2       0.96      0.47      0.63        53

#     accuracy                           0.89       381
#    macro avg       0.62      0.49      0.52       381
# weighted avg       0.87      0.89      0.86       381


def dumping():
  pipeline, root_columns = model()

  # Persist the trained pipeline and the selected feature names so other tools (CLI)
  # can load the exact preprocessing + model and know the expected input order.
  out_dir = Path(__file__).parent
  models_dir = out_dir / "models"
  models_dir.mkdir(parents=True, exist_ok=True)

  model_path = models_dir / "wine_pipeline.joblib"
  all_features_path = models_dir / "wine_all_features.joblib"

  # Save the full pipeline (preprocessing, feature selector, and model)
  joblib.dump(pipeline, model_path)
  joblib.dump(root_columns.tolist(), all_features_path)

  print(f"Saved trained pipeline to: {model_path}")
  print(f"Saved full feature list to: {all_features_path}")

def main():
  df_m = download()
  model(df_m)
  dumping()

if __name__ == "__main__":
  main()
