from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.pipeline import Pipeline
from pathlib import Path
import kagglehub
import pandas as pd
import joblib

# Download the dataset
path = kagglehub.dataset_download("yasserh/wine-quality-dataset")

# Load the CSV file (KaggleHub returns the dataset folder path)
df = pd.read_csv(f"{path}/WineQT.csv")  # The file name in that dataset is WineQT.csv

def modifcation(x):
  if x in [3,4]:
    return 0
  elif x in [5,6]:
    return 1
  elif x in [7,8]:
    return 2

df["label_quality"] = df["quality"].apply(modifcation)

X = df.iloc[:,:-1].drop(["quality","Id"],axis=1)
names = X.columns
X = X.to_numpy()
y = df.iloc[:,-1].to_numpy()

x_train,x_test,y_train,y_test = train_test_split(
  X,y,test_size=1/3,stratify=y,random_state=91,shuffle=True)

model = RandomForestClassifier(
  n_estimators=1500,max_depth=None,random_state=391,class_weight="balanced")
sfs = SequentialFeatureSelector(
  estimator=model,n_features_to_select="auto",tol=1e-5,direction="backward",cv=3)

preprocessor = Pipeline([
  ("imputation",SimpleImputer(strategy="median")),
  ("scaling",StandardScaler())
])
pipe = Pipeline([
  ("preprocessing",preprocessor),
  ("sfs",sfs),
  ("model",model)
])
pipe.fit(x_train,y_train)

y_true = y_test
y_pred = pipe.predict(x_test)
print(classification_report(y_true=y_true,y_pred=y_pred))

# Output:
#               precision    recall  f1-score   support

#            0       0.00      0.00      0.00        13
#            1       0.89      0.98      0.93       315
#            2       0.85      0.53      0.65        53
# 
#     accuracy                           0.88       381
#    macro avg       0.58      0.50      0.53       381
# weighted avg       0.85      0.88      0.86       381



# Fetching Feature Names
sfs_ = pipe.named_steps["sfs"]
scaler = pipe.named_steps["preprocessing"]

mask = sfs_.get_support()


print(names[mask])

# Output:
# Index(['volatile acidity', 'citric acid', 'free sulfur dioxide',
#        'total sulfur dioxide', 'density', 'pH', 'sulphates'],
#       dtype='object')

# Persist the trained pipeline and the selected feature names so other tools (CLI, webapps)
# can load the exact preprocessing + model and know the expected input order.

out_dir = Path(__file__).parent
model_path = out_dir / "wine_pipeline.joblib"
meta_path = out_dir / "wine_features.joblib"

# Also save the full original feature list (all 11 features) so callers can
# prompt for all inputs and let the pipeline's selector handle selecting the
# subset at predict time.
all_features_path = out_dir / "wine_all_features.joblib"

# Save the full pipeline (preprocessing, feature selector, and model)
joblib.dump(pipe, model_path)
# Save the selected feature names in order so callers can build input arrays correctly
joblib.dump(names[mask].tolist(), meta_path)
joblib.dump(names.tolist(), all_features_path)

print(f"Saved trained pipeline to: {model_path}")
print(f"Saved selected feature names to: {meta_path}")
print(f"Saved full feature list to: {all_features_path}")
