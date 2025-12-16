import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Wine Research
    In this practice project, I have done some work on the famous wine dataset from Kaggle. I have used this notebook primarily as a testing playground. You can check the different scores I received while testing with various different models at the bottom of this notebook.

    In December 2025, I added new updates. It includes, newer and more robust classifiers and techniques. In this notebook, you can see exactly how I have crafted the model to perform its best! I also wanted to test on a totally new dataset. But, I couldn't find a reasonable one. So the previous dataset is still used in the newer version.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Importing required libraries
    """)
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import time

    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    # from sklearn.neighbors import KNeighborsClassifier as knn
    from sklearn.ensemble import RandomForestClassifier,StackingClassifier

    from sklearn.model_selection import train_test_split, RandomizedSearchCV, learning_curve
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline

    from xgboost import XGBClassifier
    return (
        LDA,
        LogisticRegression,
        PCA,
        Pipeline,
        RandomForestClassifier,
        RandomizedSearchCV,
        SMOTE,
        SVC,
        SimpleImputer,
        StandardScaler,
        XGBClassifier,
        np,
        pd,
        time,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Importing the dataset and basic preprocessing
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Importing dataset
    """)
    return


@app.cell
def _(pd):
    df_raw = pd.read_csv("Datasets/WineQT.csv")
    return (df_raw,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Removing identifier column
    """)
    return


@app.cell
def _(df_raw):
    df_1 = df_raw.drop(columns=["Id"],axis=0)
    return (df_1,)


@app.cell
def _(df_1):
    df_1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Checking for null
    """)
    return


@app.cell
def _(df_1):
    df_1.isna().value_counts()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    No need to worry about null, because there are no null values
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Copying `df_1` to `df`
    """)
    return


@app.cell
def _(df_1):
    df = df_1.copy()
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Input-Output separation
    """)
    return


@app.cell
def _(df):
    x = df.iloc[:,:-1].to_numpy()
    y = df.iloc[:,-1].to_numpy()
    return x, y


@app.cell
def _(df):
    df.iloc[:,-1].value_counts()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Machine Learning Part
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Train Test Split
    """)
    return


@app.cell
def _(train_test_split, x, y):
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=1/4,random_state=32,shuffle=True,stratify=y)
    return x_train, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Defining Pipeline
    """)
    return


@app.cell
def _(
    LogisticRegression,
    PCA,
    Pipeline,
    RandomForestClassifier,
    SMOTE,
    SVC,
    SimpleImputer,
    StandardScaler,
    XGBClassifier,
):
    pipe = Pipeline([
        ("imputation",SimpleImputer(strategy="median")),
        ("scale",StandardScaler()),
        ("smote",SMOTE(random_state=21,k_neighbors=3)),
        ("dimen",PCA(n_components=5)),
        ("model",RandomForestClassifier(class_weight="balanced"))
    ])

    rf = RandomForestClassifier(class_weight="balanced",random_state=32)
    lr = LogisticRegression(class_weight="balanced",random_state=43)
    svc = SVC(class_weight="balanced",random_state=78)
    xgb = XGBClassifier(random_state=902)
    return lr, pipe, rf, svc, xgb


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Randomized Search CV
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Defining the `param grid`
    """)
    return


@app.cell
def _(LDA, PCA, lr, np, rf, svc, xgb):
    param_grid = [
        {
            "model": [rf], "model__n_estimators":np.arange(700,1100,100),"model__max_depth":[5,7,10,12], 
            "dimen": [LDA()]
        },
        {
            "model": [rf], "model__n_estimators":np.arange(700,1100,100),"model__max_depth":[5,7,10,12], 
            "dimen": [PCA(random_state=9011)], "dimen__n_components":[3,6,9]
        },
        {
            "model": [rf], "model__n_estimators":np.arange(700,1100,100),"model__max_depth":[5,7,10,12], 
            "dimen": ["passthrough"]
        },
        {
            "model": [xgb],"model__n_estimators": np.arange(700, 1100, 4).tolist(),"model__learning_rate":[0.01,0.1],
            "model__max_depth":[5,7,10,12], "dimen": [LDA()]
        },
        {
            "model": [xgb],"model__n_estimators": np.arange(700, 1100, 4).tolist(),"model__learning_rate":[0.01,0.1],
            "model__max_depth":[5,7,10,12], "dimen": [PCA(random_state=401)], "dimen__n_components": [3,6,9]
        },
        {
            "model": [xgb],"model__n_estimators": np.arange(700, 1100, 4).tolist(),"model__learning_rate":[0.01,0.1],
            "model__max_depth":[5,7,10,12], "dimen": ["passthrough"]
        },
        {
            "model": [lr], "model__C": [0.01,0.1,1,10,100], "model__solver":["lbfgs","newton-cg"],
            "model__penalty": ["l2"],"model__max_iter":[5000], "dimen":["passthrough"]
        },
        {
            "model": [svc],"model__C":[0.01,0.1,1,10,100], "model__kernel":["rbf"],"model__max_iter":[5000],
            "dimen":["passthrough"]
        }
    ]
    return (param_grid,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Running Randomized Search
    """)
    return


@app.cell
def _(RandomizedSearchCV, np, param_grid, pipe, time, x_train, y_train):
    rscv = RandomizedSearchCV(pipe,param_distributions=param_grid,n_iter=10,cv=5,n_jobs=-1,verbose=1,refit=True)
    t1 = time.time()
    rscv.fit(x_train,y_train)
    t2 = time.time()
    hour,min = np.divmod((t2-t1),60)
    print(f"Time Elapsed: {hour} Hour {min} Minutes")

    est = rscv.best_estimator_
    scr = rscv.best_score_
    config = rscv.best_params_
    print(f"Best Score = {scr}")
    print(f"Best Configuration;\n{config}")
    return


if __name__ == "__main__":
    app.run()
