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

    In December 2025, I added new updates. It includes a totally new dataset, newer and more robust classifiers and techniques. In this notebook, you can see exactly how I have crafted the model to perform its best!
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
    return (pd,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Importing the dataset and basic preprocessing
    """)
    return


@app.cell
def _(pd):
    df_raw = pd.read_csv("Datasets/Wine dataset.csv")
    df_raw
    return


if __name__ == "__main__":
    app.run()
