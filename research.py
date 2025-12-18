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
    import matplotlib.pyplot as plt
    import seaborn as sns
    import time

    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier as KNN
    from sklearn.ensemble import RandomForestClassifier,StackingClassifier

    from sklearn.model_selection import train_test_split, RandomizedSearchCV, learning_curve, GridSearchCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline

    from xgboost import XGBClassifier
    return (
        GridSearchCV,
        KNN,
        LDA,
        LogisticRegression,
        PCA,
        Pipeline,
        RandomForestClassifier,
        SMOTE,
        SVC,
        SimpleImputer,
        StandardScaler,
        XGBClassifier,
        learning_curve,
        np,
        pd,
        plt,
        sns,
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
    ### Importing dataset
    """)
    return


@app.cell
def _(pd):
    df_raw = pd.read_csv("Datasets/WineQT.csv")
    return (df_raw,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Removing identifier column
    """)
    return


@app.cell
def _(df_raw):
    df_1 = df_raw.drop(columns=["Id"],axis=0)
    return (df_1,)


@app.cell
def _(df):
    df.iloc[:,-1].value_counts()
    return


@app.cell
def _(df_1):
    df_1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Checking for null
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
    ### Encoding class labels to 0,1,2 (low, medium and high quality)
    """)
    return


@app.cell
def _(df_1):
    def mapper(x):
        if x in [3,4]:
            return 0
        elif x in [5,6]:
            return 1
        elif x in [7,8]:
            return 2

    df_1["label_quality"] = df_1["quality"].apply(mapper)
    df_2 = df_1.drop(["quality"],axis=1)
    return (df_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Copying `df_2` to `df`
    """)
    return


@app.cell
def _(df_2):
    df = df_2.copy()
    return (df,)


@app.cell
def _(df):
    df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Save Feature Names
    """)
    return


@app.cell
def _(df):
    feat_columns = df.iloc[:,:-1].columns
    all_columns = df.columns
    return (feat_columns,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Input-Output separation
    """)
    return


@app.cell
def _(df):
    x = df.iloc[:,:-1].to_numpy()
    y = df.iloc[:,-1].to_numpy()
    return x, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## EDA (Exploratory Data Analysis)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Correlational Heatmap
    """)
    return


@app.cell
def _(df, plt, sns):
    plt.figure(figsize=(12,5))
    corr = df.iloc[:,:-1].corr()
    sns.heatmap(corr,cmap="icefire",annot=True,fmt=".2f")
    plt.title("Correlational Heatmap of the features")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The heatmap clearly shows that few features actually correlate with each other. So, there are noise in the data which we have to account for. It means that we should keep dimensionality reduction approaches in Grid Search.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Class Distribution
    """)
    return


@app.cell
def _(df, plt, sns):
    sns.countplot(x="label_quality", data=df)
    plt.title("Distribution of Wine Quality Classes")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The dataset shows huge class imbalance, with medium-quality wines being the most frequent.
    This motivated the use of SMOTE during training to prevent the classifier from becoming biased
    towards the majority class.
    """)
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
    ### Defining Pipeline
    """)
    return


@app.cell
def _(
    KNN,
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
    knn = KNN()
    return knn, pipe, rf, xgb


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Grid Search CV
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Defining the `param grid`
    """)
    return


@app.cell
def _(LDA, PCA, SMOTE, knn, rf, xgb):
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
    return (param_grid,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Running Grid Search
    """)
    return


@app.cell
def _(GridSearchCV, np, param_grid, pipe, time, x_train, y_train):
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
    return (est,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Seems like Random forest with PCA and no SMOTE has yielded the best score in GridSearchCV.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **NOTE:** I tried out different configurations of param_grid when I was writing the code originally. I tested out with simpler models like Logistic Regression and SVC, but they were not yielding the best result in any case. So, I removed those configurations in order to reduce the number of fits, thus computational cost.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### PCA Loading
    """)
    return


@app.cell
def _(est):
    pca_est = est.named_steps["dimen"]
    return (pca_est,)


@app.cell
def _(pca_est):
    comps = pca_est.components_.T  # Components (Transposed)
    return (comps,)


@app.cell
def _(pca_est):
    exp_var = pca_est.explained_variance_  # Explained Variance
    return (exp_var,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As we know,

    $$ \text{Loadings} = \text{pca.components\_}^T \times \sqrt{\text{pca.explained\_variance\_}} $$
    """)
    return


@app.cell
def _(comps, exp_var):
    loadings = comps * exp_var    # Raw Loadings
    return (loadings,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now, let's decorate these in a Pandas DataFrame
    """)
    return


@app.cell
def _(comps, np):
    ii = np.size(comps,axis=1)
    return (ii,)


@app.cell
def _(feat_columns, ii, loadings, pd):
    loading_df = pd.DataFrame(columns=feat_columns,index=[f"PC{i}" for i in range(ii)],data=loadings)
    loading_df
    return (loading_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's visualize it
    """)
    return


@app.cell
def _(loading_df, plt, sns):
    melted_loadings = loading_df.reset_index().melt(id_vars='index', var_name='Component', value_name='Loading')
    melted_loadings.rename(columns={'index': 'Feature'}, inplace=True)


    g = sns.catplot(
        data=melted_loadings, 
        kind="bar", 
        x="Loading", 
        y="Feature", 
        col="Component", 
        col_wrap=3, 
        palette="Paired",
        sharex=True,
        height=4, 
        aspect=1.2,
        hue="Feature"
    )

    # 3. Add styling
    g.set_titles("{col_name}") # Titles will be PC1, PC2, etc.
    g.set_axis_labels("Loading Score", "Features")

    # Add a vertical line at 0 for every subplot to help identify direction
    for ax in g.axes.flat:
        ax.axvline(0, color='black', lw=1, alpha=0.5)

    plt.tight_layout()
    plt.show()

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    From this huge-sized plot, we can see that features like "fixed acidity", "volatile acidity" and "citric acid" have contributed a lot to the principal components. On the contrary, "alcohol", "sulphates" and "pH" have barely contributed to the PCs.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Learning Curve
    """)
    return


@app.cell
def _(est, learning_curve, np, x_train, y_train):
    train_size, train_acc, val_acc = learning_curve(
      est,x_train,y_train,train_sizes=np.linspace(0.1,1.0,10),
      cv=5,n_jobs=-1,random_state=9,shuffle=True
    )
    return train_acc, train_size, val_acc


@app.cell
def _(np, plt, train_acc, train_size, val_acc):
    train_mean = np.mean(train_acc, axis=1)
    train_std = np.std(train_acc,axis=1)
    val_mean = np.mean(val_acc,axis=1)
    val_std = np.std(val_acc,axis=1)

    plt.figure(figsize=(10,6))
    plt.plot(train_size, train_mean, color="red",marker="s",markersize=4,label="Training Accuracy")
    plt.fill_between(train_size, train_mean + train_std , train_mean - train_std, color="red",alpha=0.3)

    plt.plot(train_size, val_mean, color="orange",marker="v",markersize=4,label="Validation Accuracy")
    plt.fill_between(train_size, val_mean + val_std, val_mean - val_std, color="orange",alpha=0.3)

    plt.title("Learning Curve (Random Forest with PCA)",fontdict={"fontsize":16})
    plt.xlabel("Train Size",fontdict={"fontsize":13})
    plt.ylabel("Accuracy",fontdict={"fontsize":13})
    plt.ylim(0.6,1.03)
    plt.legend()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The learning curve plot shows clear sign of overfitting. The validation accuracy line peaks around 0.87, whereas the training accuracy line consistently remains at 1.00. In future updates, I will try to implement further techniques to reduce this gap as much as I can.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Limitations and Future Work
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This project uses a relatively small dataset, which limits the effectiveness of high-capacity models.
    Despite cross-validation, some degree of variance remains unavoidable.

    Future improvements could include:
    - Testing on a larger and more diverse wine dataset
    - Applying probability calibration for better confidence estimates
    - Evaluating class-wise ROC–AUC instead of only accuracy-based metrics
    - Performing feature importance analysis to improve interpretability
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In this notebook, we implemented some of the best standards of machine learning classification to develop a model that predicts wine quality from three classes. We also included some data visualization to help us understand the data and the model’s performance. In future updates, I’ll do my best to improve it even further.

    **Thank you** for taking the time to look at this notebook. I hope it gave you a clearer understanding of my project goals. It’s also a crucial part of the project, as I directly used code from it in the final script.
    """)
    return


if __name__ == "__main__":
    app.run()
