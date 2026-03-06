from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from src.ensemble_model import build_voting_classifier

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
    transformers = []

    if numeric_features:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_features))

    if categorical_features:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_features))

    if not transformers:
        raise ValueError("Could not build a preprocessor because no usable feature columns exist.")

    return ColumnTransformer(transformers=transformers, remainder="drop")

def build_model_pipelines(
    X: pd.DataFrame, random_state: int = 42
) -> dict[str, Pipeline]:
    preprocessor = build_preprocessor(X)

    decision_tree_pipeline = Pipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
            (
                "classifier",
                DecisionTreeClassifier(
                    random_state=random_state,
                    max_depth=8,
                    min_samples_split=4,
                    min_samples_leaf=2,
                ),
            ),
        ]
    )
    knn_pipeline = Pipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
            ("classifier", KNeighborsClassifier(n_neighbors=7, weights="distance")),
        ]
    )
    voting_pipeline = Pipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
            ("classifier", build_voting_classifier(random_state=random_state)),
        ]
    )

    return {
        "Decision Tree": decision_tree_pipeline,
        "KNN": knn_pipeline,
        "Voting Ensemble": voting_pipeline,
    }
