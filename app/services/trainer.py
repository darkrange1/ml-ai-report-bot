from __future__ import annotations

import pandas as pd
import numpy as np
import base64
import io
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns



from dataclasses import dataclass, field
from typing import Literal, Any

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

TaskType = Literal["classification", "regression", "clustering"]

@dataclass
class TrainResult:
    task_type: TaskType
    model_name: str
    metrics: dict
    plots: dict[str, str] = field(default_factory=dict)  # name -> base64_img
    metrics: dict
    plots: dict[str, str] = field(default_factory=dict)  # name -> base64_img
    cluster_profiles: dict | None = None # For LLM analysis
    pipeline: Any = None

def infer_task_type(y: pd.Series) -> TaskType:
    if y.dtype == "object" or str(y.dtype).startswith("category") or y.dtype == "bool":
        return "classification"
    unique = y.nunique(dropna=True)
    if unique <= 20:
        return "classification"
    return "regression"

def plot_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return data

def train(df: pd.DataFrame, target: str, model_type: str = "RandomForest") -> TrainResult:
    if target not in df.columns:
        raise ValueError(f"target '{target}' not found.")

    df = df.copy()
    df = df.dropna(subset=[target])

    y = df[target]
    X = df.drop(columns=[target])

    task_type: TaskType = infer_task_type(y)

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]


    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop",
    )


    model = None
    if task_type == "classification":
        if model_type == "Linear":
            model = LogisticRegression(max_iter=1000)
        elif model_type == "DecisionTree":
            model = DecisionTreeClassifier(max_depth=10, random_state=42)
        elif model_type == "XGBoost":
            model = XGBClassifier(n_estimators=100, eval_metric="logloss", random_state=42, n_jobs=-1)
        else: # Default RandomForest
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    else: # Regression
        if model_type == "Linear":
            model = Ridge() # Ridge is safer than pure LinearRegression
        elif model_type == "DecisionTree":
            model = DecisionTreeRegressor(max_depth=10, random_state=42)
        elif model_type == "XGBoost":
            model = XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

    pipe = Pipeline(steps=[
        ("prep", preprocessor),
        ("model", model),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if task_type == "classification" else None
    )

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)


    metrics = {}
    plots = {}

    if task_type == "classification":
        metrics["accuracy"] = float(accuracy_score(y_test, y_pred))
        metrics["f1_macro"] = float(f1_score(y_test, y_pred, average="macro"))
        metrics["test_size"] = int(len(y_test))
        

        if len(np.unique(y_test)) < 50: # Avoid huge matrices
            fig, ax = plt.subplots(figsize=(6, 5))
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title("Confusion Matrix")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            plots["congestion_matrix"] = plot_to_base64(fig)
            
    else: # Regression
        metrics["mae"] = float(mean_absolute_error(y_test, y_pred))
        metrics["rmse"] = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        metrics["r2"] = float(r2_score(y_test, y_pred))
        metrics["test_size"] = int(len(y_test))
        

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, ax=ax)
        # Perfect line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--')
        ax.set_title("Actual vs Predicted")
        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        plots["actual_vs_pred"] = plot_to_base64(fig)


    if hasattr(model, "feature_importances_"):
        try:
            # Get feature names from preprocessor
            # This can be tricky with OneHotEncoder, we try best effort
            feature_names = []
            if hasattr(preprocessor, "get_feature_names_out"):
                feature_names = preprocessor.get_feature_names_out()
            
            importances = model.feature_importances_
            if len(feature_names) == len(importances):
                # Sort and plot top 10
                indices = np.argsort(importances)[::-1][:10]
                top_features = [feature_names[i] for i in indices]
                top_importances = importances[indices]
                
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.barplot(x=top_importances, y=top_features, ax=ax, palette="viridis")
                ax.set_title("Top 10 Feature Importances")
                plots["feature_importance"] = plot_to_base64(fig)
        except Exception:
            pass # Skip if feature names extraction fails

    return TrainResult(
        task_type=task_type,
        model_name=f"{type(model).__name__} ({model_type})",
        metrics=metrics,
        plots=plots,
        cluster_profiles=None,
        pipeline=pipe
    )

def train_clustering(df: pd.DataFrame, n_clusters: int = 3, model_type: str = "KMeans") -> TrainResult:
    # Since we don't have a target, we use all columns.
    
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]
    
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop",
    )
    

    if model_type == "DBSCAN":
        # DBSCAN doesn't take n_clusters. We use default or heuristic eps.
        # Simple heuristic: eps=0.5 (scaled data standard), min_samples=5
        model = DBSCAN(eps=0.5, min_samples=5)
    else:
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    
    pipe = Pipeline(steps=[
        ("prep", preprocessor),
        ("model", model),
    ])
    

    pipe.fit(df)
    labels = pipe.named_steps["model"].labels_
    

    metrics = {}
    try:
        # Silhouette requires at least 2 clusters and > 1 sample
        if n_clusters > 1 and df.shape[0] > 1:
            transformed_data = pipe.named_steps["prep"].transform(df)
            metrics["silhouette_score"] = float(silhouette_score(transformed_data, labels))
    except Exception:
        metrics["silhouette_score"] = 0.0
        
    metrics["n_clusters"] = n_clusters
    

    plots = {}
    try:
        transformed_data = pipe.named_steps["prep"].transform(df)
        
        # PCA to 2D
        if transformed_data.shape[1] > 1:
            pca = PCA(n_components=2)
            components = pca.fit_transform(transformed_data)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            scatter = ax.scatter(components[:, 0], components[:, 1], c=labels, cmap='viridis', alpha=0.6)
            ax.set_title(f"Clustering (PCA 2D Projection)")
            ax.set_xlabel("PCA Component 1")
            ax.set_ylabel("PCA Component 2")
            plt.colorbar(scatter, ax=ax, label="Cluster ID")
            
            plots["cluster_pca"] = plot_to_base64(fig)
    except Exception as e:
        print(f"PCA Plot Error: {e}")
        
    try:
        # Elbow Method Plot
        # We calculate inertia for k=2 to k=10 (or less if sample size is small)
        max_k = min(11, df.shape[0])
        if max_k > 2:
            inertias = []
            k_range = range(2, max_k)
            
            # Use the same preprocessor first
            X_transformed = pipe.named_steps["prep"].transform(df)
            
            for k in k_range:
                km = KMeans(n_clusters=k, random_state=42, n_init="auto")
                km.fit(X_transformed)
                inertias.append(km.inertia_)
                
            fig_elbow, ax_elbow = plt.subplots(figsize=(8, 6))
            ax_elbow.plot(k_range, inertias, 'bo-', markersize=8)
            ax_elbow.set_title("Elbow Method (Inertia vs K)")
            ax_elbow.set_xlabel("Number of Clusters (k)")
            ax_elbow.set_ylabel("Inertia")
            ax_elbow.grid(True)
            
            plots["elbow_curve"] = plot_to_base64(fig_elbow)
    except Exception as e:
        print(f"Elbow Plot Error: {e}")

    except Exception as e:
        print(f"Elbow Plot Error: {e}")


    cluster_profiles = {}
    try:
        # We need original numeric values for interpretation, not scaled ones.
        # So we inverse transform or just use the original DF's numeric columns + labels.
        # Simpler: Use original df numeric columns + label column.
        
        # Add labels to a temporary dataframe
        df_profile = df.copy()
        df_profile["Cluster"] = labels
        
        # Group by Cluster and calc mean of numeric columns
        # Filter out noise (-1) from DBSCAN if desired, or keep it.
        profile_df = df_profile.groupby("Cluster")[num_cols].mean()
        
        # Convert to dict for LLM
        # Structure: {0: {'age': 25.5, 'income': 5000}, 1: {...}}
        cluster_profiles = profile_df.to_dict(orient="index")
    except Exception as e:
        print(f"Profiling Error: {e}")

    return TrainResult(
        task_type="clustering",
        model_name=f"{model_type} Clustering",
        metrics=metrics,
        plots=plots,
        cluster_profiles=cluster_profiles,
        pipeline=pipe
    )
