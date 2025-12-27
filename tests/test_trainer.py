import pandas as pd
import pytest
from app.services.trainer import train, infer_task_type, train_clustering

def test_infer_task_type():
    # Classification cases
    s_cat = pd.Series(["a", "b", "a"], dtype="object")
    assert infer_task_type(s_cat) == "classification"
    
    s_num_low = pd.Series([1, 2, 1, 2, 1], dtype="int")
    assert infer_task_type(s_num_low) == "classification" # < 20 unique

    # Regression cases
    s_num_high = pd.Series(range(100), dtype="int")
    assert infer_task_type(s_num_high) == "regression"

def test_train_classification():
    # Synthetic dataset
    data = {
        "feature1": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
        "feature2": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    }
    df = pd.DataFrame(data)
    
    result = train(df, target="target", model_type="RandomForest")
    
    assert result.task_type == "classification"
    assert "accuracy" in result.metrics
    assert result.metrics["accuracy"] >= 0.0 # Just check it runs
    assert "congestion_matrix" in result.plots

def test_train_regression():
    # Synthetic dataset
    # Must have > 20 unique values to be regression
    data = {
        "feature1": range(50),
        "target": [x * 2 for x in range(50)] # Perfect linear line
    }
    df = pd.DataFrame(data)
    
    result = train(df, target="target", model_type="Linear")
    
    assert result.task_type == "regression"
    assert result.metrics["r2"] > 0.9 # Should be perfect
    assert result.metrics["r2"] > 0.9 # Should be perfect
    assert "actual_vs_pred" in result.plots

def test_train_xgboost():
    # Synthetic dataset
    data = {
        "feature1": [1, 2, 3, 4, 5] * 2,
        "target": [0, 1, 0, 1, 0] * 2
    }
    df = pd.DataFrame(data)
    
    result = train(df, target="target", model_type="XGBoost")
    
    assert result.task_type == "classification"
    assert "XGBClassifier" in result.model_name
    assert result.metrics["accuracy"] >= 0.0

def test_train_clustering():
    # Synthetic dataset
    data = {
        "f1": [1, 2, 1, 2, 8, 9, 8, 9],
        "f2": [1, 1, 2, 2, 8, 8, 9, 9]
    }
    df = pd.DataFrame(data)
    
    result = train_clustering(df, n_clusters=2)
    
    assert result.task_type == "clustering"
    assert "KMeans" in result.model_name
    assert result.metrics["silhouette_score"] > 0
    assert result.task_type == "clustering"
    assert "KMeans" in result.model_name
    assert result.metrics["silhouette_score"] > 0
    assert "cluster_pca" in result.plots
    assert "elbow_curve" in result.plots

def test_train_clustering_dbscan():
    # Synthetic dataset with dense clusters
    data = {
        "x": [1, 1.1, 0.9, 10, 10.1, 9.9],
        "y": [1, 1.1, 0.9, 10, 10.1, 9.9]
    }
    df = pd.DataFrame(data)
    
    result = train_clustering(df, model_type="DBSCAN")
    
    assert result.task_type == "clustering"
    assert "DBSCAN" in result.model_name
    
    # DBSCAN might find noise (-1) or clusters (0, 1...)
    # We check if cluster_profiles is generated
    assert result.cluster_profiles is not None
    assert isinstance(result.cluster_profiles, dict)
