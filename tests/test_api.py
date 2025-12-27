from fastapi.testclient import TestClient
from app.main import app
import pandas as pd
import io

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "🚀 API is Running!" in response.text

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["ok"] is True

def test_train_validation_error_small_file():

    df = pd.DataFrame({"col1": range(5), "target": range(5)})
    csv_file = io.BytesIO()
    df.to_csv(csv_file, index=False)
    csv_file.seek(0)
    
    response = client.post(
        "/train",
        params={"target": "target"},
        files={"file": ("test.csv", csv_file, "text/csv")}
    )
    
    assert response.status_code == 400
    assert "Dataset is too small" in response.json()["detail"]

def test_train_success():
    # Create CSV with 20 rows
    df = pd.DataFrame({
        "feature1": range(20),
        "target": [x % 2 for x in range(20)] # binary target
    })
    csv_file = io.BytesIO()
    df.to_csv(csv_file, index=False)
    csv_file.seek(0)
    
    response = client.post(
        "/train",
        params={"target": "target", "model_type": "RandomForest"},
        files={"file": ("test.csv", csv_file, "text/csv")}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert response.status_code == 200
    data = response.json()
    assert "metrics" in data
    assert "plots" in data

def test_train_clustering_api():
    # Create CSV 
    df = pd.DataFrame({
        "f1": [1, 2, 8, 9] * 5,
        "f2": [1, 2, 8, 9] * 5
    })
    csv_file = io.BytesIO()
    df.to_csv(csv_file, index=False)
    csv_file.seek(0)
    
    # Don't send target
    response = client.post(
        "/train",
        params={}, # No target
        files={"file": ("test.csv", csv_file, "text/csv")}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["task_type"] == "clustering"
    assert "silhouette_score" in data["metrics"]
