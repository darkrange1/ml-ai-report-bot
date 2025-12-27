from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class TrainResponse(BaseModel):
    task_type: str
    model_name: str
    metrics: dict
    plots: Optional[Dict[str, str]] = None
    report_path: str

class ColumnsResponse(BaseModel):
    columns: List[str]

class SuggestTargetRequest(BaseModel):
    columns: List[str]
    sample_rows: Optional[List[Dict[str, Any]]] = None

class SuggestTargetResponse(BaseModel):
    target: Optional[str]
    suggested_model: str
    rationale: str
