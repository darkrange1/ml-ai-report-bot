from fastapi import HTTPException
import pandas as pd
import numpy as np

def validate_dataset(df: pd.DataFrame, target: str | None = None):
    """
    Validates the dataset ensures it is suitable for training.
    Raises HTTPException if validation fails.
    If target is None, we assume Clustering task (unsupervised).
    """
    # 1. Check strict minimum size
    if df.shape[0] < 10:
        raise HTTPException(
            status_code=400, 
            detail=f"Dataset is too small ({df.shape[0]} rows). Please provide at least 10 rows."
        )

    # If Clustering (target is None), skip target checks
    if target is None:
        if df.shape[1] < 1:
             raise HTTPException(status_code=400, detail="Dataset must have at least one numeric/categorical column.")
        return

    # 2. Check target column existence
    if target not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{target}' not found in dataset."
        )

    # 3. Check for empty target column
    if df[target].isna().all():
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{target}' is completely empty (all NaN)."
        )

    # 4. Check valid rows after dropna
    valid_rows = df.dropna(subset=[target]).shape[0]
    if valid_rows < 5:
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{target}' has too many missing values. Only {valid_rows} valid rows remaining."
        )
    
    # 5. Check if there are any features left
    if df.shape[1] < 2:
        raise HTTPException(
            status_code=400,
            detail="Dataset must have at least one feature column along with the target."
        )

    # 6. Check for extremely high cardinality in target (if classification)
    # This is a heuristic/warning check, but for now we enforce it to prevent memory explosions
    # or useless models on ID-like columns.
    unique_count = df[target].nunique()
    dtype = df[target].dtype
    is_numeric = pd.api.types.is_numeric_dtype(dtype)

    # If it looks like classification (object or low unique numeric)
    is_classification = not is_numeric or (is_numeric and unique_count < 20)
    
    if is_classification and unique_count > 500:
         raise HTTPException(
            status_code=400,
            detail=f"Target column '{target}' has {unique_count} unique classes, which is too high for this classification tool. It might be an ID column."
        )
