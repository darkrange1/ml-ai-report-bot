from pathlib import Path
from fastapi import UploadFile, HTTPException
from .security import safe_filename

def ensure_csv(upload: UploadFile) -> None:
    if not upload.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")
    if not upload.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are accepted.")

def save_upload_to_tmp(upload: UploadFile, tmp_dir: Path, max_bytes: int) -> Path:
    import tempfile
    import shutil
    import os

    ensure_csv(upload)
    
    # Create a temp file in the system temp directory
    fd, tmp_path_str = tempfile.mkstemp(suffix=".csv")
    out_path = Path(tmp_path_str)
    
    # Close the low-level file descriptor, we will write using the path
    os.close(fd)

    total = 0
    with out_path.open("wb") as f:
        while True:
            chunk = upload.file.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                out_path.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="File too large.")
            f.write(chunk)
            
    return out_path
