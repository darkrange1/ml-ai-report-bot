from __future__ import annotations

from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse

from .config import settings
from .schemas import (
    TrainResponse,
    ColumnsResponse,
    SuggestTargetRequest,
    SuggestTargetResponse,
)
from .utils.files import save_upload_to_tmp
from .services.data_prep import load_csv, basic_profile
from .services.trainer import train, train_clustering
from .services.report import render_report
from .services.llm import suggest_target, fallback_suggest
from .services.validator import validate_dataset

APP_ROOT = Path(__file__).resolve().parent.parent
TMP_DIR = APP_ROOT / "tmp"
REPORTS_DIR = APP_ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="ML Report Bot API", version="1.4.0")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# serve reports folder (serve under same origin for CSS access)
app.mount("/reports", StaticFiles(directory=str(REPORTS_DIR)), name="reports")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
def read_root():
    return """
    <html>
        <head>
            <title>ML Report Bot API</title>
            <style>
                body { font-family: sans-serif; text-align: center; padding-top: 50px; background: #f9fafb; }
                .container { max-width: 600px; margin: 0 auto; background: white; padding: 40px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
                h1 { color: #2563eb; }
                p { color: #4b5563; line-height: 1.6; }
                code { background: #f3f4f6; padding: 4px 8px; border-radius: 4px; font-family: monospace; }
                a { color: #2563eb; text-decoration: none; font-weight: bold; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🚀 API is Running!</h1>
                <p>ML Report Bot backend service started successfully.</p>
                <p>
                    📌 <strong>API Documentation:</strong> Visit <a href="/docs">/docs</a>.<br>
                    🎨 <strong>To use the UI:</strong> Open a new terminal and run:
                </p>
                <p><code>streamlit run ui/app.py</code></p>
            </div>
        </body>
    </html>
    """

@app.get("/health")
def health():
    return {"ok": True, "env": settings.app_env, "llm_active": bool(settings.gemini_api_key)}

@app.post("/suggest-target", response_model=SuggestTargetResponse)
def suggest_target_endpoint(payload: SuggestTargetRequest):
    cols = payload.columns or []
    if not cols:
        raise HTTPException(status_code=400, detail="columns is empty.")

    if settings.gemini_api_key:
        sug = suggest_target(
            api_key=settings.gemini_api_key,
            columns=cols,
            sample_rows=payload.sample_rows,
        )
    else:
        sug = fallback_suggest(cols)

    return SuggestTargetResponse(
        target=sug.target,
        suggested_model=sug.suggested_model,
        rationale=sug.rationale
    )

@app.post("/train", response_model=TrainResponse)
def train_endpoint(target: str | None = None, model_type: str = "RandomForest", file: UploadFile = File(...)):
    # If target is "None" string from UI, treat as None
    if target == "None" or target == "":
        target = None
    max_bytes = settings.max_upload_mb * 1024 * 1024
    tmp_path = save_upload_to_tmp(file, TMP_DIR, max_bytes=max_bytes)

    try:
        df = load_csv(str(tmp_path))
        

        validate_dataset(df, target)

        profile = basic_profile(df)
        
        if target:
            result = train(df, target=target, model_type=model_type)
        else:
            # Clustering (Unsupervised)
            # Use model_type to pass n_clusters if needed, but for now fixed or parsed
            result = train_clustering(df, n_clusters=3)

        report_path = render_report(
            out_dir=REPORTS_DIR,
            template_dir=TEMPLATES_DIR,
            title="ML Dataset Report",
            profile=profile,
            train_result={
                "task_type": result.task_type,
                "model_name": result.model_name,
                "metrics": result.metrics,
                "plots": result.plots,
            }
        )

        return TrainResponse(
            task_type=result.task_type,
            model_name=result.model_name,
            metrics=result.metrics,
            plots=result.plots,
            report_path=str(report_path),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training/report error: {e}")
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

@app.get("/download-report")
def download_report(path: str):
    p = Path(path)
    reports_dir = REPORTS_DIR.resolve()
    p_abs = p.resolve()

    if not str(p_abs).startswith(str(reports_dir)):
        raise HTTPException(status_code=403, detail="Invalid report path.")
    if not p_abs.exists():
        raise HTTPException(status_code=404, detail="Report not found.")

    return FileResponse(str(p_abs), media_type="text/html", filename=p_abs.name)

@app.get("/open-report", response_class=HTMLResponse)
def open_report(path: str):
    """
    To allow UI to 'Open in New Tab'.
    Actually we serve the file under /reports; this endpoint just generates a safe URL.
    """
    p = Path(path)
    reports_dir = REPORTS_DIR.resolve()
    p_abs = p.resolve()

    if not str(p_abs).startswith(str(reports_dir)):
        raise HTTPException(status_code=403, detail="Invalid report path.")
    if not p_abs.exists():
        raise HTTPException(status_code=404, detail="Report not found.")

    # /reports/<filename> şeklinde aynı origin'de açılır → CSS yüklenir
    return HTMLResponse(
        f'<!doctype html><html><head><meta charset="utf-8"></head>'
        f'<body style="margin:0"><script>location.href="/reports/{p_abs.name}";</script></body></html>'
    )
