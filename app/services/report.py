from __future__ import annotations
from pathlib import Path
from datetime import datetime
from jinja2 import Environment, FileSystemLoader, select_autoescape
from app.config import settings
from .llm import analyze_clusters

def render_report(
    out_dir: Path,
    template_dir: Path,
    title: str,
    profile: dict,
    train_result: dict,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        autoescape=select_autoescape(["html"])
    )
    template = env.get_template("report.html")

    analysis_html = ""
    if train_result.get("cluster_profiles"):
        # Call LLM to analyze
        # We need original columns, we can infer from keys of first profile
        cols = []
        if train_result["cluster_profiles"]:
             first_key = next(iter(train_result["cluster_profiles"]))
             cols = list(train_result["cluster_profiles"][first_key].keys())
        
        analysis_html = analyze_clusters(settings.gemini_api_key, train_result["cluster_profiles"], cols)

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    html = template.render(
        title=title,
        generated_at=now,
        profile=profile,
        train=train_result,
        cluster_analysis=analysis_html
    )

    filename = f"report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.html"
    out_path = out_dir / filename
    out_path.write_text(html, encoding="utf-8")
    return out_path
