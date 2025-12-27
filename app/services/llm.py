from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

DEFAULT_GEMINI_MODEL = "gemini-1.5-flash"

@dataclass
class LLMSuggestion:
    target: Optional[str]
    suggested_model: str = "RandomForest"
    rationale: str = ""

def fallback_suggest(columns: List[str]) -> LLMSuggestion:
    """

    """
    preferred = [
        "target", "label", "y", "class", "outcome",
        "price", "salary", "amount", "score"
    ]
    cols_lower = {c.lower(): c for c in columns}
    for p in preferred:
        if p in cols_lower:
            return LLMSuggestion(
                target=cols_lower[p], 
                suggested_model="RandomForest",
                rationale=f"fallback matched '{p}'"
            )
    return LLMSuggestion(target=None, suggested_model="RandomForest", rationale="fallback could not decide")

def suggest_target(
    api_key: str,
    columns: List[str],
    sample_rows: Optional[List[Dict[str, Any]]] = None,
    model_name: str = DEFAULT_GEMINI_MODEL,
) -> LLMSuggestion:
    """

    """
    if not api_key:
        return fallback_suggest(columns)

    try:
        import google.generativeai as genai
    except Exception:
        return fallback_suggest(columns)

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    sample_rows = (sample_rows or [])[:8]

    prompt = f"""
You are helping a machine learning app. The user uploads a CSV.
1. Pick the most likely target (label) column for training.
2. Suggest the best model among ["RandomForest", "Linear", "DecisionTree", "XGBoost"].

Rules:
- Target should be the column to predict. Avoid IDs.
- Model Selection:
  - "Linear": for simple linear relationships (e.g. price prediction based on size).
  - "DecisionTree": for simple rules or if explainability is key.
  - "RandomForest": for complex, noisy data (default choice).
  - "XGBoost": for high-performance needs or winning competitions (complex data).
- Return ONLY JSON: {{"target":"<col or null>", "suggested_model":"<model>", "rationale":"..."}}

Columns: {columns}

Sample rows (up to 8):
{sample_rows}
""".strip()

    try:
        resp = model.generate_content(prompt)
        text = (resp.text or "").strip()
    except Exception:
        return fallback_suggest(columns)

    import json
    try:
        if text.startswith("```"):
            text = text.strip().strip("`")
            text = text.replace("json", "", 1).strip()
        data = json.loads(text)
        target = data.get("target", None)
        suggested_model = data.get("suggested_model", "RandomForest")
        rationale = str(data.get("rationale", "")).strip() or "LLM rationale not provided."

        if target is not None and target not in columns:
            return LLMSuggestion(target=None, suggested_model="RandomForest", rationale="LLM target not in columns")

        # Validate model selection
        if suggested_model not in ["RandomForest", "Linear", "DecisionTree", "XGBoost"]:
            suggested_model = "RandomForest"

        return LLMSuggestion(target=target, suggested_model=suggested_model, rationale=rationale)
    except Exception:
        return LLMSuggestion(target=target, suggested_model=suggested_model, rationale=rationale)
    except Exception:
        return LLMSuggestion(target=None, suggested_model="RandomForest", rationale="LLM output invalid JSON")

def analyze_clusters(api_key: str, cluster_profiles: Dict[int, Dict[str, float]], columns: List[str]) -> str:
    """

    """
    if not api_key:
        return "<p><i>API key missing. Cannot generate cluster analysis.</i></p>"

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(DEFAULT_GEMINI_MODEL)
        
        # Limit profiles to top 8 clusters to save tokens
        profiles_text = ""
        for cid, stats in list(cluster_profiles.items())[:8]:
            stats_str = ", ".join([f"{k}: {v:.2f}" for k, v in stats.items()])
            profiles_text += f"\n- Cluster {cid}: {stats_str}"

        prompt = f"""
You are a Senior Data Scientist analyzing customer segments (clusters).
Based on the following mean values for each cluster, write a brief, professional HTML formatted summary.

Cluster Profiles:
{profiles_text}

Instructions:
1. For each cluster, give it a **Professional Name** (e.g. "High-Value Customers", "Churn Risk Group", "Early Adopters").
2. Describe key characteristics in 1-2 sentences.
3. Use HTML tags like <h3>, <b>, <ul> for readability.
4. Keep it concise.
""".strip()
        
        resp = model.generate_content(prompt)
        return resp.text or "<p>No analysis generated.</p>"
    except Exception as e:
        return f"<p>Error generating analysis: {e}</p>"
