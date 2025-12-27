import os
import requests
import streamlit as st
import pandas as pd

st.set_page_config(page_title="ML Report Bot", page_icon="📊", layout="centered")
API_BASE = os.getenv("API_BASE", "http://localhost:8000")

st.title("📊 ML Report Bot (Web)")
st.caption("Upload CSV → Auto-detect target → Train/Analyze → Download Report / Open in New Tab")

uploaded = st.file_uploader("Upload your CSV file", type=["csv"])
if not uploaded:
    st.stop()

try:
    df_preview = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

columns = df_preview.columns.tolist()
sample_rows = df_preview.head(8).fillna("").to_dict(orient="records")

suggested_target = None
suggested_model = "RandomForest"
rationale = ""

try:
    sug = requests.post(
        f"{API_BASE}/suggest-target",
        json={"columns": columns, "sample_rows": sample_rows},
        timeout=30
    )
    if sug.status_code == 200:
        data = sug.json() or {}
        suggested_target = data.get("target")
        suggested_model = data.get("suggested_model", "RandomForest")
        rationale = data.get("rationale", "")
except Exception:
    pass

if suggested_target:
    st.info(f"💡 **AI Suggestion:**\n\n- **Target:** `{suggested_target}`\n- **Model:** `{suggested_model}`\n\n_{rationale}_")


target_options = ["(Clustering - No Target)"] + columns


real_target_index = 0
if suggested_target and suggested_target in columns:
    # +1 because of the new Clustering option at index 0
    real_target_index = columns.index(suggested_target) + 1

st.subheader("1) Select Target and Model")
col_target, col_model = st.columns(2)

selected_ui_target = None

with col_target:
    selected_ui_target = st.selectbox("Select Target Column", options=target_options, index=real_target_index)

is_clustering = (selected_ui_target == "(Clustering - No Target)")
target_payload = "None" if is_clustering else selected_ui_target

with col_model:
    if is_clustering:
        st.info("ℹ️ Select K-Means (3 Groups) or DBSCAN (Density).")
        model_type = st.selectbox("Select Algorithm", ["KMeans", "DBSCAN"])
    else:
        model_options = ["RandomForest", "Linear", "DecisionTree", "XGBoost"]
        try:
            model_index = model_options.index(suggested_model)
        except ValueError:
            model_index = 0
        model_type = st.selectbox("Select Model", options=model_options, index=model_index)

st.subheader("2) Run Training")
run_btn = st.button("🚀 Analyze / Train", use_container_width=True)

st.subheader("CSV Preview")
st.dataframe(df_preview.head(20), use_container_width=True)

if not run_btn:
    st.stop()

with st.spinner("Training model, generating graphs..."):
    files = {"file": (uploaded.name, uploaded.getvalue(), "text/csv")}
    resp = requests.post(
        f"{API_BASE}/train",
        params={"target": target_payload, "model_type": model_type},
        files=files,
        timeout=180,
    )

if resp.status_code != 200:
    st.error(f"API Error ({resp.status_code}): {resp.text}")
    st.stop()

data = resp.json()
st.success("✅ Operation Complete!")

st.subheader("Result")
st.write("**Task Type:**", data["task_type"])
st.write("**Model:**", data["model_name"])
st.write("**Metrics:**")
st.json(data["metrics"])

report_path = data["report_path"]
filename = report_path.split("/")[-1]

st.subheader("3) Report")


dl = requests.get(f"{API_BASE}/download-report", params={"path": report_path}, timeout=60)
if dl.status_code != 200:
    st.error(f"Could not download report ({dl.status_code}): {dl.text}")
    st.stop()

st.download_button(
    "📄 Download HTML Report",
    data=dl.content,
    file_name=filename,
    mime="text/html",
    use_container_width=True
)


open_url = f"{API_BASE}/reports/{filename}"
st.link_button("🌐 Open Report in New Tab", open_url, use_container_width=True)
