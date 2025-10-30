# app.py
import streamlit as st
import pandas as pd
from agents.orchestrator_agent import OrchestratorAgent
from config.settings import DEBUG_MODE

st.set_page_config(page_title="Airline AI Agent - Smart Summarizer", layout="wide")
st.title("✈️ Airline AI Agent — Smart Summarizer")

st.markdown("""
Upload an airline CSV (or Excel) dataset. The AI agent will:
- infer column meanings (semantic inference),
- detect patterns & anomalies (statistical + isolation forest),
- generate a reader-friendly executive summary (LLM).
""")

u = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
if u:
    try:
        if u.name.endswith(".csv"):
            df = pd.read_csv(u)
        else:
            df = pd.read_excel(u)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    st.success(f"Loaded {len(df)} rows × {len(df.columns)} columns")
    if DEBUG_MODE:
        st.subheader("Data preview")
        st.dataframe(df.head())

    orchestrator = OrchestratorAgent()
    with st.spinner("Running AI agents…"):
        report, plots = orchestrator.run(df)

    st.subheader("Executive Summary")
    st.markdown(report, unsafe_allow_html=True)

    st.subheader("Visualizations")
    for fig in plots:
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Please upload a dataset to begin analysis.")
