import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from app.dashboard_tab import render_dashboard_tab
from app.query_tab import render_query_tab


st.set_page_config(page_title="Autonomous Agents RAG", layout="wide")

query_tab, dashboard_tab = st.tabs(["Ask a question", "Evaluation dashboard"])

with query_tab:
    render_query_tab()

with dashboard_tab:
    render_dashboard_tab()
