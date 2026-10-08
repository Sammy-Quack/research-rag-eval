import streamlit as st

from app.dashboard_tab import render_dashboard_tab


st.set_page_config(page_title="RAG evaluation dashboard", layout="wide")
render_dashboard_tab()
