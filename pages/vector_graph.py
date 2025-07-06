import os
import streamlit as st

from langraph_utils import OUTPUT_DIR
from ui_helpers import render_sidebar

st.set_page_config(page_title="Vector Graph", page_icon="📊", layout="wide")
render_sidebar()

st.title("Interactive Graph Vector Database")
html_path = os.path.join(OUTPUT_DIR, "vectorstore_3d.html")
if not os.path.exists(html_path):
    st.warning(f"{html_path} not found. Run scripts/create_vectorplot.py to generate it.")
else:
    with open(html_path, "r") as f:
        html = f.read()
    st.components.v1.html(html, height=700, width=900, scrolling=False)
