import streamlit as st
from ui_helpers import render_sidebar

st.set_page_config(page_title="RAG Omeka", page_icon="🎨", layout="wide")

st.title("RAG Omeka Demo")
render_sidebar()

st.write("Select a page from the sidebar to get started.")
