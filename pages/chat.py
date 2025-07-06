import streamlit as st

from langraph_utils import build_graph, fetch_image
from ui_helpers import render_sidebar

st.set_page_config(page_title="ART CHAT", page_icon="🎨", layout="wide")
render_sidebar()

graph = build_graph()

st.title("🎨 ART CHAT")

query = st.text_input("Ask about an artwork:")
if st.button("Search") and query.strip():
    with st.spinner("Retrieving..."):
        result = graph.invoke({"question": query})
    st.subheader("Answer")
    st.write(result["generation"].content)

    st.subheader("Retrieved Documents")
    for i, doc in enumerate(result["documents"], start=1):
        meta = doc.metadata
        link = meta.get("collection_link")
        img_url = fetch_image(link)
        with st.expander(f"Doc {i}: {meta.get('title', 'Untitled')}"):
            if img_url:
                st.image(img_url, width=200)
            st.write(doc.page_content)
            if link:
                st.markdown(f"[View on Omeka]({link})")
