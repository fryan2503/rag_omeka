import streamlit as st


def render_sidebar():
    st.sidebar.markdown(
        """
        ### About the Author
        **Ryan Singh**
        [GitHub](https://github.com/fryan2503/rag_omeka)

        ### How to Use ART CHAT
        - Ask questions about artworks in our digital collection.
        - The AI will search our vector database and return relevant artworks and information.
        - Click on document expanders to view details and images.
        - For best results, use artist names, artwork titles, or art periods in your queries.

        ---
        Built for digital exploration and educational use by using the artwork at the **Richard and Carole Cocks Art Museum** (ART). This project is not affiliated with ART or any of its partners.
        Questions or feedback? reach out singhr7@miamioh.edu
        """
    )
