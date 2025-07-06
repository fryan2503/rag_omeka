import numpy as np
import pandas as pd
import plotly.express as px
import umap
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

EMBEDDINGS_DIRECTORY = "./vstore"
OUTPUT_HTML = "vectorstore_3d.html"
OUTOUT_DIR = "./assets"


def load_vectorstore():
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS.load_local(
        EMBEDDINGS_DIRECTORY,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True,
    )


def extract_embeddings(vstore):
    num_vectors = vstore.index.ntotal
    return np.vstack([vstore.index.reconstruct(i) for i in range(num_vectors)])


def format_metadata(m):
    try:
        meta = m.metadata if hasattr(m, "metadata") else {}
        title = meta.get("title") or getattr(m, "title", "") or ""
        creator = meta.get("creator") or getattr(m, "creator", "") or ""
        date = meta.get("date") or getattr(m, "date", "") or ""
        medium = meta.get("medium") or getattr(m, "medium", "") or ""
        subject = meta.get("subject") or getattr(m, "subject", "") or ""
        desc = getattr(m, "page_content", "") or ""
    except Exception:
        title = creator = date = medium = subject = desc = ""
    short_desc = (desc[:80] + "...") if len(desc) > 80 else desc
    return f"<b>{title}</b><br>{creator}<br>{date}<br><i>{medium}</i><br>{subject}<br>{short_desc}"


def build_dataframe(vstore, embeddings):
    num_vectors = embeddings.shape[0]
    try:
        metadatas = [m for m in vstore.docstore._dict.values()]
        tooltips = [format_metadata(m) for m in metadatas]
    except Exception:
        tooltips = [f"Vector {i}" for i in range(num_vectors)]

    reducer = umap.UMAP(n_components=3, random_state=42)
    proj_3d = reducer.fit_transform(embeddings)

    df = pd.DataFrame({
        "x": proj_3d[:, 0],
        "y": proj_3d[:, 1],
        "z": proj_3d[:, 2],
        "tooltip": tooltips,
    })
    # Simple color scheme: gradient based on index
    df["color"] = np.arange(len(df))
    return df


def create_plot(df):
    fig = px.scatter_3d(
        df,
        x="x",
        y="y",
        z="z",
        color="color",
        color_continuous_scale="Turbo",
        hover_name="tooltip",
        title="FAISS Vector Store Visualization (UMAP 3D)",
        hover_data={"tooltip": True, "x": False, "y": False, "z": False, "color": False}
    )
    fig.update_traces(
    marker=dict(size=3, opacity=0.8),
    hovertemplate="%{customdata[0]}"
)
    fig.update_layout(
        height=700,
        width=900,
        scene=dict(
            bgcolor="black",
            xaxis=dict(visible=True, showgrid=True, gridcolor="gray"),
            yaxis=dict(visible=True, showgrid=True, gridcolor="gray"),
            zaxis=dict(visible=True, showgrid=True, gridcolor="gray"),
        ),
        margin=dict(l=0, r=0, b=0, t=30),
    )
    return fig


def main():
    vstore = load_vectorstore()
    embeddings = extract_embeddings(vstore)
    df = build_dataframe(vstore, embeddings)
    fig = create_plot(df)
    fig.write_html(
        f"{OUTOUT_DIR}/{OUTPUT_HTML}",
        include_plotlyjs="cdn",
    )
    print(f"Saved plot to {OUTOUT_DIR}/{OUTPUT_HTML}")


if __name__ == "__main__":
    main()