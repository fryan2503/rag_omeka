import re
import requests
from bs4 import BeautifulSoup
import streamlit as st
from typing import Annotated, List, TypedDict

from langchain.vectorstores import FAISS
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# --- Vector store and model ----------------------------------------------------
EMBEDDINGS_DIRECTORY = "./vstore"
embedding_model = NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local")
vectorstore = FAISS.load_local(
    EMBEDDINGS_DIRECTORY,
    embeddings=embedding_model,
    allow_dangerous_deserialization=True,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
llm = ChatOllama(model="llama3", temperature=0)

rag_prompt = """You are an assistant for question-answering tasks. 

Here is the context to use to answer the question:

{context} 

Think carefully about the above context. 

Now, review the user question:

{question}

Provide an answer to this questions using only the above context. 
Do not make up any information that is not in the context.
If you cannot find an answer, say "I don't know".

Answer:"""

# --- Graph state ---------------------------------------------------------------
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    route: str
    documents: List[str]
    question: str
    generation: str
    keywords: str

# --- Helper functions ----------------------------------------------------------
def extract_keywords(state):
    """
    Extracts relevant search keywords from a user's question for use in a vector search engine.

    This function is designed for art museum curation contexts. It takes the user's question from the provided state,
    constructs a strict prompt for a language model to extract only the most relevant keywords (such as artist names,
    art styles, periods, materials, etc.), and returns them as a comma-separated string.

    Args:
        state (dict): A dictionary containing at least the key "question" with the user's input string.

    Returns:
        dict: A dictionary with a single key "keywords" mapping to the extracted, comma-separated keywords string.

    Side Effects:
        Prints debug information about the extraction process and the resulting keywords.
    """
    print("--- Extracting Keywords ---")
    question = state["question"]

    # Strict and focused prompt
    prompt = (
        "You are an art museum curator using a vector search engine to retrieve artworks from a digital collection. "
        "Extract only the most relevant search keywords from the user's question.\n\n"
        "Your output MUST be:\n"
        "- A strict list of relevant keywords or short phrases\n"
        "- Comma-separated\n"
        "No explanations, no extra text, no labels\n"
        " Only keywords related to paintings, artists, styles, periods, materials, etc.\n\n"
        "Correct format:\n"
        "Input: What are some oil paintings by Van Gogh from his time in Arles?\n"
        "Output:  Van Gogh, oil painting, Arles\n\n"
        "Do NOT output like this:\n"
        "- 'Here are your keywords: ...'\n"
        "- 'The relevant terms are...'\n"
        "- Any bullet points or sentences\n\n"
        f"User question: \"{question}\"\n"
        "Keywords (comma-separated):"
    )

    # LLM call
    keywords = llm.invoke([HumanMessage(content=prompt)]).content.strip()

    print(f"Extracted Keywords from the question: {keywords}")
    return {"keywords": keywords}


def retrieve(state): # Retriver
    """
    Retrieve documents from vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    keywords = state["keywords"]
    question = state["question"]

    documents_question = retriever.invoke(question)
    documents_keywords = retriever.invoke(keywords)

    # Deduplicate by page content
    seen_pages = set()
    unique_documents = []
    for doc in documents_question + documents_keywords:
        if doc.page_content not in seen_pages:
            unique_documents.append(doc)
            seen_pages.add(doc.page_content)

    print(f"Retrieved {len(unique_documents)} unique documents for question: {question} with keywords: {keywords}")
    return {"documents": unique_documents}

def generate(state):
    """

    Generate an AI response using the LLM, given the current state and retrieved documents.

    Args:
        state (dict): The current graph state, expected to contain 'messages' and 'documents'.

    Returns:
        dict: Updated state with the AI response appended to the 'messages' key.
    """
    print("-----GENERATE------")

    question = state["question"]
    documents = state["documents"]
    
    context = "\n".join([doc.page_content for doc in documents])  # adjust if needed
    prompt = rag_prompt.format(context=context, question=question)
    generation = llm.invoke([HumanMessage(content=prompt)])
    return {"generation": generation}

# --- Utility to fetch artwork image -------------------------------------------


def fetch_image(url: str | None):
    """Try retrieving an <img> from the Omeka item page."""
    if not url:
        return None
    try:
        html = requests.get(url, timeout=10).text
        soup = BeautifulSoup(html, "html.parser")
        img = soup.find("img")
        return img["src"] if img and img.get("src") else None
    except Exception:
        return None

# --- Build LangGraph -----------------------------------------------------------
builder = StateGraph(State)
builder.add_node("extract_keywords", extract_keywords)
builder.add_node("retrieve", retrieve)
builder.add_node("generate", generate)
builder.add_edge(START, "extract_keywords")
builder.add_edge("extract_keywords", "retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", END)
graph = builder.compile()

# --- Streamlit UI --------------------------------------------------------------
st.set_page_config(page_title="Art Museum RAG", page_icon="🎨", layout="wide")
st.markdown(
    """
    <style>
    .main { background-color: #f5f5f5; }
    .stButton button { background-color:#4CAF50; color:white; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("🎨 Art Museum LangGraph Chat")

query = st.text_input("Ask about an artwork:")
submit = st.button("Search")

if submit and query.strip():
    with st.spinner("Retrieving..."):
        result = graph.invoke({"question": query})
    st.subheader("Answer")
    st.write(result["generation"].content)

    st.subheader("Retrieved Documents")
    for i, doc in enumerate(result["documents"], start=1):
        meta = doc.metadata
        link = meta.get("collection_link", None)
        print(link)

        img_url = fetch_image(link)
        with st.expander(f"Doc {i}: {meta.get('title', 'Untitled')}"):
            if img_url:
                st.image(img_url, use_column_width=True)
            st.write(doc.page_content)
            if link:
                st.markdown(f"[View on Omeka]({link})")
