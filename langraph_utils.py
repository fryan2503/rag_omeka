import os
import requests
from bs4 import BeautifulSoup
from typing import Annotated, List, TypedDict

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

EMBEDDINGS_DIRECTORY = "./vstore"
OUTPUT_DIR = "./assets"

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.load_local(
    EMBEDDINGS_DIRECTORY,
    embeddings=embedding_model,
    allow_dangerous_deserialization=True,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})


# LLM setup --------------------------------------------------------------------

def choose_cloud_local(model_name: str):
    global llm
    if model_name == "local":
        llm = ChatOllama(model="llama3", temperature=0)
    elif model_name == "cloud":
        llm = ChatOpenAI(model="gpt-4", temperature=0)
    else:
        raise ValueError(f"Unsupported input {model_name}")

choose_cloud_local("cloud")  # Change to "local" for local model


rag_prompt = """You are an assistant for question-answering tasks.

Here is the context to use to answer the question:

{context}

Think carefully about the above context.

Now, review the user question:

{question}

Provide an answer to this questions using only the above context.
Do not make up any information that is not in the context.
If you cannot find an answer, say \"I don't know\".

Answer:"""


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    route: str
    documents: List[str]
    question: str
    generation: str
    keywords: str


# Helper functions -------------------------------------------------------------

def extract_keywords(state):
    print("--- Extracting Keywords ---")
    question = state["question"]

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

    keywords = llm.invoke([HumanMessage(content=prompt)]).content.strip()
    print(f"Extracted Keywords from the question: {keywords}")
    return {"keywords": keywords}


def retrieve(state):
    print("---RETRIEVE---")
    keywords = state["keywords"]
    question = state["question"]

    documents_question = retriever.invoke(question)
    documents_keywords = retriever.invoke(keywords)

    seen_pages = set()
    unique_documents = []
    for doc in documents_question + documents_keywords:
        if doc.page_content not in seen_pages:
            unique_documents.append(doc)
            seen_pages.add(doc.page_content)

    print(
        f"Retrieved {len(unique_documents)} unique documents for question: {question} with keywords: {keywords}"
    )
    return {"documents": unique_documents}


def generate(state):
    print("-----GENERATE------")

    question = state["question"]
    documents = state["documents"]

    context = "\n".join([doc.page_content for doc in documents])
    prompt = rag_prompt.format(context=context, question=question)
    generation = llm.invoke([HumanMessage(content=prompt)])
    return {"generation": generation}


# Utility to fetch artwork image ------------------------------------------------

def fetch_image(url: str | None):
    if not url:
        return None
    try:
        html = requests.get(url, timeout=10).text
        soup = BeautifulSoup(html, "html.parser")
        img = soup.find("img")
        return img["src"] if img and img.get("src") else None
    except Exception:
        return None


# Build LangGraph --------------------------------------------------------------

def build_graph():
    builder = StateGraph(State)
    builder.add_node("extract_keywords", extract_keywords)
    builder.add_node("retrieve", retrieve)
    builder.add_node("generate", generate)
    builder.add_edge(START, "extract_keywords")
    builder.add_edge("extract_keywords", "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)
    return builder.compile()
