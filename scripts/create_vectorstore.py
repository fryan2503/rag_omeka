EMBEDDINGS_DIRECTORY = './vstore'  # Directory to store embeddings

import json
from langchain.vectorstores import FAISS
from langchain_nomic.embeddings import NomicEmbeddings
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings

# Load the data
with open("/Users/ryan/PycharmProjects/pythonProject/College/projects/rag_omeka/data/extracted_data.json") as file:
    data = json.load(file)

# Load the embedding model
use_openai = True  # Set to True to use OpenAI embeddings, False for Nomic local

if use_openai:
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
else:
    embedding_model = NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local")

docs = []
for art in data:
    # Create a readable summary for embedding (page_content)
    page_content = (
        f"Title: {art.get('Title', 'Unknown')}\n"
        f"Creator: {art.get('Creator', 'Unknown')}\n"
        f"Date: {art.get('Date', 'Unknown')}\n"
        f"Medium: {art.get('Medium', 'Unknown')}\n"
        f"Format: {art.get('Format', 'Unknown')}\n"
        f"Subject: {art.get('Subject', 'Unknown')}\n"
        f"Description: {art.get('Description', 'No description provided.')}\n"
        f"Tags: {', '.join(art.get('Tags', []))}\n"
    )

    # Create metadata dictionary
    metadata = {
        "id": art.get("Identifier"),
        "title": art.get("Title"),
        "creator": art.get("Creator"),
        "date": art.get("Date"),
        "medium": art.get("Medium"),
        "format": art.get("Format"),
        "subject": art.get("Subject"),
        "donor": art.get("Donor"),
        "citation": art.get("Citation"),
        "tags": art.get("Tags", []),
        "collection_link": art.get("Collection Link", None)
    }

    docs.append(Document(page_content=page_content, metadata=metadata))

vectorstore = FAISS.from_documents(
    documents=docs,
    embedding=embedding_model
)

# Persist the FAISS index to disk
vectorstore.save_local(EMBEDDINGS_DIRECTORY)
print(f"Vector store created and saved to {EMBEDDINGS_DIRECTORY}")