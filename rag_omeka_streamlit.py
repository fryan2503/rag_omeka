from dotenv import load_dotenv, find_dotenv
import streamlit as st
from langchain.vectorstores import FAISS
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_ollama import OllamaLLM, ChatOllama
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
)
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Load environment variables (if you have a .env file)
_ = load_dotenv(find_dotenv())

# 2. Constants
EMBEDDINGS_DIRECTORY = "scripts/vstore"  # same as in your notebook
EMBEDDING_MODEL_NAME = "nomic-embed-text-v1.5"

# 3. Load embedding model and FAISS vector store
embedding_model = NomicEmbeddings(model=EMBEDDING_MODEL_NAME, inference_mode="local")
vectorstore = FAISS.load_local(
    EMBEDDINGS_DIRECTORY, 
    embeddings=embedding_model, 
    allow_dangerous_deserialization=True
)

# 4. Create a retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 10, "fetch_k": 20})

# 5. Set up LLM and prompt template
local_llm = OllamaLLM(model="llama3")
llm = ChatOllama(model="llama3", temperature=0)
system_template = (
    "You are an expert art historian. "
    "You will be given a question and a set of documents.\n"
    "Your task is to provide a comprehensive answer based on the context provided.\n\n"
    "Question: {question}\n\n"
    "Context: {context}\n\n"
    "If the answer is not in the documents, say 'I don't know'."
)
prompt = ChatPromptTemplate(
    input_variables=["context", "question"],
    messages=[
        HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=["context", "question"],
                template=system_template
            )
        )
    ]
)

# 6. Build the RAG chain
rag_chain = (
    RunnablePassthrough.assign(
        context=lambda x: "\n\n".join(
            d.page_content for d in retriever.get_relevant_documents(x["question"])
        )
    )
    | prompt
    | llm
    | StrOutputParser()
)

# 7. Streamlit UI
st.set_page_config(page_title="Art History RAG Chat", layout="wide")
st.title("🖼️ Art History RAG Chat")

question = st.text_area("Enter your question about the artworks", height=150)

if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving documents and generating answer..."):
            response = rag_chain.invoke({"question": question})
        st.subheader("Answer")
        st.write(response)

        st.subheader("Retrieved Documents")
        docs = retriever.get_relevant_documents(question)
        for i, doc in enumerate(docs, start=1):
            st.markdown(f"**Doc {i}:**")
            st.write(doc.page_content)