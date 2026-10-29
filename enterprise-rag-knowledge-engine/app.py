import os
import streamlit as st
from src.data_loader import load_and_split_document
from src.embeddings import build_vector_store
from src.model import create_rag_chain

st.set_page_config(page_title="Enterprise RAG Engine", layout="wide")
st.title("⚡ Enterprise LLM RAG & Knowledge Engine")

uploaded_file = st.sidebar.file_uploader("Upload Knowledge Document (PDF/TXT)", type=["pdf", "txt"])

if uploaded_file:
    file_path = f"./data/raw/{uploaded_file.name}"
    os.makedirs("./data/raw", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    chunks = load_and_split_document(file_path)
    vectorstore = build_vector_store(chunks)
    st.sidebar.success(f"Successfully indexed {len(chunks)} document chunks!")

    query = st.text_input("Ask a question about your knowledge base:")
    if query:
        qa_chain = create_rag_chain(vectorstore)
        response = qa_chain.invoke({"query": query})
        
        st.write("### Answer:")
        st.write(response["result"])

        with st.expander("Retrieved Source Chunks"):
            for doc in response["source_documents"]:
                st.info(doc.page_content)
