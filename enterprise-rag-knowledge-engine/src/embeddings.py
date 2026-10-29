from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from src.config import EMBEDDING_MODEL_NAME, CHROMA_PERSIST_DIR

def build_vector_store(chunks, persist_directory=CHROMA_PERSIST_DIR):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=persist_directory)
    return vectorstore

def load_vector_store(persist_directory=CHROMA_PERSIST_DIR):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return Chroma(persist_directory=persist_directory, embedding_function=embeddings)
