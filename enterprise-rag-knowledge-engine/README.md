# Enterprise LLM RAG & Knowledge Engine

A production-grade Retrieval-Augmented Generation (RAG) system built with Python, LangChain, Hugging Face, and ChromaDB.

## Features
- Modular document ingestion for PDF/TXT files.
- Vector semantic indexing using SentenceTransformers.
- Context-aware Q&A interface using Streamlit.
- Docker container support for easy deployment.

## Installation & Setup
```bash
git clone https://github.com/Yuvi-cloud/enterprise-rag-knowledge-engine.git
cd enterprise-rag-knowledge-engine
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
