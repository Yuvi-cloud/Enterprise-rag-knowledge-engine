# Enterprise RAG Knowledge Engine

## System Architecture
1. **Document Ingestion**: Supports PDF and TXT files.
2. **Text Splitting**: Uses `RecursiveCharacterTextSplitter` with chunk size of 500 and overlap of 50.
3. **Embeddings & Vector Store**: `sentence-transformers/all-MiniLM-L6-v2` embeddings stored in `ChromaDB`.
4. **LLM Retrieval**: HuggingFaceHub or Local Transformer model integration via `LangChain`.
