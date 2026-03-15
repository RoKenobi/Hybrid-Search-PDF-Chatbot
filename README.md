# 📄 RAG Document Assistant

A production-grade Retrieval-Augmented Generation (RAG) system that answers questions from PDFs with grounded responses and full citations. Built to solve enterprise knowledge retrieval challenges using Hybrid Search.

## 🚀 Live Demo

[Insert Streamlit Cloud Link Here]

## ✨ Features

- Hybrid Search (BM25 + Vector) for high-precision retrieval
- Grounded responses with strict hallucination guardrails
- Inline citations with source file and page number
- Interactive Streamlit UI with chat history
- Document upload and automatic ingestion pipeline
- Session analytics and error logging

## 🛠️ Tech Stack

- Language: Python 3.12
- LLM: Google Gemini 2.5 Flash
- Embeddings: Sentence Transformers (all-MiniLM-L6-v2)
- Vector DB: ChromaDB
- Framework: LangChain
- UI: Streamlit
- Retrieval: BM25 + Vector Ensemble

## 📦 Installation

1. Clone the repository

2. Create a virtual environment

    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install dependencies

    pip install -r requirements.txt

4. Set up environment variables

    Create a .env file in the root directory and add your API key:

    GOOGLE_API_KEY=your_api_key_here

## 🚀 Usage

1. Ingest Documents

    Run the ingestion script to process PDFs in the data/documents folder:

    python ingestion/ingest.py

2. Launch the App

    Start the Streamlit interface:

    streamlit run app/streamlit_app.py

3. Interact

    Upload PDFs via the UI or ask questions about pre-loaded documents.

## 📂 Project Structure

rag_gemini_assistant/
    ├── app/
    │   └── streamlit_app.py      # Main UI application
    ├── ingestion/
    │   └── ingest.py             # Document loading and embedding
    ├── retrieval/
    │   └── query.py              # Hybrid search and LLM logic
    ├── utils/
    │   └── helpers.py            # API key management
    ├── data/
    │   └── documents/            # PDF storage
    ├── .env                      # Environment variables (gitignored)
    ├── .gitignore                # Git ignore rules
    ├── requirements.txt          # Dependencies
    └── README.md                 # Project documentation

## 🔍 Key Engineering Decisions

- Hybrid Search: Pure vector search often misses exact keywords. We combine BM25 (keyword) and Vector (semantic) search with a 40/60 weight ensemble for optimal recall.
- Citation Enforcement: The system prompt mandates inline citations. If context is missing, the model is instructed to decline answering rather than hallucinate.
- Metadata Preservation: Chunk metadata (source file, page number) is preserved during ingestion to enable precise source tracking in the UI.

## 📄 License

MIT License
