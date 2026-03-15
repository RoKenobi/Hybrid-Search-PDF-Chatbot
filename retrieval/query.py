from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from utils.helpers import load_api_key

def load_vector_db():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectordb = Chroma(
        persist_directory="vector_db",
        embedding_function=embeddings
    )
    return vectordb

def retrieve_docs(query, k=8):
    """
    ADVANCED HYBRID SEARCH: Combines Vector Search (meaning) with BM25 (exact keywords).
    Features manual deduplication and metadata injection for exact file matching.
    """
    vectordb = load_vector_db()
    
    # 1. Vector Search (Semantic)
    vector_retriever = vectordb.as_retriever(search_kwargs={"k": k})
    vector_docs = vector_retriever.invoke(query)
    
    # 2. BM25 Search (Keyword) - WITH METADATA BLINDSPOT FIX
    db_data = vectordb.get(include=['documents', 'metadatas'])
    
    # We inject the filename directly into the text the BM25 algorithm reads.
    # Now if a user searches for "iot14", it actually finds it.
    all_docs = [
        Document(
            page_content=f"File: {meta.get('source', 'Unknown')}\n{doc}", 
            metadata=meta
        ) 
        for doc, meta in zip(db_data['documents'], db_data['metadatas'])
    ]
    
    bm25_retriever = BM25Retriever.from_documents(all_docs)
    bm25_retriever.k = k
    bm25_docs = bm25_retriever.invoke(query)
    
    # 3. Manual Ensemble: Combine and deduplicate
    all_results = []
    seen_content = set()
    
    # Add vector results first (prioritize meaning)
    for doc in vector_docs:
        if doc.page_content not in seen_content:
            all_results.append(doc)
            seen_content.add(doc.page_content)
    
    # Add BM25 results (exact keyword matches)
    for doc in bm25_docs:
        if doc.page_content not in seen_content:
            all_results.append(doc)
            seen_content.add(doc.page_content)
    
    # Return exactly the number of chunks we asked for
    return all_results[:k]

def format_context_with_metadata(docs):
    """
    Format chunks with source metadata so the model can cite correctly.
    """
    formatted_chunks = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get('source', 'unknown.pdf')
        page = doc.metadata.get('page', 'N/A')
        
        # Extract just the filename if the path is long
        if '\\' in source:
            source = source.split('\\')[-1]
        elif '/' in source:
            source = source.split('/')[-1]
            
        chunk_text = f"""
[Chunk {i+1}]
Source: {source} | Page: {page}
Content:
{doc.page_content}
"""
        formatted_chunks.append(chunk_text)
    
    return "\n\n".join(formatted_chunks)

def create_llm():
    api_key = load_api_key()
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.2,
        max_output_tokens=2048,
    )
    return llm

def build_system_prompt():
    """
    System message with explicit guardrails for hallucination prevention and proper synthesis.
    """
    return SystemMessage(content="""
You are a precise Document Assistant. Follow these rules STRICTLY:

1. Answer ONLY using information from the provided Context chunks.
2. If the answer is not in the Context, reply EXACTLY: "I'm sorry, but I cannot find the answer to this in the provided documents."
3. Provide a single inline citation at the end of a complete thought or paragraph, rather than after every single sentence: [Source: filename, Page X]
4. If multiple chunks support a claim, cite all: [Source: file1.pdf, Page 3; file2.pdf, Page 5]
5. Do NOT hallucinate, infer, or use outside knowledge.
6. Quote verbatim when precision is critical; otherwise paraphrase accurately.
7. Keep answers concise and well-structured.
8. When asked to summarize, list, or extract sections like 'Work Experience', synthesize all provided chunks to ensure no items are missed.
""")

def ask_question(question):
    """
    Full RAG flow: Search DB -> Build Prompt -> Ask Gemini -> Return Answer + Sources.
    """
    # Step 1: Retrieve relevant chunks (Bottleneck fixed: k=8)
    docs = retrieve_docs(question, k=8)
    
    # Step 2: Format context WITH metadata for citations
    context = format_context_with_metadata(docs)
    
    # Step 3: Create LLM with system message
    llm = create_llm()
    system_prompt = build_system_prompt()
    
    # Step 4: Build user message
    user_prompt = HumanMessage(content=f"""
CONTEXT:
{context}

QUESTION: 
{question}

Provide your answer with inline citations for every claim.
""")
    
    # Step 5: Invoke and return
    response = llm.invoke([system_prompt, user_prompt])
    
    return response.content, docs