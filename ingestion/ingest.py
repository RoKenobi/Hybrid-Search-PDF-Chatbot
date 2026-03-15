import os
from langchain_community.document_loaders import PyPDFLoader
#from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def load_documents(folder_path):
    """Finds all PDFs in the folder and extracts their text."""
    documents = []
    
    # Check if the folder exists first
    if not os.path.exists(folder_path):
        print(f"Error: The folder {folder_path} does not exist.")
        return documents

    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            print(f"Loading: {file}")
            loader = PyPDFLoader(os.path.join(folder_path, file))
            docs = loader.load()
            documents.extend(docs)
    
    return documents

def split_documents(documents):
    """Splits large documents into smaller, overlapping chunks."""
    # chunk_size: How many characters per chunk
    # chunk_overlap: Keeps context between chunks so sentences don't get cut in half
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} total chunks.")
    return chunks


def create_vector_store(chunks):
    """Turns text chunks into numbers and saves them to a local database."""
    print("Initializing embedding model...")
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    import os

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db_path = "vector_db"

    # 🚨 THE FIX: Safely clear out the old database so we don't duplicate chunks
    if os.path.exists(db_path):
        print("Clearing old database to prevent duplicates...")
        # Load the old collection and delete it safely to avoid Windows file locks
        old_db = Chroma(persist_directory=db_path, embedding_function=embeddings)
        old_db.delete_collection()

    print("Creating fresh vector database...")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_path
    )
    
    print("Vector store created and saved to 'vector_db' folder.")
    return vectordb

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    DOCS_PATH = os.path.join(current_dir, "..", "data", "documents")
    
    # 1. Load
    raw_docs = load_documents(DOCS_PATH)
    
    # 2. Split
    if raw_docs:
        final_chunks = split_documents(raw_docs)
        
        # 3. Embed & Store (The new part!)
        create_vector_store(final_chunks)
        print("Step 4 complete: Your knowledge base is ready.")
    else:
        print("No PDFs found. Add a file to test.")