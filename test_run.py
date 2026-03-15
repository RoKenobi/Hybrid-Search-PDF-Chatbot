from retrieval.query import ask_question

# Ask something that is actually inside the PDF you uploaded!
question = "What is the main summary of this document?"

print("--- Querying Gemini ---")
answer, sources = ask_question(question)

print(f"\nAI ANSWER:\n{answer}")
print("\n--- SOURCES USED ---")
for doc in sources:
    print(f"- {doc.metadata.get('source')} (Page {doc.metadata.get('page')})")