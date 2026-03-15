import sys
import os
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

# Ensure Python can find your retrieval code
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from retrieval.query import ask_question
from utils.helpers import load_api_key

def run_evaluation():
    print("🚀 Starting RAG Pipeline Evaluation...")
    
    # 1. Define your test questions and the *actual* correct answers based on your PDFs
    test_data = [
        {
            "question": "What is the main topic of the IoT document?",
            "ground_truth": "The document discusses applying Self-Sovereign Identity (SSI) to the constrained Internet of Things (IoT)."
        },
        {
            "question": "What role did the candidate hold at Game Plus?",
            "ground_truth": "The candidate was an AI developer/Co-Founder at Game Plus."
        }
    ]
    
    answers = []
    contexts = []
    
    # 2. Run your pipeline to generate answers to grade
    for item in test_data:
        print(f"Asking: {item['question']}")
        ans, docs = ask_question(item['question'])
        answers.append(ans)
        # Ragas expects contexts as a list of strings
        contexts.append([d.page_content for d in docs])
        
    # 3. Prepare data for RAGAS
    data = {
        "question": [item["question"] for item in test_data],
        "answer": answers,
        "contexts": contexts,
        "ground_truth": [item["ground_truth"] for item in test_data]
    }
    dataset = Dataset.from_dict(data)
    
    # 4. Configure Gemini as the judge model
    api_key = load_api_key()
    judge_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
    judge_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # 5. Run the Evaluation
    print("🧠 Grading answers for Faithfulness (No Hallucinations) and Relevancy...")
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=judge_llm,
        embeddings=judge_embeddings
    )
    
    # 6. Output the final scorecard!
    print("\n✅ === EVALUATION SCORECARD ===")
    df = result.to_pandas()
    print(df[['question', 'faithfulness', 'answer_relevancy']])
    print("\nOverall Scores:")
    print(result)

if __name__ == "__main__":
    run_evaluation()