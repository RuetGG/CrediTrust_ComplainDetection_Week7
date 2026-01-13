import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline 

FAISS_INDEX__PATH = 'vector_store/faiss_index.index'
METADATA_PATH = "vector_store/metadata.pkl"

index = faiss.read_index(FAISS_INDEX__PATH)

with open(METADATA_PATH, 'rb') as f:
    metadata = pickle.load(f)
    
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2"
)

llm = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=200,
)

PROMPT_TEMPLATE = """
You are a financial analyst assistant for CrediTrust.
Answer the user's question using ONLY the complaint excerpts below.
If the context does not contain enough information, say:
"I do not have enough information to answer this question."

Context:
{context}

Question:
{question}

Answer:
"""

def retrieve_chunks(question, k=3):
    query_embedding = embedding_model.encode([question]).astype("float32")
    distances, indices = index.search(query_embedding, k)
    
    results = []
    for idx in indices[0]:
        results.append({
            "text": metadata[idx]["text"],
            "product": metadata[idx]["product"],
            "complaint_id": metadata[idx]["complaint_id"]
        })
        
    return results

def rag_answer(question, k=5):
    retrieved = retrieve_chunks(question, k)
    
    context = "\n\n".join(
        [r['text'] for r in retrieved]
        )
    prompt = PROMPT_TEMPLATE.format(
        context = context,
        question = question
    )
    response = llm(prompt)[0]['generated_text']
    
    return response, retrieved

if __name__ == "__main__":
    
    questions = [
         "Why are customers complaining?",
    "Why do customers report?",
    "Why are money transfers delayed?",
    "What about about high personal loan fees?"
    ]
    evaluation = []
    
    for q in questions:
        answer, sources = rag_answer(q)
        evaluation.append({
            "Question": q,
            "Generated Answer": answer,
            "Retrieved Sources": [f'{s["product"]} {s["complaint_id"]}' for s in sources[:2]],
            "Quality Score (1-5)": "-",
            "Comments":"-"
        })
    for e in evaluation:
        print("Question:", e["Question"])
        print("Answer:", e["Generated Answer"])
        print("Sources:", e["Retrieved Sources"])
        print("Quality:", e["Quality Score (1-5)"])
        print("Comments:", e["Comments"])
        print("---")

