import chromadb
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="forecast_rag")

# Load Sentence Transformer for Embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

### Function to Retrieve Relevant Information ###
def retrieve_relevant_info(query, top_k=3):
    """Retrieve the most relevant forecasts & document texts for a query."""
    query_embedding = embedding_model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return [res["text"] for res in results["metadatas"][0]]

### Function to Answer Queries Using DistilBERT ###
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

def answer_query(query):
    """Retrieve context from ChromaDB and generate a response using DistilBERT."""
    context = " ".join(retrieve_relevant_info(query,top_k=3))
    
    if not context:
        return "No relevant information found."

    response = qa_pipeline(question=query, context=context)
    return response["answer"]



# Example Query
query = "What is the Forecast for Indonesia on 2025-09-01?"
response = answer_query(query)

print("🔹 User Query:", query)
print("🔹 AI Response:", response)