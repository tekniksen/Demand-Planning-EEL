import chromadb
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="forecast_rag")
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# Load Sentence Transformer for Embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

class AnswerGenerator:

    def __init__(self):
        pass


    ### Function to Retrieve Relevant Information ###
    def retrieve_relevant_info(self, query, top_k=3):
        """Retrieve the most relevant forecasts & document texts for a query."""
        query_embedding = embedding_model.encode(query).tolist()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        return [res["text"] for res in results["metadatas"][0]]

 
    def answer_query(self,query):
        """Retrieve context from ChromaDB and generate a response using DistilBERT."""
        context_list = self.retrieve_relevant_info(query, top_k=3)
        context = " ".join(context_list) if context_list else ""

        if not context:
            return "No relevant information found. Try rephrasing your query."

        response = qa_pipeline(question=query, context=context)
        return response["answer"]
