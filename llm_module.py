from transformers import pipeline
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
import json

class ForecastLLM:
    def __init__(self, model_name="mistral-7b-instruct", use_rag=False):
        """
        Initializes the LLM pipeline.
        :param model_name: Name of the Hugging Face model to use.
        :param use_rag: Boolean to enable RAG-based retrieval.
        """
        self.use_rag = use_rag
        self.pipeline = pipeline("text-generation", model=model_name)
        
        if use_rag:
            self.embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            self.vector_store = FAISS.load_local("vector_db", self.embedder)
            self.qa_chain = RetrievalQA.from_chain_type(llm=HuggingFacePipeline(self.pipeline), retriever=self.vector_store.as_retriever())

    def query_llm(self, user_query, predictions_json=None):
        """
        Query the LLM with or without RAG.
        :param user_query: The query from the user.
        :param predictions_json: JSON string containing forecast data (optional).
        :return: AI-generated response.
        """
        context = ""
        if predictions_json:
            context = f"Here are the prediction results: {predictions_json}.\n"
        
        if self.use_rag:
            response = self.qa_chain.run(context + user_query)
        else:
            response = self.pipeline(f"{context} Question: {user_query}", max_length=200, truncation=True)[0]['generated_text']
        
        return response
