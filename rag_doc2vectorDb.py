import os
import chromadb
import pandas as pd
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Load Sentence Transformer for Embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="forecast_rag")

class RAG_store_and_retrieve:

    def __init__(self, pdf_folder, forecast_csv):
        self.pdf_folder = pdf_folder
        self.forecast_csv = forecast_csv

    ### Function to Extract Text from PDFs ###
    def extract_text_from_pdf(self,pdf_path):
        """Extract text from a PDF file."""
        reader = PdfReader(pdf_path)
        text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
        return text

    ### Function to Store PDFs in ChromaDB ###
    def store_pdf_in_chromadb(self):
        """Embed and store PDF content in ChromaDB."""
        for filename in os.listdir(self.pdf_folder):
            print("Reading",filename)
            if filename.endswith(".pdf"):
                pdf_path = os.path.join(self.pdf_folder, filename)
                pdf_text = self.extract_text_from_pdf(pdf_path)
                embedding = embedding_model.encode(pdf_text).tolist()
                
                collection.add(
                    ids=[filename],
                    embeddings=[embedding],
                    metadatas=[{"text": pdf_text, "source": filename}]
                )
                print(f"Stored PDF: {filename}")
        print("Stored all PDFs data to ChromaDB!")

    ### Function to Store Forecasting Predictions in ChromaDB ###
    def store_forecasts_in_chromadb(self):
        """Embed and store demand forecasting predictions in ChromaDB."""
        df = pd.read_csv(self.forecast_csv)
        print("Read the forecast_csv file")

        for idx, row in df.iterrows():
            forecast_text = f"Forecast for {row['unique_id']} on {row['ds']}: {row['forecast']} units."
            embedding = embedding_model.encode(forecast_text).tolist()
            
            collection.add(
                ids=[f"forecast_{idx}"],
                embeddings=[embedding],
                metadatas=[{"text": forecast_text, "source": "forecast"}]
            )
        
        print("Stored forecasting data to ChromaDB!")

if __name__ == "__main__":
    # Load PDFs and Forecast Data into ChromaDB
    rag = RAG_store_and_retrieve(pdf_folder="./SupportingDocsOrg", forecast_csv="future_forecasts.csv")
    rag.store_pdf_in_chromadb()
    rag.store_forecasts_in_chromadb()