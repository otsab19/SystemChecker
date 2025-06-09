import os
import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from typing import List, Dict, Any
from config.settings import settings


class VectorStoreManager:
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            google_api_key=settings.GEMINI_API_KEY
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
        )

        # Initialize ChromaDB
        os.makedirs(settings.VECTOR_DB_PATH, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=settings.VECTOR_DB_PATH,
            settings=ChromaSettings(anonymized_telemetry=False)
        )

        # Get or create collection
        try:
            self.collection = self.client.get_collection(settings.COLLECTION_NAME)
        except:
            self.collection = self.client.create_collection(
                name=settings.COLLECTION_NAME,
                metadata={"description": "System information for RAG"}
            )

    def add_system_data(self, system_data: str, metadata: Dict[str, Any] = None):
        """Add system data to vector store"""
        # Split text into chunks
        chunks = self.text_splitter.split_text(system_data)

        if not chunks:
            return

        # Generate embeddings
        embeddings = self.embeddings.embed_documents(chunks)

        # Prepare data for ChromaDB
        ids = [f"chunk_{i}_{metadata.get('timestamp', 'unknown')}" for i in range(len(chunks))]
        metadatas = [metadata or {} for _ in chunks]

        # Add to collection
        self.collection.add(
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
            ids=ids
        )

    def query_similar(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Query similar documents from vector store"""
        # Generate query embedding
        query_embedding = self.embeddings.embed_query(query)

        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

        # Format results
        formatted_results = []
        for i in range(len(results["documents"][0])):
            formatted_results.append({
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i]
            })

        return formatted_results

    def clear_old_data(self, days_old: int = 7):
        """Clear data older than specified days"""
        # This is a simplified implementation
        # In production, you'd want more sophisticated cleanup
        pass