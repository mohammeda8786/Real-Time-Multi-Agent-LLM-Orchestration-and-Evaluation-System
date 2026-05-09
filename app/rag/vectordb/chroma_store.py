

from typing import List, Dict
import numpy as np

class ChromaStore:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize()
    
    def _initialize(self):
        try:
            import chromadb
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            print(f"✅ ChromaDB initialized at {self.persist_directory}")
        except ImportError as e:
            print(f"⚠️ ChromaDB error: {e}")
            self.client = None
    
    def create_collection(self, name: str = "knowledge_base"):
        if self.client:
            try:
                self.collection = self.client.create_collection(name=name)
                print(f"✅ Created collection: {name}")
            except:
                self.collection = self.client.get_collection(name)
                print(f"✅ Using existing collection: {name}")
    
    def add_documents(self, ids: List[str], embeddings: List, texts: List[str], metadatas: List[Dict]):
        if self.collection:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )
            print(f"✅ Added {len(texts)} documents")
    
    def search(self, query_embedding: List, n_results: int = 5) -> List[Dict]:
        """Search for similar documents - FIXED VERSION"""
        if not self.collection:
            return []
        
        # FIX: Ensure query_embedding is a flat list or 2D list
        # Convert to proper format for ChromaDB
        if isinstance(query_embedding, list):
            # Check if it's a list of lists (3 levels deep)
            if len(query_embedding) > 0 and isinstance(query_embedding[0], list):
                # Flatten if it's nested
                if isinstance(query_embedding[0][0], list):
                    # Too nested - take the first one
                    query_embedding = query_embedding[0]
                # Still 2D? Take first
                if len(query_embedding) > 0 and isinstance(query_embedding[0], list):
                    query_embedding = query_embedding[0]
        
        # Ensure it's a flat list of floats
        if isinstance(query_embedding, list) and len(query_embedding) > 0:
            if isinstance(query_embedding[0], list):
                query_embedding = query_embedding[0]
        
        # Now query
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],  # ChromaDB expects list of embeddings
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            
            formatted = []
            if results and results['documents'] and len(results['documents'][0]) > 0:
                for i in range(len(results['documents'][0])):
                    formatted.append({
                        "text": results['documents'][0][i],
                        "source": results['metadatas'][0][i].get("source", "unknown"),
                        "similarity_score": 1 - results['distances'][0][i],
                        "metadata": results['metadatas'][0][i]
                    })
            return formatted
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    def count(self) -> int:
        if self.collection:
            return self.collection.count()
        return 0
