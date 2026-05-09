"""
RAG Pipeline - Complete orchestration of all RAG components
"""

from typing import List, Dict, Optional
import os

from app.rag.loaders import PDFLoader, WebLoader
from app.rag.chunking import TextChunker
from app.rag.embedding import EmbeddingModel
from app.rag.vectordb import ChromaStore
from app.rag.retrieval import Retriever, Reranker

class RAGPipeline:
    """
    Complete RAG Pipeline Orchestrator
    Handles: Load → Chunk → Embed → Store → Retrieve → Rerank
    """
    
    def __init__(self, 
                 vector_db_type: str = "chroma",
                 persist_directory: str = "./chroma_db",
                 chunk_size: int = 500,
                 chunk_overlap: int = 50):
        
        self.persist_directory = persist_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize components
        self.loaders = {
            'pdf': PDFLoader(),
            'web': WebLoader()
        }
        self.chunker = TextChunker(chunk_size, chunk_overlap)
        self.embedding = EmbeddingModel()
        
        # Initialize vector store
        if vector_db_type == "chroma":
            self.vector_store = ChromaStore(persist_directory)
        else:
            raise ValueError(f"Unknown vector DB type: {vector_db_type}")
        
        self.vector_store.create_collection()
        
        # Initialize retriever
        self.retriever = Retriever(self.vector_store, self.embedding)
        self.reranker = Reranker()
        
        print("="*50)
        print("✅ RAG Pipeline Initialized")
        print(f"   Vector DB: {vector_db_type}")
        print(f"   Embedding: {self.embedding.model_name}")
        print(f"   Chunk Size: {chunk_size}")
        print("="*50)
    
    def index_documents(self, source: str, **kwargs) -> int:
        """
        Index documents from various sources
        source: 'pdf', 'web', 'directory', 'sample'
        """
        print("\n📚 INDEXING DOCUMENTS")
        print("-"*40)
        
        # Step 1: Load documents
        documents = []
        
        if source == 'sample':
            documents = self._get_sample_documents()
        elif source == 'pdf':
            loader = self.loaders['pdf']
            if 'filepath' in kwargs:
                documents = loader.load(kwargs['filepath'])
            elif 'directory' in kwargs:
                documents = loader.load_directory(kwargs['directory'])
        elif source == 'web':
            loader = self.loaders['web']
            if 'url' in kwargs:
                documents = loader.load(kwargs['url'])
            elif 'urls' in kwargs:
                documents = loader.load_multiple(kwargs['urls'])
        else:
            raise ValueError(f"Unknown source: {source}")
        
        print(f"📄 Loaded {len(documents)} documents")
        
        # Step 2: Chunk documents
        all_chunks = []
        for doc in documents:
            chunks = self.chunker.chunk_document(doc, strategy="recursive")
            all_chunks.extend(chunks)
        
        print(f"✂️ Created {len(all_chunks)} chunks")
        
        # Step 3: Generate embeddings
        texts = [chunk["text"] for chunk in all_chunks]
        embeddings = self.embedding.encode(texts)
        
        print(f"🔄 Generated {len(embeddings)} embeddings")
        
        # Step 4: Store in vector database
        ids = [chunk["chunk_id"] for chunk in all_chunks]
        metadatas = [
            {"source": chunk["source"], "chunk_index": chunk["chunk_index"]} 
            for chunk in all_chunks
        ]
        
        self.vector_store.add_documents(ids, embeddings.tolist(), texts, metadatas)
        
        print(f"💾 Stored {self.vector_store.count()} chunks")
        print("="*50)
        
        return len(all_chunks)
    
    def retrieve(self, query: str, hops: int = 2, top_k: int = 5, rerank: bool = True) -> List[Dict]:
        """
        Retrieve relevant documents for a query
        """
        print(f"\n🔍 RETRIEVING: {query[:100]}")
        print(f"   Hops: {hops}, Top K: {top_k}")
        
        # Retrieve
        results = self.retriever.multi_hop(query, hops=hops, n_results_per_hop=top_k)
        
        print(f"   Retrieved {len(results)} documents")
        
        # Rerank if requested
        if rerank and results:
            results = self.reranker.rerank(query, results)
            print(f"   Reranked results")
        
        # Limit to top_k
        results = results[:top_k]
        
        return results
    
    def search(self, query: str, hops: int = 2, top_k: int = 5) -> List[Dict]:
        """
        Search with citations
        """
        results = self.retrieve(query, hops, top_k, rerank=True)
        
        # Add citations
        for i, result in enumerate(results, 1):
            result['citation'] = f"[{i}]"
        
        return results
    
    def get_status(self) -> Dict:
        """Get pipeline status"""
        return {
            "initialized": True,
            "documents_indexed": self.vector_store.count(),
            "embedding_model": self.embedding.model_name,
            "chunk_size": self.chunk_size,
            "vector_db_type": self.vector_store.__class__.__name__
        }
    
    def _get_sample_documents(self) -> List[Dict]:
        """Get sample documents for testing"""
        return [
            {
                "id": "python_intro",
                "text": "Python is a high-level, interpreted programming language created by Guido van Rossum. It emphasizes code readability and simplicity. Python is widely used for web development with Django and Flask, data science with pandas and numpy, and artificial intelligence with TensorFlow and PyTorch.",
                "source": "sample/python_docs"
            },
            {
                "id": "java_intro",
                "text": "Java is a class-based, object-oriented programming language designed for portability. Java applications run on the Java Virtual Machine (JVM). It is commonly used for enterprise applications, Android development, and large-scale systems. Spring Boot is the most popular framework for Java web development.",
                "source": "sample/java_docs"
            },
            {
                "id": "rl_intro",
                "text": "Reinforcement Learning (RL) is a machine learning paradigm where an agent learns to make decisions by interacting with an environment. The agent receives rewards or penalties and learns to maximize cumulative reward. RL is used in robotics, game playing (AlphaGo), and autonomous systems.",
                "source": "sample/ai_research"
            },
            {
                "id": "sl_intro",
                "text": "Supervised Learning (SL) is a machine learning approach where models are trained on labeled data. Each training example includes input features and the correct output label. SL is used for classification and regression tasks like spam detection, image recognition, and price prediction.",
                "source": "sample/ml_textbook"
            },
            {
                "id": "robotics_comparison",
                "text": "For robotics applications, reinforcement learning excels for sequential decision-making tasks like navigation and manipulation. Supervised learning is better for perception tasks like object detection and recognition. Hybrid approaches combining both are becoming popular in advanced robotics systems.",
                "source": "sample/robotics_journal"
            }
        ]