"""
Embedding Model - Convert text to vectors
"""

from typing import List, Union
import numpy as np

class EmbeddingModel:
    """Generate embeddings for text chunks"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.dimension = 384
        self._load_model()
    
    def _load_model(self):
        """Load the embedding model with validation and timing"""
        import time
        start_time = time.time()
        
        try:
            from sentence_transformers import SentenceTransformer
            print(f"Loading embedding model: {self.model_name}...")
            
            self.model = SentenceTransformer(self.model_name)
            self.dimension = getattr(self.model, "get_embedding_dimension", None)
            if callable(self.dimension):
                self.dimension = self.model.get_embedding_dimension()
            else:
                self.dimension = self.model.get_sentence_embedding_dimension()
            
            load_time = time.time() - start_time
            print(f"Loaded embedding model: {self.model_name}")
            print(f"   Dimension: {self.dimension}")
            print(f"   Load time: {load_time:.2f}s")
            
        except ImportError as e:
            print(f"sentence-transformers not installed: {e}")
            print("Install with: pip install sentence-transformers")
            self.model = None
        except Exception as e:
            print(f"Failed to load embedding model {self.model_name}: {e}")
            print("Check model name and internet connection")
            self.model = None
    
    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings for text(s)"""
        if isinstance(texts, str):
            texts = [texts]
        
        if self.model:
            # Real embeddings
            embeddings = self.model.encode(texts)
        else:
            # Mock embeddings for testing
            embeddings = self._mock_embeddings(texts)
        
        return np.array(embeddings)
    
    def _mock_embeddings(self, texts: List[str]) -> List:
        """Generate mock embeddings when model not available"""
        import hashlib
        
        embeddings = []
        for text in texts:
            # Create deterministic mock embedding based on text hash
            hash_val = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
            emb = [float((hash_val + i) % 1000) / 1000 for i in range(self.dimension)]
            embeddings.append(emb)
        
        return embeddings
    
    def encode_query(self, query: str) -> np.ndarray:
        """Encode a search query"""
        return self.encode(query)