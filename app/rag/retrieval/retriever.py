"""
Retriever - Multi-hop document retrieval
"""

from typing import List, Dict
import numpy as np

class Retriever:
    """Multi-hop document retriever"""
    
    def __init__(self, vector_store, embedding_model):
        self.vector_store = vector_store
        self.embedding = embedding_model
    
    def single_hop(self, query: str, n_results: int = 5) -> List[Dict]:
        """Single retrieval hop"""
        query_embedding = self.embedding.encode_query(query)
        results = self.vector_store.search(query_embedding.tolist(), n_results)
        return results
    
    def multi_hop(self, query: str, hops: int = 2, n_results_per_hop: int = 3) -> List[Dict]:
        """
        Multi-hop retrieval:
        Hop 1: Search with original query
        Hop 2: Extract concepts from Hop 1, search again
        """
        all_results = []
        
        # Hop 1
        hop1_results = self.single_hop(query, n_results_per_hop)
        all_results.extend(hop1_results)
        
        # Hop 2
        if hops >= 2 and hop1_results:
            concepts = self._extract_concepts(hop1_results)
            refined_query = f"{query} Related to: {concepts}"
            hop2_results = self.single_hop(refined_query, n_results_per_hop)
            all_results.extend(hop2_results)
        
        # Hop 3 (optional, for complex queries)
        if hops >= 3 and hop1_results:
            deep_query = self._generate_deep_query(query, hop1_results)
            hop3_results = self.single_hop(deep_query, n_results_per_hop)
            all_results.extend(hop3_results)
        
        # Remove duplicates by text content
        seen = set()
        unique_results = []
        for result in all_results:
            text_preview = result["text"][:100]
            if text_preview not in seen:
                seen.add(text_preview)
                unique_results.append(result)
        
        return unique_results
    
    def _extract_concepts(self, results: List[Dict]) -> str:
        """Extract key concepts from retrieval results"""
        combined = " ".join([r["text"][:200] for r in results[:2]])
        
        # Simple keyword extraction
        words = combined.split()
        important = [w for w in words if len(w) > 6][:5]
        
        return " ".join(important) if important else combined[:100]
    
    def _generate_deep_query(self, original_query: str, results: List[Dict]) -> str:
        """Generate query for third hop"""
        concepts = self._extract_concepts(results)
        return f"Detailed information about: {original_query}. Key aspects: {concepts}"
    
    def retrieve_with_citations(self, query: str, hops: int = 2) -> List[Dict]:
        """Retrieve with citation tracking"""
        results = self.multi_hop(query, hops)
        
        for i, result in enumerate(results, 1):
            result["citation"] = f"[{i}]"
            result["relevance"] = self._get_relevance_label(result["similarity_score"])
        
        return results
    
    def _get_relevance_label(self, score: float) -> str:
        """Get human-readable relevance label"""
        if score >= 0.8:
            return "high"
        elif score >= 0.6:
            return "medium"
        elif score >= 0.4:
            return "low"
        return "very_low"