"""
Reranker - Re-rank retrieved documents for better relevance
"""

from typing import List, Dict

class Reranker:
    """Re-rank retrieved documents"""
    
    def __init__(self):
        pass
    
    def rerank(self, query: str, results: List[Dict]) -> List[Dict]:
        """
        Re-rank results based on additional criteria
        Currently uses simple scoring, can be extended with cross-encoders
        """
        for result in results:
            # Boost score based on keyword matches
            query_terms = set(query.lower().split())
            text_terms = set(result["text"].lower().split())
            
            common_terms = query_terms & text_terms
            keyword_boost = len(common_terms) / max(len(query_terms), 1)
            
            # Apply boost
            result["similarity_score"] = min(1.0, result["similarity_score"] * (1 + keyword_boost * 0.3))
        
        # Sort by new score
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        return results
    
    def cross_encoder_rerank(self, query: str, results: List[Dict]) -> List[Dict]:
        """
        Use cross-encoder for more accurate reranking
        Requires sentence-transformers cross-encoder model
        """
        try:
            from sentence_transformers import CrossEncoder
            
            model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            
            pairs = [(query, r["text"]) for r in results]
            scores = model.predict(pairs)
            
            for i, result in enumerate(results):
                result["cross_encoder_score"] = float(scores[i])
                # Combine with existing score
                result["similarity_score"] = (result["similarity_score"] + float(scores[i])) / 2
            
            results.sort(key=lambda x: x["similarity_score"], reverse=True)
            
        except ImportError:
            print("CrossEncoder not available. Install: pip install sentence-transformers")
        
        return results