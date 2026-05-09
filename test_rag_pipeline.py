#!/usr/bin/env python
"""
Test the complete RAG pipeline
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.rag.pipeline import RAGPipeline

def main():
    print("="*60)
    print("TESTING RAG PIPELINE")
    print("="*60)
    
    # Initialize pipeline
    pipeline = RAGPipeline(vector_db_type="chroma")
    
    # Index sample documents
    print("\n📚 Indexing sample documents...")
    num_chunks = pipeline.index_documents(source="sample")
    print(f"✅ Indexed {num_chunks} chunks")
    
    # Test queries
    test_queries = [
        "What is Python used for?",
        "Tell me about Java web development",
        "Compare reinforcement learning and supervised learning for robotics"
    ]
    
    for query in test_queries:
        print(f"\n{'─'*50}")
        print(f"📝 QUERY: {query}")
        print(f"{'─'*50}")
        
        results = pipeline.search(query, hops=2, top_k=3)
        
        for result in results:
            print(f"\n{result['citation']} 📄 Source: {result['source']}")
            print(f"   📊 Score: {result['similarity_score']:.3f}")
            print(f"   📝 Content: {result['text'][:150]}...")
    
    # Show status
    print("\n" + "="*60)
    print("PIPELINE STATUS")
    print("="*60)
    status = pipeline.get_status()
    for key, value in status.items():
        print(f"   {key}: {value}")

if __name__ == "__main__":
    main()