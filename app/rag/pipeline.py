"""
RAG Pipeline - Complete orchestration of all RAG components
"""

from typing import List, Dict, Optional, Any
import logging
import os

from app.rag.loaders import PDFLoader, WebLoader
from app.rag.chunking import TextChunker

logger = logging.getLogger(__name__)
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
        
        logger.info(
            "rag_pipeline_initialized",
            extra={
                "vector_db_type": vector_db_type,
                "embedding_model": self.embedding.model_name,
                "chunk_size": chunk_size,
            },
        )
    
    def index_documents(self, source: str, **kwargs) -> int:
        """
        Index documents from various sources
        source: 'pdf', 'web', 'directory', 'sample'
        """
        logger.info("indexing_documents", extra={"source": source, "document_count": len(documents)})
        
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
        
        logger.info("documents_loaded", extra={"count": len(documents)})
        
        # Step 2: Chunk documents
        all_chunks = []
        for doc in documents:
            chunks = self.chunker.chunk_document(doc, strategy="recursive")
            all_chunks.extend(chunks)
        
        logger.info("chunks_created", extra={"chunk_count": len(all_chunks)})
        
        # Step 3: Generate embeddings
        texts = [chunk["text"] for chunk in all_chunks]
        embeddings = self.embedding.encode(texts)
        
        logger.info("embeddings_generated", extra={"count": len(embeddings)})
        
        # Step 4: Store in vector database
        ids = [chunk["chunk_id"] for chunk in all_chunks]
        metadatas = [
            {
                "source": chunk["source"],
                "chunk_index": chunk["chunk_index"],
                "chunk_id": chunk["chunk_id"],
            }
            for chunk in all_chunks
        ]
        
        self.vector_store.add_documents(ids, embeddings.tolist(), texts, metadatas)
        
        logger.info("chunks_stored", extra={"stored_count": self.vector_store.count()})
        
        return len(all_chunks)
    
    def retrieve(self, query: str, hops: int = 2, top_k: int = 5, rerank: bool = True) -> List[Dict]:
        """
        Retrieve relevant documents for a query
        """
        logger.info("retrieving", extra={"query_preview": query[:100], "hops": hops, "top_k": top_k})
        
        # Retrieve
        results = self.retriever.multi_hop(query, hops=hops, n_results_per_hop=top_k)
        
        logger.info("retrieved_documents", extra={"count": len(results)})
        
        # Rerank if requested
        if rerank and results:
            results = self.reranker.rerank(query, results)
            logger.info("reranked_results", extra={"query": query[:100], "result_count": len(results)})
        
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

    async def search_with_reasoning(self, query: str, llm, hops: int = 2, top_k: int = 5) -> tuple:
        """
        Multi-hop retrieval with explicit second-hop queries (LLM-planned) + rerank + dedupe.
        Returns (results, reasoning_path dict).
        """
        from app.utils.json_extract import extract_json_object

        reasoning_path: List[Dict[str, Any]] = []
        hop1 = self.retriever.single_hop(query, n_results=max(top_k, 5))
        reasoning_path.append({"step": 1, "query": query, "hits": len(hop1)})

        follow_ups: List[str] = []
        entities: List[str] = []
        if hop1 and hops >= 2:
            preview = "\n".join(f"- {r.get('text', '')[:280]}" for r in hop1[:3])
            plan_prompt = f"""Given the user question and first retrieval snippets, propose focused follow-up search phrases.
Return compact JSON only:
{{"entities": ["..."], "follow_up_queries": ["phrase1", "phrase2"]}}
User question: {query}
Snippets:
{preview}
JSON:"""
            resp = await llm.generate(plan_prompt, temperature=0.1, max_tokens=200)
            parsed = extract_json_object(resp.get("text") or "")
            if parsed:
                follow_ups = [str(x) for x in parsed.get("follow_up_queries") or []][:2]
                entities = [str(x) for x in parsed.get("entities") or []][:5]
            reasoning_path.append(
                {"step": 2, "planned_queries": follow_ups, "entities": entities, "raw_plan": (resp.get("text") or "")[:400]}
            )

        merged: List[Dict] = list(hop1)
        if hop1 and hops >= 2:
            top_text = hop1[0].get("text", "")[:250]
            top_chunk_id = hop1[0].get("chunk_id") or hop1[0].get("metadata", {}).get("chunk_id")
            if follow_ups:
                for fq in follow_ups:
                    dependent_query = f"{query} {top_text} {fq}"
                    merged.extend(self.retriever.single_hop(dependent_query, n_results=top_k))
                reasoning_path.append(
                    {
                        "step": 2,
                        "planned_queries": follow_ups,
                        "source_chunk_id": top_chunk_id,
                        "dependent_query_context": top_text,
                    }
                )
            else:
                refined = f"{query} Additional evidence from first-hop result: {top_text}"
                merged.extend(self.retriever.single_hop(refined, n_results=top_k))
                reasoning_path.append(
                    {
                        "step": 2,
                        "fallback_query": refined,
                        "source_chunk_id": top_chunk_id,
                    }
                )

        merged = self.retriever.dedupe_results(merged)
        if merged:
            merged = self.reranker.rerank(query, merged)
        merged = merged[:top_k]
        for i, result in enumerate(merged, 1):
            result["citation"] = f"[{i}]"
        reasoning_path.append({"step": 3, "after_dedupe": len(merged)})
        return merged, {"hops_executed": min(hops, 2), "path": reasoning_path}
    
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
            },
            {
                "id": "ai_intro",
                "text": "Artificial intelligence (AI) is the field of creating systems that perform tasks requiring human-like reasoning, perception, or learning. Modern AI includes machine learning, deep learning, and retrieval-augmented systems.",
                "source": "sample/ai_overview"
            },
            {
                "id": "ml_intro",
                "text": "Machine learning (ML) is a subset of AI where models learn patterns from data rather than being fully hand-programmed. Common paradigms include supervised, unsupervised, and reinforcement learning.",
                "source": "sample/ml_overview"
            },
            {
                "id": "rag_intro",
                "text": "Retrieval-augmented generation (RAG) combines a retriever that fetches relevant documents with a language model that generates answers grounded in those documents, improving factual accuracy and traceability.",
                "source": "sample/rag_overview"
            },
            {
                "id": "earth_science",
                "text": "Earth is not flat; it is approximately an oblate spheroid. Evidence includes ships disappearing hull-first over the horizon, lunar eclipses showing Earth's round shadow, and satellite imagery.",
                "source": "sample/geography"
            },
            {
                "id": "basic_math",
                "text": "In standard arithmetic on natural numbers, 2 + 2 equals 4. Claims that 2 + 2 equals 5 contradict elementary arithmetic without redefining symbols or using non-standard contexts.",
                "source": "sample/math_facts"
            },
        ]