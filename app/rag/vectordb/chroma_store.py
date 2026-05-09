
from __future__ import annotations

import logging
import time
from typing import Dict, List

logger = logging.getLogger(__name__)


class ChromaStore:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_client()

    def _initialize_client(self) -> None:
        t0 = time.perf_counter()
        try:
            import chromadb
            from chromadb import __version__ as chromadb_version

            try:
                from pydantic import __version__ as pydantic_version
            except ImportError:
                pydantic_version = "unknown"

            logger.info(
                "chroma_client_init",
                extra={
                    "chromadb_version": chromadb_version,
                    "pydantic_version": pydantic_version,
                    "persist_directory": self.persist_directory,
                },
            )

            client_class = getattr(chromadb, "PersistentClient", None) or getattr(chromadb, "Client", None)
            if client_class is None:
                raise AttributeError("No valid ChromaDB client class found")

            if client_class.__name__ == "PersistentClient":
                self.client = client_class(path=self.persist_directory)
            else:
                self.client = client_class()

            latency_ms = (time.perf_counter() - t0) * 1000
            logger.info(
                "chroma_client_ready",
                extra={"persist_directory": self.persist_directory, "latency_ms": round(latency_ms, 2)},
            )
        except ImportError as e:
            logger.error(
                "chroma_import_failed",
                extra={"error": str(e), "remediation": "pip install chromadb (see requirements.txt)"},
            )
            self.client = None
        except Exception as e:
            logger.exception(
                "chroma_client_failed",
                extra={
                    "error": str(e),
                    "remediation": "Check disk permissions for persist path and pydantic/chromadb versions.",
                },
            )
            self.client = None

    def create_collection(self, name: str = "knowledge_base") -> None:
        """
        Idempotent collection attach: never fails solely because the collection already exists.
        Prefers get_or_create_collection when available (ChromaDB >= 0.4).
        """
        if not self.client:
            logger.error("chroma_collection_skipped", extra={"reason": "no_client", "collection": name})
            return

        t0 = time.perf_counter()
        action = "unknown"

        try:
            if hasattr(self.client, "get_or_create_collection"):
                self.collection = self.client.get_or_create_collection(name=name)
                action = "get_or_create"
            else:
                try:
                    self.collection = self.client.get_collection(name=name)
                    action = "reused_get_collection"
                except Exception:
                    self.collection = self.client.create_collection(name=name)
                    action = "created_new"
        except Exception as first_exc:
            logger.warning(
                "chroma_collection_primary_failed",
                extra={"collection": name, "error": str(first_exc)},
            )
            try:
                self.collection = self.client.get_collection(name=name)
                action = "reused_after_primary_failure"
            except Exception:
                try:
                    self.collection = self.client.create_collection(name=name)
                    action = "created_after_get_failed"
                except Exception as second_exc:
                    logger.error(
                        "chroma_collection_unrecoverable",
                        extra={
                            "collection": name,
                            "error": str(second_exc),
                            "remediation": "Remove lock files under chroma_db if another process crashed, or use a fresh persist path.",
                        },
                    )
                    self.collection = None
                    return

        latency_ms = (time.perf_counter() - t0) * 1000
        doc_count = self.collection.count() if self.collection else 0
        logger.info(
            "chroma_collection_ready",
            extra={
                "collection": name,
                "action": action,
                "latency_ms": round(latency_ms, 2),
                "document_count": doc_count,
            },
        )

    def add_documents(self, ids: List[str], embeddings: List, texts: List[str], metadatas: List[Dict]):
        if self.collection:
            self.collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
            logger.info("chroma_documents_added", extra={"count": len(texts)})

    def search(self, query_embedding: List, n_results: int = 5) -> List[Dict]:
        if not self.collection:
            return []

        if isinstance(query_embedding, list):
            if len(query_embedding) > 0 and isinstance(query_embedding[0], list):
                if isinstance(query_embedding[0][0], list):
                    query_embedding = query_embedding[0]
                if len(query_embedding) > 0 and isinstance(query_embedding[0], list):
                    query_embedding = query_embedding[0]

        if isinstance(query_embedding, list) and len(query_embedding) > 0:
            if isinstance(query_embedding[0], list):
                query_embedding = query_embedding[0]

        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )

            formatted = []
            if results and results["documents"] and len(results["documents"][0]) > 0:
                for i in range(len(results["documents"][0])):
                    meta = results["metadatas"][0][i] or {}
                    formatted.append(
                        {
                            "chunk_id": meta.get("chunk_id") or f"chroma_{i}",
                            "text": results["documents"][0][i],
                            "source": meta.get("source", "unknown"),
                            "similarity_score": 1 - results["distances"][0][i],
                            "metadata": meta,
                        }
                    )
            return formatted
        except Exception as e:
            logger.error("chroma_search_failed", extra={"error": str(e)})
            return []

    def count(self) -> int:
        if self.collection:
            return self.collection.count()
        return 0
