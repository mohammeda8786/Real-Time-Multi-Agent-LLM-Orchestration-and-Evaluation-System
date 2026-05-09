"""
Text Chunker - Split documents into manageable chunks
"""

from typing import List, Dict
import re

class TextChunker:
    """Handle text chunking with multiple strategies"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_fixed_size(self, text: str) -> List[str]:
        """Split into fixed-size chunks"""
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            chunks.append(text[start:end])
            start += self.chunk_size - self.chunk_overlap
        
        return chunks
    
    def chunk_by_paragraph(self, text: str) -> List[str]:
        """Split by paragraphs"""
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def chunk_by_sentence(self, text: str) -> List[str]:
        """Split by sentences"""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def chunk_recursive(self, text: str) -> List[str]:
        """Recursive chunking using multiple separators"""
        separators = ["\n\n", "\n", ". ", " ", ""]
        chunks = []
        self._recursive_split(text, separators, 0, chunks)
        return chunks
    
    def _recursive_split(self, text: str, separators: List[str], depth: int, chunks: List[str]):
        """Recursive splitting logic"""
        if depth >= len(separators) or len(text) <= self.chunk_size:
            if text.strip():
                chunks.append(text.strip())
            return
        
        separator = separators[depth]
        if not separator:
            parts = list(text)
        else:
            parts = text.split(separator)
        
        current_chunk = ""
        for part in parts:
            if len(current_chunk) + len(part) + len(separator) <= self.chunk_size:
                current_chunk += part + separator
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = part + separator
        
        if current_chunk:
            chunks.append(current_chunk.strip())
    
    def chunk_document(self, document: Dict, strategy: str = "fixed") -> List[Dict]:
        """Chunk a document with metadata preservation"""
        text = document.get("text", "")
        
        if strategy == "paragraph":
            chunks_text = self.chunk_by_paragraph(text)
        elif strategy == "sentence":
            chunks_text = self.chunk_by_sentence(text)
        elif strategy == "recursive":
            chunks_text = self.chunk_recursive(text)
        else:
            chunks_text = self.chunk_fixed_size(text)
        
        chunks = []
        for i, chunk_text in enumerate(chunks_text):
            chunks.append({
                "chunk_id": f"{document.get('id', 'doc')}_chunk_{i}",
                "text": chunk_text,
                "source": document.get("source", "unknown"),
                "chunk_index": i,
                "parent_doc": document.get("id"),
                "metadata": document.get("metadata", {})
            })
        
        return chunks