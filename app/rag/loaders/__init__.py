"""
Document Loaders - Load from various sources
"""

from app.rag.loaders.pdf_loader import PDFLoader
from app.rag.loaders.web_loader import WebLoader

__all__ = ['PDFLoader', 'WebLoader']