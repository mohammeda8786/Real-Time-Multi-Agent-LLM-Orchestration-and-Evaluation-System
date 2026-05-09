"""
PDF Document Loader - Extract text from PDF files
"""

from typing import List, Dict
import os

class PDFLoader:
    """Load and extract text from PDF documents"""
    
    def __init__(self):
        self.supported_extensions = ['.pdf']
    
    def load(self, filepath: str) -> List[Dict]:
        """
        Load PDF file and extract text
        Returns list of document chunks with metadata
        """
        documents = []
        
        try:
            # Try to use pypdf if available
            from pypdf import PdfReader
            
            reader = PdfReader(filepath)
            full_text = ""
            
            for page in reader.pages:
                full_text += page.extract_text()
            
            documents.append({
                "id": os.path.basename(filepath),
                "text": full_text,
                "source": filepath,
                "type": "pdf",
                "pages": len(reader.pages)
            })
            
            print(f"Loaded PDF: {filepath} ({len(reader.pages)} pages)")
            
        except ImportError:
            # Fallback to mock data if pypdf not installed
            print(f"pypdf not installed. Using mock data for {filepath}")
            documents.append({
                "id": os.path.basename(filepath),
                "text": f"Sample content from {filepath}. Install pypdf for real PDF loading.",
                "source": filepath,
                "type": "pdf_mock"
            })
        except Exception as e:
            print(f"Error loading PDF {filepath}: {e}")
        
        return documents
    
    def load_directory(self, directory: str) -> List[Dict]:
        """Load all PDFs from a directory"""
        all_documents = []
        for filename in os.listdir(directory):
            if filename.endswith('.pdf'):
                filepath = os.path.join(directory, filename)
                docs = self.load(filepath)
                all_documents.extend(docs)
        return all_documents