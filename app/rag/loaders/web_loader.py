"""
Web Document Loader - Fetch content from URLs
"""

from typing import List, Dict
import requests
from urllib.parse import urlparse

class WebLoader:
    """Load content from web pages"""
    
    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def load(self, url: str) -> List[Dict]:
        """
        Load content from a URL
        Returns list of document chunks with metadata
        """
        documents = []
        
        try:
            response = requests.get(url, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()
            
            # Extract text (simplified - in production use BeautifulSoup)
            content = response.text
            
            # Remove HTML tags (basic)
            import re
            text = re.sub(r'<[^>]+>', ' ', content)
            text = re.sub(r'\s+', ' ', text).strip()
            
            documents.append({
                "id": urlparse(url).netloc,
                "text": text[:5000],  # Limit length
                "source": url,
                "type": "web",
                "title": url
            })
            
            print(f"✅ Loaded web page: {url}")
            
        except requests.Timeout:
            print(f"❌ Timeout loading {url}")
            documents.append({
                "id": url,
                "text": f"TIMEOUT: Could not load {url} within {self.timeout}s",
                "source": url,
                "type": "error",
                "error": "timeout"
            })
        except requests.RequestException as e:
            print(f"❌ Error loading {url}: {e}")
            documents.append({
                "id": url,
                "text": f"ERROR: {str(e)}",
                "source": url,
                "type": "error",
                "error": "request_failed"
            })
        
        return documents
    
    def load_multiple(self, urls: List[str]) -> List[Dict]:
        """Load multiple URLs"""
        all_documents = []
        for url in urls:
            docs = self.load(url)
            all_documents.extend(docs)
        return all_documents