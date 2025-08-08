from scihub import SciHub
import re
import requests
import urllib3
from typing import Dict

# Disable HTTPS certificate warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def create_scihub_instance():
    """Creates a SciHub instance with settings"""
    sh = SciHub()
    sh.timeout = 10  # Keep original timeout
    return sh


def extract_pdf_link_from_html(html_content, base_url):
    """Extracts PDF link from HTML"""
    patterns = [
        r'src="(.*?\.pdf.*?)"',
        r'href="(.*?\.pdf.*?)"',
        r'location\.href\s*=\s*["\']([^"\']*\.pdf[^"\']*)["\']',
        r'"(https?://[^"]*\.pdf[^"]*)"'
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, html_content, re.IGNORECASE)
        for match in matches:
            if match.startswith('http'):
                return match
            elif match.startswith('//'):
                return f"https:{match}"
            elif match.startswith('/'):
                return f"{base_url}{match}"
    
    return None


def search_with_direct_url(doi):
    """Backup search method through direct URLs with more mirrors"""
    mirrors = [
        "https://sci-hub.ru",
        "https://sci-hub.st", 
        "https://sci-hub.se",
        "https://sci-hub.ren"  # Added more mirrors like in main version
    ]
    
    for mirror in mirrors:
        try:
            direct_url = f"{mirror}/{doi}"
            response = requests.get(direct_url, timeout=3, verify=False)  # Increased timeout
            
            if response.status_code == 200:
                pdf_link = extract_pdf_link_from_html(response.text, mirror)
                if pdf_link:
                    return {
                        'doi': doi,
                        'pdf_url': pdf_link,
                        'status': 'success',
                        'method': 'direct_url',
                        'mirror': mirror
                    }
        except Exception as e:
            print(f"   Error with mirror {mirror}: {e}")
            continue
    
    return None  # Return None when not found, like in original version


class SciHubSearcher:
    """
    Minimal SciHub searcher for DOI-based paper lookup
    Only contains functionality needed for Semantic Scholar integration
    """
    
    def __init__(self):
        # Initialize SciHub
        self.scihub = create_scihub_instance()
    
    def search_paper_by_doi(self, doi: str) -> Dict:
        """Search for paper by DOI with enhanced error handling"""
        try:
            result = self.scihub.fetch(doi)
            return {
                'doi': doi,
                'pdf_url': result['url'],
                'status': 'success',
                'title': result.get('title', ''),
                'author': result.get('author', ''),
                'year': result.get('year', ''),
                'method': 'standard'
            }
        except Exception as e:
            # print(f"Standard search failed: {str(e)}")
            backup_result = search_with_direct_url(doi)
            
            if backup_result:  # Check if backup_result is not None
                print(f"✅ Backup method successful!")
                return backup_result
            else:
                # print(f"❌ All methods failed")
                return {'doi': doi, 'status': 'not_found'}


__all__ = ["SciHubSearcher"] 