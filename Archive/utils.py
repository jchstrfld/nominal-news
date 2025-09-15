# utils.py

from urllib.parse import urlparse

def get_source_name(url: str) -> str:
    """
    Extracts a readable source name from a URL.
    """
    try:
        domain = urlparse(url).netloc.lower().replace('www.', '')
        domain_map = {
            "reuters.com": "Reuters",
            "bbc.com": "BBC",
            "cnn.com": "CNN",
            "nytimes.com": "NYT",
            "foxnews.com": "Fox News",
            "apnews.com": "AP",
            "washingtonpost.com": "Washington Post",
            "theguardian.com": "The Guardian",
            "npr.org": "NPR",
            "bloomberg.com": "Bloomberg",
            "abcnews.go.com": "ABC News",
            "cbsnews.com": "CBS News",
            "politico.com": "Politico",
            "axios.com": "Axios"
        }
        return domain_map.get(domain, domain.split('.')[0].capitalize())
    except:
        return "Unknown Source"
