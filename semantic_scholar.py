"""
Semantic Scholar API Integration for Climate-AI Paper Monitor

Provides an alternative/backup source for fetching papers when arXiv is unavailable.
Semantic Scholar has more permissive rate limits and includes citation data.

API Docs: https://api.semanticscholar.org/
"""

import requests
from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass
import time

# Import Paper dataclass from main module
from paper_monitor import Paper


SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1/paper/search"
SEMANTIC_SCHOLAR_BULK_API = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"

# Predefined search queries for climate-AI intersection
CLIMATE_AI_QUERIES = [
    "climate machine learning",
    "weather forecasting deep learning",
    "earth system model neural network",
    "climate change AI prediction",
    "carbon emissions machine learning",
    "renewable energy forecasting",
    "extreme weather prediction neural",
    "climate downscaling deep learning",
    "satellite remote sensing machine learning climate",
    "agricultural yield prediction AI",
]


def fetch_semantic_scholar(
    query: str,
    limit: int = 100,
    year_from: Optional[int] = None,
    fields_of_study: Optional[list[str]] = None,
    api_key: Optional[str] = None
) -> list[Paper]:
    """
    Fetch papers from Semantic Scholar API.
    
    Args:
        query: Search query string
        limit: Maximum papers to return (max 100 per request)
        year_from: Only include papers from this year onwards
        fields_of_study: Filter by fields (e.g., ["Computer Science", "Environmental Science"])
        api_key: Optional API key for higher rate limits
    
    Returns:
        List of Paper objects
    """
    papers = []
    
    # Build request
    params = {
        "query": query,
        "limit": min(limit, 100),  # API max is 100
        "fields": "paperId,title,abstract,authors,year,publicationDate,externalIds,fieldsOfStudy,citationCount"
    }
    
    if year_from:
        params["year"] = f"{year_from}-"
    
    if fields_of_study:
        params["fieldsOfStudy"] = ",".join(fields_of_study)
    
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key
    
    try:
        response = requests.get(
            SEMANTIC_SCHOLAR_API,
            params=params,
            headers=headers,
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        
        for item in data.get("data", []):
            # Skip if no abstract
            if not item.get("abstract"):
                continue
            
            # Parse publication date
            pub_date_str = item.get("publicationDate")
            if pub_date_str:
                try:
                    published = datetime.fromisoformat(pub_date_str)
                except ValueError:
                    published = datetime(item.get("year", 2024), 1, 1)
            else:
                published = datetime(item.get("year", 2024), 1, 1)
            
            # Get arXiv ID if available
            external_ids = item.get("externalIds", {})
            arxiv_id = external_ids.get("ArXiv", item.get("paperId", ""))
            
            # Build URL
            if external_ids.get("ArXiv"):
                url = f"https://arxiv.org/abs/{external_ids['ArXiv']}"
            elif external_ids.get("DOI"):
                url = f"https://doi.org/{external_ids['DOI']}"
            else:
                url = f"https://www.semanticscholar.org/paper/{item['paperId']}"
            
            # Get author names
            authors = [
                author.get("name", "Unknown")
                for author in item.get("authors", [])
            ]
            
            # Get fields as categories
            categories = item.get("fieldsOfStudy") or []
            
            paper = Paper(
                arxiv_id=arxiv_id,
                title=item.get("title", "Untitled"),
                authors=authors,
                abstract=item.get("abstract", ""),
                categories=categories,
                published=published,
                url=url
            )
            papers.append(paper)
        
        print(f"  Fetched {len(papers)} papers for query: '{query}'")
        
    except requests.exceptions.RequestException as e:
        print(f"  Error fetching '{query}': {e}")
    
    return papers


def fetch_climate_ai_papers(
    days_back: int = 7,
    max_per_query: int = 50,
    api_key: Optional[str] = None
) -> list[Paper]:
    """
    Fetch climate-AI papers using multiple predefined queries.
    
    Args:
        days_back: How many days back to search
        max_per_query: Maximum papers per search query
        api_key: Optional Semantic Scholar API key
    
    Returns:
        Deduplicated list of Paper objects
    """
    all_papers = []
    current_year = datetime.now().year
    
    print(f"Fetching from Semantic Scholar ({len(CLIMATE_AI_QUERIES)} queries)...")
    
    for query in CLIMATE_AI_QUERIES:
        papers = fetch_semantic_scholar(
            query=query,
            limit=max_per_query,
            year_from=current_year - 1,  # Last 2 years for broader coverage
            api_key=api_key
        )
        all_papers.extend(papers)
        
        # Rate limiting - Semantic Scholar allows 100 requests/5 min without key
        time.sleep(1)
    
    # Deduplicate by paper ID
    seen_ids = set()
    unique_papers = []
    for paper in all_papers:
        if paper.arxiv_id not in seen_ids:
            seen_ids.add(paper.arxiv_id)
            unique_papers.append(paper)
    
    # Filter by date
    cutoff = datetime.now() - timedelta(days=days_back)
    recent_papers = [
        p for p in unique_papers 
        if p.published.replace(tzinfo=None) >= cutoff
    ]
    
    print(f"Total: {len(unique_papers)} unique papers, {len(recent_papers)} from last {days_back} days")
    
    return recent_papers


def get_paper_details(paper_id: str, api_key: Optional[str] = None) -> Optional[dict]:
    """
    Get detailed information about a specific paper.
    
    Args:
        paper_id: Semantic Scholar paper ID or arXiv ID
        api_key: Optional API key
    
    Returns:
        Paper details dict or None
    """
    url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}"
    params = {
        "fields": "paperId,title,abstract,authors,year,publicationDate,venue,citationCount,referenceCount,influentialCitationCount,fieldsOfStudy,s2FieldsOfStudy,externalIds,tldr"
    }
    
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching paper details: {e}")
        return None


def get_paper_citations(paper_id: str, limit: int = 10, api_key: Optional[str] = None) -> list[dict]:
    """
    Get papers that cite a given paper.
    
    Args:
        paper_id: Semantic Scholar paper ID
        limit: Maximum citations to return
        api_key: Optional API key
    
    Returns:
        List of citing paper info
    """
    url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/citations"
    params = {
        "fields": "title,authors,year,citationCount",
        "limit": limit
    }
    
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data.get("data", [])
    except Exception as e:
        print(f"Error fetching citations: {e}")
        return []


# Example usage
if __name__ == "__main__":
    import os
    
    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    
    print("Testing Semantic Scholar integration...\n")
    
    # Fetch recent climate-AI papers
    papers = fetch_climate_ai_papers(
        days_back=30,  # Last month
        max_per_query=20,
        api_key=api_key
    )
    
    print(f"\n📚 Found {len(papers)} recent papers\n")
    
    # Show top 5
    for paper in papers[:5]:
        print(f"📄 {paper.title}")
        print(f"   Authors: {', '.join(paper.authors[:3])}{'...' if len(paper.authors) > 3 else ''}")
        print(f"   Published: {paper.published.strftime('%Y-%m-%d')}")
        print(f"   Categories: {', '.join(paper.categories[:3])}")
        print(f"   URL: {paper.url}")
        print()
