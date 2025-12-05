"""
Climate-AI Research Paper Monitor & Digest Generator

Monitors arXiv for papers at the intersection of AI and climate science,
filters for relevance, and generates structured summaries.
"""

import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional
import json
import re
import time


@dataclass
class Paper:
    """Represents an arXiv paper."""
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    categories: list[str]
    published: datetime
    url: str
    relevance_score: Optional[int] = None
    summary: Optional[str] = None


# ============================================================================
# 1. PAPER SOURCES
# ============================================================================

ARXIV_CATEGORIES = [
    "cs.AI",           # Artificial Intelligence
    "cs.LG",           # Machine Learning
    "cs.CL",           # Computation and Language
    "physics.ao-ph",   # Atmospheric and Oceanic Physics
    "physics.geo-ph",  # Geophysics
    "econ.GN",         # General Economics (incl. environmental)
    "q-bio.QM",        # Quantitative Methods
    "stat.ML",         # Machine Learning (Statistics)
]

ARXIV_API_URL = "http://export.arxiv.org/api/query"


def fetch_arxiv_papers(
    categories: list[str],
    max_results: int = 100,
    days_back: int = 7
) -> list[Paper]:
    """
    Fetch recent papers from arXiv for specified categories.
    
    Args:
        categories: List of arXiv category codes
        max_results: Maximum papers to fetch per category
        days_back: How many days back to search
    
    Returns:
        List of Paper objects
    """
    papers = []
    
    for category in categories:
        print(f"Fetching papers from {category}...")
        
        # Build query - search by category
        query = f"cat:{category}"
        
        params = {
            "search_query": query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending"
        }
        
        try:
            # arXiv requires a proper User-Agent header
            headers = {
                "User-Agent": "ClimateAIDigest/1.0 (https://github.com/your-repo; your@email.com)"
            }
            response = requests.get(ARXIV_API_URL, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Parse XML response
            root = ET.fromstring(response.content)
            namespace = {"atom": "http://www.w3.org/2005/Atom"}
            
            for entry in root.findall("atom:entry", namespace):
                # Extract paper details
                arxiv_id = entry.find("atom:id", namespace).text.split("/")[-1]
                title = entry.find("atom:title", namespace).text.strip().replace("\n", " ")
                abstract = entry.find("atom:summary", namespace).text.strip().replace("\n", " ")
                
                # Get authors
                authors = [
                    author.find("atom:name", namespace).text
                    for author in entry.findall("atom:author", namespace)
                ]
                
                # Get categories
                cats = [
                    cat.attrib.get("term")
                    for cat in entry.findall("atom:category", namespace)
                ]
                
                # Get published date
                published_str = entry.find("atom:published", namespace).text
                published = datetime.fromisoformat(published_str.replace("Z", "+00:00"))
                
                # Filter by date
                cutoff = datetime.now(published.tzinfo) - timedelta(days=days_back)
                if published < cutoff:
                    continue
                
                # Get URL
                url = f"https://arxiv.org/abs/{arxiv_id}"
                
                paper = Paper(
                    arxiv_id=arxiv_id,
                    title=title,
                    authors=authors,
                    abstract=abstract,
                    categories=cats,
                    published=published,
                    url=url
                )
                papers.append(paper)
            
            # Rate limiting - be nice to arXiv
            time.sleep(3)
            
        except Exception as e:
            print(f"Error fetching {category}: {e}")
            continue
    
    # Remove duplicates (papers can be in multiple categories)
    seen_ids = set()
    unique_papers = []
    for paper in papers:
        if paper.arxiv_id not in seen_ids:
            seen_ids.add(paper.arxiv_id)
            unique_papers.append(paper)
    
    print(f"Fetched {len(unique_papers)} unique papers")
    return unique_papers


# ============================================================================
# 2. RELEVANCE FILTERING
# ============================================================================

# Keywords for boolean filtering
CLIMATE_KEYWORDS = [
    "climate", "weather", "earth system", "carbon", "ghg", "greenhouse",
    "adaptation", "mitigation", "emission", "atmospheric", "ocean",
    "temperature", "precipitation", "drought", "flood", "extreme weather",
    "renewable", "energy transition", "decarbonization", "net zero",
    "land use", "deforestation", "biodiversity", "ecosystem",
    "agriculture", "crop", "food security", "water resource",
    "sea level", "ice sheet", "glacier", "permafrost",
]

AI_KEYWORDS = [
    "machine learning", "deep learning", "neural network", "ai", 
    "artificial intelligence", "foundation model", "llm", "large language",
    "transformer", "diffusion model", "generative", "prediction",
    "forecasting", "classification", "regression", "reinforcement learning",
    "computer vision", "nlp", "natural language", "embedding",
    "surrogate model", "emulator", "downscaling",
]


def keyword_filter(paper: Paper) -> bool:
    """
    Step A: Apply boolean keyword filter.
    Returns True if paper mentions both climate AND AI keywords.
    """
    text = (paper.title + " " + paper.abstract).lower()
    
    has_climate = any(kw in text for kw in CLIMATE_KEYWORDS)
    has_ai = any(kw in text for kw in AI_KEYWORDS)
    
    return has_climate and has_ai


def score_relevance_prompt(abstract: str) -> str:
    """Generate the prompt for LLM relevance scoring."""
    return f"""Score from 0-5 how relevant this paper is to the intersection of AI and climate science.

Scoring guide:
- 5 = Directly contributes to climate modeling, mitigation, or adaptation using AI/ML
- 4 = Strong AI application to environmental/earth science problems
- 3 = Moderate relevance - AI methods applicable to climate, or climate data with some ML
- 2 = Tangential - one of AI or climate is central, other is minor
- 1 = Weak connection - mentions both but not substantive
- 0 = Unrelated to AI+climate intersection

Abstract:
{abstract}

Respond with ONLY a single integer 0-5, nothing else."""


def score_relevance_local(paper: Paper) -> int:
    """
    Heuristic-based relevance scoring (no LLM needed).
    Use this as a fallback or for quick filtering.
    """
    text = (paper.title + " " + paper.abstract).lower()
    score = 0
    
    # Count climate keyword matches
    climate_matches = sum(1 for kw in CLIMATE_KEYWORDS if kw in text)
    ai_matches = sum(1 for kw in AI_KEYWORDS if kw in text)
    
    # Base score on keyword density
    if climate_matches >= 3 and ai_matches >= 3:
        score = 5
    elif climate_matches >= 2 and ai_matches >= 2:
        score = 4
    elif climate_matches >= 1 and ai_matches >= 1:
        score = 3
    elif climate_matches >= 1 or ai_matches >= 1:
        score = 2
    
    # Bonus for high-signal terms
    high_signal = [
        "climate model", "weather forecast", "earth system model",
        "carbon footprint", "emission predict", "climate change",
        "extreme event", "climate projection"
    ]
    if any(term in text for term in high_signal):
        score = min(5, score + 1)
    
    return score


# ============================================================================
# 3. SUMMARIZATION
# ============================================================================

SUMMARY_PROMPT = """Summarize the following climate-AI research paper in a structured format.

Title: {title}
Authors: {authors}
Abstract: {abstract}

Provide your summary in this exact format:

## Main Contribution
[1-2 sentences on the core contribution]

## What's Novel
[What makes this work new or different]

## Method
[Key technical approach]

## Data Used
[Datasets or data sources mentioned]

## Climate Science Relevance
[How this connects to climate/environmental science]

## Practical Implications
[Real-world applications or impact]

## Limitations
[Acknowledged or apparent limitations]

## Why It Matters
[Broader significance]

## TL;DR
[One sentence summary]
"""


def generate_summary_prompt(paper: Paper) -> str:
    """Generate the summarization prompt for a paper."""
    return SUMMARY_PROMPT.format(
        title=paper.title,
        authors=", ".join(paper.authors[:5]) + ("..." if len(paper.authors) > 5 else ""),
        abstract=paper.abstract
    )


# ============================================================================
# 4. MARKDOWN DIGEST GENERATION
# ============================================================================

def generate_digest(
    papers: list[Paper],
    output_path: str = "climate_ai_digest.md"
) -> str:
    """
    Generate a markdown digest from processed papers.
    
    Args:
        papers: List of Paper objects with summaries
        output_path: Path to save the markdown file
    
    Returns:
        The markdown content as a string
    """
    today = datetime.now().strftime("%Y-%m-%d")
    
    md_lines = [
        f"# 🌍 Climate-AI Research Digest",
        f"",
        f"**Generated:** {today}",
        f"**Papers reviewed:** {len(papers)}",
        f"",
        f"---",
        f"",
    ]
    
    # Group by relevance score
    high_relevance = [p for p in papers if (p.relevance_score or 0) >= 4]
    medium_relevance = [p for p in papers if (p.relevance_score or 0) == 3]
    
    if high_relevance:
        md_lines.append("## 🔥 High Relevance Papers\n")
        for paper in high_relevance:
            md_lines.extend(_format_paper_entry(paper))
    
    if medium_relevance:
        md_lines.append("## 📊 Medium Relevance Papers\n")
        for paper in medium_relevance:
            md_lines.extend(_format_paper_entry(paper))
    
    # Add footer
    md_lines.extend([
        "---",
        "",
        "## 📋 Sources",
        "",
        "Papers sourced from arXiv categories:",
        ", ".join(f"`{cat}`" for cat in ARXIV_CATEGORIES),
        "",
        "---",
        f"*Generated by Climate-AI Paper Monitor*"
    ])
    
    content = "\n".join(md_lines)
    
    with open(output_path, "w") as f:
        f.write(content)
    
    print(f"Digest saved to {output_path}")
    return content


def _format_paper_entry(paper: Paper) -> list[str]:
    """Format a single paper for the digest."""
    lines = [
        f"### [{paper.title}]({paper.url})",
        f"",
        f"**Authors:** {', '.join(paper.authors[:3])}{'...' if len(paper.authors) > 3 else ''}",
        f"**Published:** {paper.published.strftime('%Y-%m-%d')}",
        f"**Categories:** {', '.join(paper.categories[:3])}",
        f"**Relevance Score:** {'⭐' * (paper.relevance_score or 0)}",
        f"",
    ]
    
    if paper.summary:
        lines.append(paper.summary)
    else:
        # Fallback to abstract excerpt
        abstract_excerpt = paper.abstract[:500] + "..." if len(paper.abstract) > 500 else paper.abstract
        lines.extend([
            "**Abstract:**",
            f"> {abstract_excerpt}",
        ])
    
    lines.extend(["", "---", ""])
    return lines


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_pipeline(
    days_back: int = 7,
    max_papers_per_category: int = 50,
    min_relevance_score: int = 3,
    use_llm_scoring: bool = False,
    output_path: str = "climate_ai_digest.md"
) -> list[Paper]:
    """
    Run the full paper monitoring and digest pipeline.
    
    Args:
        days_back: How many days of papers to fetch
        max_papers_per_category: Max papers per arXiv category
        min_relevance_score: Minimum score to include (0-5)
        use_llm_scoring: Whether to use LLM for scoring (requires API)
        output_path: Where to save the digest
    
    Returns:
        List of processed papers
    """
    print("=" * 60)
    print("Climate-AI Paper Monitor Pipeline")
    print("=" * 60)
    
    # Step 1: Fetch papers
    print("\n📥 Step 1: Fetching papers from arXiv...")
    papers = fetch_arxiv_papers(
        categories=ARXIV_CATEGORIES,
        max_results=max_papers_per_category,
        days_back=days_back
    )
    
    # Step 2a: Keyword filter
    print("\n🔍 Step 2a: Applying keyword filter...")
    filtered_papers = [p for p in papers if keyword_filter(p)]
    print(f"  {len(filtered_papers)}/{len(papers)} papers passed keyword filter")
    
    # Step 2b: Relevance scoring
    print("\n📊 Step 2b: Scoring relevance...")
    for paper in filtered_papers:
        if use_llm_scoring:
            # Placeholder - would call Claude API here
            prompt = score_relevance_prompt(paper.abstract)
            print(f"  [LLM scoring not implemented - using heuristic]")
            paper.relevance_score = score_relevance_local(paper)
        else:
            paper.relevance_score = score_relevance_local(paper)
    
    # Filter by minimum score
    scored_papers = [p for p in filtered_papers if (p.relevance_score or 0) >= min_relevance_score]
    print(f"  {len(scored_papers)} papers with relevance >= {min_relevance_score}")
    
    # Step 3: Generate digest
    print("\n📝 Step 3: Generating markdown digest...")
    generate_digest(scored_papers, output_path)
    
    print("\n✅ Pipeline complete!")
    print(f"  Total papers processed: {len(papers)}")
    print(f"  Papers in digest: {len(scored_papers)}")
    
    return scored_papers


if __name__ == "__main__":
    # Run with default settings
    papers = run_pipeline(
        days_back=7,
        max_papers_per_category=30,
        min_relevance_score=3,
        use_llm_scoring=False,
        output_path="climate_ai_digest.md"
    )
