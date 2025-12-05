"""
Claude API Integration for Climate-AI Paper Monitor

Provides LLM-based relevance scoring and paper summarization.
Requires ANTHROPIC_API_KEY environment variable.
"""

import os
import json
from typing import Optional
import requests

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"


def get_api_key() -> str:
    """Get API key from environment."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError(
            "ANTHROPIC_API_KEY environment variable not set. "
            "Get your key from https://console.anthropic.com/"
        )
    return key


def call_claude(
    prompt: str,
    system: Optional[str] = None,
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 1024,
    temperature: float = 0.3
) -> str:
    """
    Call Claude API with a prompt.
    
    Args:
        prompt: The user message
        system: Optional system prompt
        model: Model to use
        max_tokens: Maximum response tokens
        temperature: Sampling temperature
    
    Returns:
        The assistant's response text
    """
    headers = {
        "x-api-key": get_api_key(),
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    
    if system:
        payload["system"] = system
    
    response = requests.post(
        ANTHROPIC_API_URL,
        headers=headers,
        json=payload,
        timeout=60
    )
    response.raise_for_status()
    
    data = response.json()
    return data["content"][0]["text"]


def score_paper_relevance(abstract: str) -> int:
    """
    Use Claude to score paper relevance (0-5).
    
    Args:
        abstract: Paper abstract text
    
    Returns:
        Integer score 0-5
    """
    prompt = f"""Score from 0-5 how relevant this paper is to the intersection of AI and climate science.

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

    try:
        response = call_claude(prompt, max_tokens=10, temperature=0)
        # Extract just the number
        score = int(response.strip())
        return max(0, min(5, score))  # Clamp to 0-5
    except Exception as e:
        print(f"Error scoring paper: {e}")
        return 0


def summarize_paper(title: str, authors: list[str], abstract: str) -> str:
    """
    Generate a structured summary of a paper.
    
    Args:
        title: Paper title
        authors: List of author names
        abstract: Paper abstract
    
    Returns:
        Markdown-formatted summary
    """
    author_str = ", ".join(authors[:5])
    if len(authors) > 5:
        author_str += f" et al. ({len(authors)} total)"
    
    prompt = f"""Summarize this climate-AI research paper in a structured format.

**Title:** {title}
**Authors:** {author_str}
**Abstract:** {abstract}

Provide a concise summary with these sections:

**Main Contribution:** [1-2 sentences]
**Method:** [Key technical approach]  
**Data:** [Datasets mentioned]
**Climate Relevance:** [Connection to climate/environment]
**Implications:** [Real-world applications]
**Limitations:** [If apparent]
**TL;DR:** [One sentence]

Keep each section brief (1-2 sentences max). Use bullet points sparingly."""

    system = """You are a research assistant specializing in AI applications to climate science. 
Provide clear, technical summaries that highlight both the AI methodology and climate science relevance.
Be concise and precise."""

    try:
        summary = call_claude(prompt, system=system, max_tokens=600, temperature=0.3)
        return summary
    except Exception as e:
        print(f"Error summarizing paper: {e}")
        return f"*Summary generation failed: {e}*"


def batch_process_papers(papers: list, min_score: int = 3) -> list:
    """
    Process a batch of papers: score and summarize those above threshold.
    
    Args:
        papers: List of Paper objects (from paper_monitor.py)
        min_score: Minimum relevance score to summarize
    
    Returns:
        Updated list of papers with scores and summaries
    """
    import time
    
    print(f"Processing {len(papers)} papers with Claude API...")
    
    for i, paper in enumerate(papers):
        print(f"  [{i+1}/{len(papers)}] Scoring: {paper.title[:50]}...")
        
        # Score relevance
        paper.relevance_score = score_paper_relevance(paper.abstract)
        print(f"    Score: {paper.relevance_score}")
        
        # Only summarize if above threshold
        if paper.relevance_score >= min_score:
            print(f"    Generating summary...")
            paper.summary = summarize_paper(
                paper.title, 
                paper.authors, 
                paper.abstract
            )
        
        # Rate limiting (be nice to the API)
        time.sleep(1)
    
    return papers


# Example usage
if __name__ == "__main__":
    # Test with a sample abstract
    test_abstract = """
    We present ClimateBERT, a transformer-based model fine-tuned on climate science 
    literature for improved understanding of climate-related text. Our model achieves 
    state-of-the-art performance on climate fact verification, document classification, 
    and entity extraction tasks. We demonstrate applications in analyzing IPCC reports 
    and corporate climate disclosures. The model and datasets are publicly available.
    """
    
    print("Testing Claude API integration...")
    print(f"API Key configured: {'Yes' if os.environ.get('ANTHROPIC_API_KEY') else 'No'}")
    
    if os.environ.get("ANTHROPIC_API_KEY"):
        score = score_paper_relevance(test_abstract)
        print(f"\nRelevance score: {score}/5")
        
        summary = summarize_paper(
            "ClimateBERT: A Pretrained Language Model for Climate-Related Text",
            ["Alice Smith", "Bob Johnson", "Carol Williams"],
            test_abstract
        )
        print(f"\nSummary:\n{summary}")
    else:
        print("\nSet ANTHROPIC_API_KEY to test API calls")
