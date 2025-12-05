#!/usr/bin/env python3
"""
Climate-AI Paper Digest Runner

Usage:
    python run_digest.py                    # Basic run with heuristic scoring
    python run_digest.py --use-llm          # Use Claude API for scoring/summaries
    python run_digest.py --days 3           # Last 3 days only
    python run_digest.py --min-score 4      # Only high-relevance papers
    python run_digest.py --source semantic  # Use Semantic Scholar instead of arXiv
    python run_digest.py --source both      # Use both sources
"""

import argparse
import os
from datetime import datetime

from paper_monitor import (
    run_pipeline,
    fetch_arxiv_papers,
    keyword_filter,
    score_relevance_local,
    generate_digest,
    ARXIV_CATEGORIES
)
from semantic_scholar import fetch_climate_ai_papers as fetch_semantic_scholar_papers


def main():
    parser = argparse.ArgumentParser(
        description="Generate a digest of recent Climate-AI papers from arXiv"
    )
    parser.add_argument(
        "--days", "-d",
        type=int,
        default=7,
        help="Number of days to look back (default: 7)"
    )
    parser.add_argument(
        "--max-papers", "-m",
        type=int,
        default=30,
        help="Max papers per arXiv category (default: 30)"
    )
    parser.add_argument(
        "--min-score", "-s",
        type=int,
        default=3,
        choices=[0, 1, 2, 3, 4, 5],
        help="Minimum relevance score to include (default: 3)"
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use Claude API for scoring and summaries (requires ANTHROPIC_API_KEY)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file path (default: climate_ai_digest_YYYY-MM-DD.md)"
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        default=None,
        help="Override arXiv categories to search"
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["arxiv", "semantic", "both"],
        default="both",
        help="Paper source: arxiv, semantic (Semantic Scholar), or both (default: both)"
    )
    
    args = parser.parse_args()
    
    # Generate default output filename
    if args.output is None:
        today = datetime.now().strftime("%Y-%m-%d")
        args.output = f"output/climate_ai_digest_{today}.md"
    
    # Check for API key if using LLM
    if args.use_llm and not os.environ.get("ANTHROPIC_API_KEY"):
        print("⚠️  Warning: --use-llm specified but ANTHROPIC_API_KEY not set")
        print("   Falling back to heuristic scoring")
        args.use_llm = False
    
    # Print configuration
    print("\n" + "=" * 60)
    print("🌍 Climate-AI Paper Digest Generator")
    print("=" * 60)
    print(f"  Days back:      {args.days}")
    print(f"  Max per cat:    {args.max_papers}")
    print(f"  Min score:      {args.min_score}")
    print(f"  Use LLM:        {args.use_llm}")
    print(f"  Source:         {args.source}")
    print(f"  Output:         {args.output}")
    print(f"  Categories:     {args.categories or 'default'}")
    print("=" * 60 + "\n")
    
    # Fetch papers from selected source(s)
    all_papers = []
    
    if args.source in ["arxiv", "both"]:
        print("📥 Fetching papers from arXiv...")
        categories = args.categories or ARXIV_CATEGORIES
        try:
            arxiv_papers = fetch_arxiv_papers(
                categories=categories,
                max_results=args.max_papers,
                days_back=args.days
            )
            all_papers.extend(arxiv_papers)
            print(f"  ✓ Got {len(arxiv_papers)} papers from arXiv")
        except Exception as e:
            print(f"  ✗ arXiv fetch failed: {e}")
            if args.source == "arxiv":
                print("  Tip: Try --source semantic as a backup")
    
    if args.source in ["semantic", "both"]:
        print("📥 Fetching papers from Semantic Scholar...")
        try:
            ss_api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
            semantic_papers = fetch_semantic_scholar_papers(
                days_back=args.days,
                max_per_query=args.max_papers // 5,  # Spread across queries
                api_key=ss_api_key
            )
            all_papers.extend(semantic_papers)
            print(f"  ✓ Got {len(semantic_papers)} papers from Semantic Scholar")
        except Exception as e:
            print(f"  ✗ Semantic Scholar fetch failed: {e}")
    
    # Deduplicate (papers might appear in both sources)
    seen_ids = set()
    papers = []
    for paper in all_papers:
        if paper.arxiv_id not in seen_ids:
            seen_ids.add(paper.arxiv_id)
            papers.append(paper)
    
    print(f"\n📚 Total unique papers: {len(papers)}")
    
    if not papers:
        print("\n⚠️  No papers fetched. Check your network connection.")
        return
    
    # Apply keyword filter
    print(f"\n🔍 Applying keyword filter...")
    filtered = [p for p in papers if keyword_filter(p)]
    print(f"  {len(filtered)}/{len(papers)} papers passed filter")
    
    # Run pipeline
    if args.use_llm:
        # Full pipeline with LLM
        from claude_integration import batch_process_papers
        
        print(f"\n🤖 Processing with Claude API...")
        processed = batch_process_papers(filtered, min_score=args.min_score)
        
        # Filter to minimum score
        final = [p for p in processed if (p.relevance_score or 0) >= args.min_score]
        
        print(f"\n📝 Generating digest...")
        generate_digest(final, args.output)
        
    else:
        # Heuristic-only scoring
        print(f"\n📊 Scoring relevance (heuristic)...")
        for paper in filtered:
            paper.relevance_score = score_relevance_local(paper)
        
        # Filter by minimum score
        scored_papers = [p for p in filtered if (p.relevance_score or 0) >= args.min_score]
        print(f"  {len(scored_papers)} papers with relevance >= {args.min_score}")
        
        print(f"\n📝 Generating digest...")
        generate_digest(scored_papers, args.output)
    
    print(f"\n✅ Done! Digest saved to: {args.output}")


if __name__ == "__main__":
    main()
