# 🌍 Climate-AI Paper Digest Generator

Automated pipeline to monitor arXiv and Semantic Scholar for papers at the intersection of **AI/ML** and **climate science**, filter for relevance, and generate structured markdown digests.

## Features

- **Multi-source fetching**: Pulls from arXiv (8 categories) + Semantic Scholar (10 climate-AI queries)
- **Automatic fallback**: If arXiv is unavailable, Semantic Scholar provides backup coverage
- **Two-stage filtering**: Keyword boolean filter + relevance scoring
- **Flexible scoring**: Heuristic-based (fast) or LLM-based (accurate)
- **Structured summaries**: Consistent format for easy comparison
- **Markdown output**: Ready for GitHub, Notion, or any markdown viewer
- **GitHub Actions**: Automated weekly digests with commit to repo

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run with default settings (both sources, heuristic scoring)
python run_digest.py

# Use only Semantic Scholar (more reliable from cloud environments)
python run_digest.py --source semantic

# Run with Claude API for better scoring/summaries
export ANTHROPIC_API_KEY="your-key-here"
python run_digest.py --use-llm

# Customize
python run_digest.py --days 3 --min-score 4 --output my_digest.md
```

## CLI Options

| Flag | Description | Default |
|------|-------------|---------|
| `--days`, `-d` | Days to look back | 7 |
| `--max-papers`, `-m` | Max papers per category/query | 30 |
| `--min-score`, `-s` | Minimum relevance (0-5) | 3 |
| `--source` | Paper source: `arxiv`, `semantic`, or `both` | `both` |
| `--use-llm` | Use Claude API for scoring | False |
| `--output`, `-o` | Output file path | `climate_ai_digest_YYYY-MM-DD.md` |
| `--categories` | Override arXiv categories | (see below) |

## Paper Sources

### arXiv Categories

- `cs.AI` - Artificial Intelligence
- `cs.LG` - Machine Learning  
- `cs.CL` - Computation and Language
- `physics.ao-ph` - Atmospheric and Oceanic Physics
- `physics.geo-ph` - Geophysics
- `econ.GN` - General Economics
- `q-bio.QM` - Quantitative Methods
- `stat.ML` - Machine Learning (Statistics)

### Semantic Scholar Queries

- "climate machine learning"
- "weather forecasting deep learning"
- "earth system model neural network"
- "climate change AI prediction"
- "carbon emissions machine learning"
- "renewable energy forecasting"
- "extreme weather prediction neural"
- "climate downscaling deep learning"
- "satellite remote sensing machine learning climate"
- "agricultural yield prediction AI"

## Relevance Scoring Guide

| Score | Meaning |
|-------|---------|
| ⭐⭐⭐⭐⭐ (5) | Directly contributes to climate modeling/mitigation using AI |
| ⭐⭐⭐⭐ (4) | Strong AI application to environmental problems |
| ⭐⭐⭐ (3) | Moderate relevance - methods applicable to climate |
| ⭐⭐ (2) | Tangential connection |
| ⭐ (1) | Weak connection |
| (0) | Unrelated |

## Project Structure

```
climate_ai_digest/
├── .github/
│   └── workflows/
│       └── weekly-digest.yml  # GitHub Actions automation
├── digests/                   # Generated digests (created by workflow)
├── paper_monitor.py           # Core pipeline: arXiv fetch, filter, score
├── semantic_scholar.py        # Semantic Scholar API integration
├── claude_integration.py      # Claude API for LLM scoring/summaries
├── run_digest.py              # CLI runner script
├── demo.py                    # Demo with sample papers
├── requirements.txt
└── README.md
```

## GitHub Actions Automation

The included workflow automatically generates a weekly digest every Monday at 8am UTC.

### Setup

1. **Add repository secrets** (Settings → Secrets → Actions):
   - `ANTHROPIC_API_KEY` - Your Claude API key (required for `--use-llm`)
   - `SEMANTIC_SCHOLAR_API_KEY` - Optional, for higher rate limits

2. **Enable Actions** (Settings → Actions → General):
   - Allow read and write permissions

3. **Manual trigger**:
   - Go to Actions → Weekly Climate-AI Digest → Run workflow
   - Customize days, min score, and LLM usage

### Workflow outputs

- Digests are committed to `digests/YYYY-MM-DD.md`
- `digests/latest.md` always points to the most recent
- Artifacts are retained for 30 days

### Customizing schedule

Edit `.github/workflows/weekly-digest.yml`:

```yaml
on:
  schedule:
    # Every Monday at 8am UTC
    - cron: '0 8 * * 1'
    
    # Daily at 6am UTC
    # - cron: '0 6 * * *'
    
    # Twice weekly (Mon & Thu)
    # - cron: '0 8 * * 1,4'
```

## Environment Variables

| Variable | Purpose | Required |
|----------|---------|----------|
| `ANTHROPIC_API_KEY` | Claude API for LLM scoring/summaries | For `--use-llm` |
| `SEMANTIC_SCHOLAR_API_KEY` | Higher rate limits (100→5000 req/5min) | Optional |

## Extending

### Add custom Semantic Scholar queries

Edit `CLIMATE_AI_QUERIES` in `semantic_scholar.py`:

```python
CLIMATE_AI_QUERIES = [
    "your custom query",
    # ...existing queries...
]
```

### Custom keywords for filtering

Edit `CLIMATE_KEYWORDS` and `AI_KEYWORDS` in `paper_monitor.py` to tune the boolean filter.

### Add email notifications

Add to the GitHub workflow:

```yaml
- name: Send email
  uses: dawidd6/action-send-mail@v3
  with:
    server_address: smtp.gmail.com
    server_port: 465
    username: ${{ secrets.EMAIL_USERNAME }}
    password: ${{ secrets.EMAIL_PASSWORD }}
    subject: Weekly Climate-AI Digest
    to: your@email.com
    from: Climate-AI Bot
    body: New digest available!
    attachments: digests/latest.md
```

## License

MIT
