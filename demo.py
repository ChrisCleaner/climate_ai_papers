#!/usr/bin/env python3
"""
Demo script - generates a sample digest with mock papers
to show the output format when arXiv API isn't available.
"""

from datetime import datetime, timedelta, timezone
from paper_monitor import Paper, generate_digest, score_relevance_local

# Sample papers that would typically come from arXiv
SAMPLE_PAPERS = [
    Paper(
        arxiv_id="2412.00001",
        title="FourCastNet-2: Global Weather Forecasting with Graph Neural Networks",
        authors=["Alice Chen", "Bob Kumar", "Carol Zhang", "David Lee"],
        abstract="""We present FourCastNet-2, an improved graph neural network architecture 
        for global weather prediction. Our model achieves 0.5-degree resolution forecasts 
        up to 10 days ahead, outperforming operational NWP models on precipitation and 
        temperature metrics. We demonstrate significant improvements in extreme weather 
        event prediction, particularly for tropical cyclones and heat waves. The model 
        runs 1000x faster than traditional numerical weather prediction systems while 
        maintaining comparable accuracy. Training was performed on 40 years of ERA5 
        reanalysis data.""",
        categories=["physics.ao-ph", "cs.LG"],
        published=datetime.now(timezone.utc) - timedelta(days=2),
        url="https://arxiv.org/abs/2412.00001"
    ),
    Paper(
        arxiv_id="2412.00002", 
        title="ClimateBERT-v2: Large Language Models for Climate Science Text Understanding",
        authors=["Elena Rodriguez", "Frank Miller", "Grace Wang"],
        abstract="""We introduce ClimateBERT-v2, a domain-adapted large language model 
        fine-tuned on 2 million climate science publications. Our model excels at 
        climate fact verification, IPCC report analysis, and extracting climate 
        commitments from corporate disclosures. We benchmark on ClimateQA and achieve 
        state-of-the-art performance. We release the model weights and a new dataset 
        of 50,000 annotated climate claims. Applications include automated analysis 
        of national climate pledges and greenwashing detection.""",
        categories=["cs.CL", "cs.AI"],
        published=datetime.now(timezone.utc) - timedelta(days=1),
        url="https://arxiv.org/abs/2412.00002"
    ),
    Paper(
        arxiv_id="2412.00003",
        title="Deep Learning Emulators for Earth System Model Components: A Review",
        authors=["Henry Smith", "Isabel Jones", "James Brown", "Karen Davis", "Leo Wilson"],
        abstract="""Earth System Models (ESMs) are computationally expensive, limiting 
        ensemble sizes and scenario exploration. This review surveys neural network 
        emulators that replace or augment ESM components including: atmospheric 
        radiation, ocean mixing, land surface processes, and carbon cycle dynamics. 
        We synthesize approaches using convolutional networks, transformers, and 
        physics-informed architectures. Key challenges include preserving conservation 
        laws, handling rare events, and uncertainty quantification. We identify 
        promising directions for hybrid modeling and discuss implications for 
        climate projections.""",
        categories=["physics.ao-ph", "cs.LG", "physics.geo-ph"],
        published=datetime.now(timezone.utc) - timedelta(days=3),
        url="https://arxiv.org/abs/2412.00003"
    ),
    Paper(
        arxiv_id="2412.00004",
        title="Satellite-Based Crop Yield Prediction Using Vision Transformers",
        authors=["Maria Garcia", "Nathan Park"],
        abstract="""Accurate crop yield prediction is essential for food security under 
        climate change. We apply Vision Transformers (ViT) to multi-spectral satellite 
        imagery for county-level yield estimation across the US Corn Belt. Our model 
        incorporates temporal attention to capture crop phenology and achieves 8% 
        improvement over CNN baselines. We analyze attention maps to identify key 
        growth stages and drought stress indicators. The approach generalizes across 
        crop types and can inform agricultural adaptation strategies.""",
        categories=["cs.CV", "cs.LG"],
        published=datetime.now(timezone.utc) - timedelta(days=4),
        url="https://arxiv.org/abs/2412.00004"
    ),
    Paper(
        arxiv_id="2412.00005",
        title="Reinforcement Learning for Optimal Building Energy Management",
        authors=["Oscar Tanaka", "Patricia Liu", "Quinn Ahmad"],
        abstract="""Buildings account for 40% of global energy consumption. We develop 
        a multi-agent reinforcement learning system for HVAC control that reduces 
        energy use while maintaining occupant comfort. Our approach uses model-based 
        RL with learned dynamics and handles the partial observability of building 
        thermal dynamics. Deployed in 5 commercial buildings, we achieve 15-25% 
        energy savings compared to rule-based controllers. We discuss integration 
        with demand response programs and grid decarbonization.""",
        categories=["cs.AI", "cs.LG", "eess.SY"],
        published=datetime.now(timezone.utc) - timedelta(days=5),
        url="https://arxiv.org/abs/2412.00005"
    ),
    Paper(
        arxiv_id="2412.00006",
        title="Diffusion Models for High-Resolution Climate Downscaling",
        authors=["Rachel Kim", "Steve Johnson"],
        abstract="""Global climate models operate at coarse resolutions (50-100km), 
        insufficient for local impact assessment. We introduce ClimDiff, a diffusion 
        model for statistical downscaling to 1km resolution. Conditioned on coarse GCM 
        outputs and high-resolution topography, our model generates physically 
        consistent temperature and precipitation fields. We preserve extreme value 
        statistics better than existing methods. Evaluation on CORDEX data shows 
        improved representation of orographic precipitation and urban heat islands.""",
        categories=["physics.ao-ph", "cs.LG"],
        published=datetime.now(timezone.utc) - timedelta(days=2),
        url="https://arxiv.org/abs/2412.00006"
    ),
]

# Add mock summaries
SAMPLE_SUMMARIES = {
    "2412.00001": """**Main Contribution:** Next-generation global weather forecasting using graph neural networks at 0.5° resolution.

**Method:** Graph neural network architecture processing atmospheric state as nodes on an icosahedral mesh with message passing.

**Data:** 40 years of ERA5 reanalysis data for training and validation.

**Climate Relevance:** Enables rapid ensemble forecasting for extreme weather events; critical for climate adaptation planning.

**Implications:** Could democratize high-quality weather forecasting; 1000x speedup enables uncertainty quantification.

**Limitations:** Skill degrades beyond 10 days; limited validation in data-sparse regions.

**TL;DR:** GNN-based weather model matches operational forecasts at 1000x the speed.""",

    "2412.00002": """**Main Contribution:** Domain-adapted LLM for automated analysis of climate science literature and corporate disclosures.

**Method:** BERT-based model fine-tuned on 2M climate papers with task-specific heads for QA and claim verification.

**Data:** Climate science corpus plus 50K annotated claims dataset (released).

**Climate Relevance:** Enables scalable analysis of climate commitments and detection of greenwashing.

**Implications:** Could automate monitoring of national climate pledges and corporate net-zero claims.

**Limitations:** English-only; may struggle with highly technical paleoclimate content.

**TL;DR:** LLM fine-tuned for climate text achieves SOTA on fact verification and corporate disclosure analysis.""",

    "2412.00003": """**Main Contribution:** Comprehensive review of ML emulators for computationally expensive Earth System Model components.

**Method:** Literature synthesis covering CNN, transformer, and physics-informed approaches across ESM components.

**Data:** Review of ~150 papers applying ML to atmospheric, ocean, land, and carbon cycle modeling.

**Climate Relevance:** Identifies pathways to accelerate climate projections and enable larger ensembles.

**Implications:** Hybrid ML-physics models could enable real-time climate services and improved uncertainty bounds.

**Limitations:** Review scope limited to peer-reviewed literature through mid-2024.

**TL;DR:** Authoritative review of neural network approaches to accelerating Earth System Model components.""",
}


def main():
    print("🌍 Generating sample Climate-AI digest...\n")
    
    # Score papers
    for paper in SAMPLE_PAPERS:
        paper.relevance_score = score_relevance_local(paper)
        print(f"  {paper.relevance_score}⭐ {paper.title[:50]}...")
        
        # Add mock summary if available
        if paper.arxiv_id in SAMPLE_SUMMARIES:
            paper.summary = SAMPLE_SUMMARIES[paper.arxiv_id]
    
    # Filter high relevance
    high_relevance = [p for p in SAMPLE_PAPERS if p.relevance_score >= 3]
    
    # Generate digest
    output_path = "SAMPLE_climate_ai_digest.md"
    generate_digest(high_relevance, output_path)
    
    print(f"\n✅ Sample digest generated: {output_path}")
    print(f"   Papers included: {len(high_relevance)}")


if __name__ == "__main__":
    main()
