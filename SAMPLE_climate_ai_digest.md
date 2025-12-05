# 🌍 Climate-AI Research Digest

**Generated:** 2025-12-03
**Papers reviewed:** 6

---

## 🔥 High Relevance Papers

### [FourCastNet-2: Global Weather Forecasting with Graph Neural Networks](https://arxiv.org/abs/2412.00001)

**Authors:** Alice Chen, Bob Kumar, Carol Zhang...
**Published:** 2025-12-01
**Categories:** physics.ao-ph, cs.LG
**Relevance Score:** ⭐⭐⭐⭐⭐

**Main Contribution:** Next-generation global weather forecasting using graph neural networks at 0.5° resolution.

**Method:** Graph neural network architecture processing atmospheric state as nodes on an icosahedral mesh with message passing.

**Data:** 40 years of ERA5 reanalysis data for training and validation.

**Climate Relevance:** Enables rapid ensemble forecasting for extreme weather events; critical for climate adaptation planning.

**Implications:** Could democratize high-quality weather forecasting; 1000x speedup enables uncertainty quantification.

**Limitations:** Skill degrades beyond 10 days; limited validation in data-sparse regions.

**TL;DR:** GNN-based weather model matches operational forecasts at 1000x the speed.

---

### [Deep Learning Emulators for Earth System Model Components: A Review](https://arxiv.org/abs/2412.00003)

**Authors:** Henry Smith, Isabel Jones, James Brown...
**Published:** 2025-11-30
**Categories:** physics.ao-ph, cs.LG, physics.geo-ph
**Relevance Score:** ⭐⭐⭐⭐⭐

**Main Contribution:** Comprehensive review of ML emulators for computationally expensive Earth System Model components.

**Method:** Literature synthesis covering CNN, transformer, and physics-informed approaches across ESM components.

**Data:** Review of ~150 papers applying ML to atmospheric, ocean, land, and carbon cycle modeling.

**Climate Relevance:** Identifies pathways to accelerate climate projections and enable larger ensembles.

**Implications:** Hybrid ML-physics models could enable real-time climate services and improved uncertainty bounds.

**Limitations:** Review scope limited to peer-reviewed literature through mid-2024.

**TL;DR:** Authoritative review of neural network approaches to accelerating Earth System Model components.

---

### [Satellite-Based Crop Yield Prediction Using Vision Transformers](https://arxiv.org/abs/2412.00004)

**Authors:** Maria Garcia, Nathan Park
**Published:** 2025-11-29
**Categories:** cs.CV, cs.LG
**Relevance Score:** ⭐⭐⭐⭐⭐

**Abstract:**
> Accurate crop yield prediction is essential for food security under 
        climate change. We apply Vision Transformers (ViT) to multi-spectral satellite 
        imagery for county-level yield estimation across the US Corn Belt. Our model 
        incorporates temporal attention to capture crop phenology and achieves 8% 
        improvement over CNN baselines. We analyze attention maps to identify key 
        growth stages and drought stress indicators. The approach generalizes across 
     ...

---

### [Reinforcement Learning for Optimal Building Energy Management](https://arxiv.org/abs/2412.00005)

**Authors:** Oscar Tanaka, Patricia Liu, Quinn Ahmad
**Published:** 2025-11-28
**Categories:** cs.AI, cs.LG, eess.SY
**Relevance Score:** ⭐⭐⭐⭐

**Abstract:**
> Buildings account for 40% of global energy consumption. We develop 
        a multi-agent reinforcement learning system for HVAC control that reduces 
        energy use while maintaining occupant comfort. Our approach uses model-based 
        RL with learned dynamics and handles the partial observability of building 
        thermal dynamics. Deployed in 5 commercial buildings, we achieve 15-25% 
        energy savings compared to rule-based controllers. We discuss integration 
        with de...

---

### [Diffusion Models for High-Resolution Climate Downscaling](https://arxiv.org/abs/2412.00006)

**Authors:** Rachel Kim, Steve Johnson
**Published:** 2025-12-01
**Categories:** physics.ao-ph, cs.LG
**Relevance Score:** ⭐⭐⭐⭐⭐

**Abstract:**
> Global climate models operate at coarse resolutions (50-100km), 
        insufficient for local impact assessment. We introduce ClimDiff, a diffusion 
        model for statistical downscaling to 1km resolution. Conditioned on coarse GCM 
        outputs and high-resolution topography, our model generates physically 
        consistent temperature and precipitation fields. We preserve extreme value 
        statistics better than existing methods. Evaluation on CORDEX data shows 
        improve...

---

## 📊 Medium Relevance Papers

### [ClimateBERT-v2: Large Language Models for Climate Science Text Understanding](https://arxiv.org/abs/2412.00002)

**Authors:** Elena Rodriguez, Frank Miller, Grace Wang
**Published:** 2025-12-02
**Categories:** cs.CL, cs.AI
**Relevance Score:** ⭐⭐⭐

**Main Contribution:** Domain-adapted LLM for automated analysis of climate science literature and corporate disclosures.

**Method:** BERT-based model fine-tuned on 2M climate papers with task-specific heads for QA and claim verification.

**Data:** Climate science corpus plus 50K annotated claims dataset (released).

**Climate Relevance:** Enables scalable analysis of climate commitments and detection of greenwashing.

**Implications:** Could automate monitoring of national climate pledges and corporate net-zero claims.

**Limitations:** English-only; may struggle with highly technical paleoclimate content.

**TL;DR:** LLM fine-tuned for climate text achieves SOTA on fact verification and corporate disclosure analysis.

---

---

## 📋 Sources

Papers sourced from arXiv categories:
`cs.AI`, `cs.LG`, `cs.CL`, `physics.ao-ph`, `physics.geo-ph`, `econ.GN`, `q-bio.QM`, `stat.ML`

---
*Generated by Climate-AI Paper Monitor*