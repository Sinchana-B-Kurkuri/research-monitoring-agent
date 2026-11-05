# Overleaf Research Paper: Relevance-Gated Agentic Workflow for Autonomous Research Monitoring

**Author:** Sinchana B. Kurkuri\
**Target Venue:** ACM SIG Proceedings Two-Column Format (`acmart.cls`, `sigconf`)\
**Generated:** October 15, 2026\

---

## 1. Project Overview & File Manifest

This directory contains the complete, publication-grade academic research paper prepared for direct upload to Overleaf.

### Directory Structure
```
overleaf_research_paper/
├── main.tex                  # Primary LaTeX root document
├── references.bib            # Complete BibTeX bibliography (18 verified scholarly references)
├── README.md                 # This instruction guide
├── figures/                  # Publication-ready graphics and experimental plots
│   ├── e2_learning_curves.png
│   ├── e3_signal_ablation.png
│   ├── e4_centroid_update.png
│   ├── e5_efficiency.png
│   ├── e6_human_review_value.png
│   ├── final_pilot_before_after.png
│   └── fig1.png
└── sections/                 # Modular LaTeX section files
    ├── 01_introduction.tex   # Introduction, Problem Formulation, Contributions
    ├── 02_related_work.tex   # Related Literature & Scholarly Grounding
    ├── 03_architecture.tex   # State Machine & TikZ End-to-End Workflow Diagram
    ├── 04_methodology.tex    # Dense Embeddings, Percentile Gating, Adaptation Math
    ├── 05_experiments.tex    # Corpus, Stratified Sampling, Frozen Test Split
    ├── 06_results.tex        # Experiments E1–E7, Benchmark Tables 1–5
    ├── 07_discussion.tex     # Validity Threats, Scalability, Trade-offs
    ├── 08_conclusion.tex     # Concluding Remarks & Prioritized Future Work
    ├── 09_acknowledgments.tex# Acknowledgments
    └── 10_appendices.tex     # System Parameters (Table 6) & Summary Prompt
```

---

## 2. Step-by-Step Instructions for Overleaf

1. **Upload the ZIP archive:**
   - Log in to your Overleaf account ([www.overleaf.com](https://www.overleaf.com)).
   - Click the green **"New Project"** button in the top-left dashboard.
   - Select **"Upload Project"**.
   - Drag and drop or browse to select `overleaf_research_paper.zip`.

2. **Verify Project Settings:**
   - In Overleaf, open the project settings menu (click **"Menu"** in the top-left corner of the editor).
   - Ensure the following settings are configured:
     - **Compiler:** `pdfLaTeX` (or `XeLaTeX`)
     - **TeX Live Version:** `2023` or `2024` (Recommended / Default)
     - **Main document:** `main.tex`

3. **Recompile:**
   - Click the green **"Recompile"** button (or press `Ctrl + Enter` / `Cmd + Enter`).
   - The compiled two-column PDF will render in the right-hand preview panel.

---

## 3. Academic Integrity & Empirical Boundaries

In compliance with academic rigor and the project audit:
- **Offline Benchmark Results (E1–E6):** All metrics (precision 0.600, ROC-AUC 0.978, 0.0% LOW miss rate, 98.0% benchmark call reduction) are grounded strictly in the 171-paper stratified evaluation dataset and frozen test split.
- **Live Production Telemetry (E7):**
  - The live pilot was executed across 2 active days and 3 cycles (1 live, 2 replay).
  - Measured live token savings are formally designated **`NOT_MEASURABLE`** due to cold-start gating and absent production token logs in evaluated cycles.
  - Pilot hypotheses H1, H2, and H3 remain **`PENDING / EXCLUDED`** due to insufficient active days ($<5$) and feedback counts ($<30$).
  - Readiness tier is locked at **`EVALUATION_PARTIALLY_AVAILABLE`**.
  - The Phase 14 demonstration video is not yet recorded.
