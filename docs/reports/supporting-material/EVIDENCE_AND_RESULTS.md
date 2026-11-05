# EMPIRICAL EVIDENCE AND EXPERIMENTAL RESULTS COMPENDIUM

**Repository:** `https://github.com/Sinchana-B-Kurkuri/research-monitoring-agent`\
**Generated:** October 15, 2026\
**Evaluation Standard:** Roadmap Section 8 (Evaluation Plan), Phase 12 Experiment Suite, and Phase 13 Final Evaluation.

---

## 1. Ground Truth Dataset & Sampling Stratification

The evaluation foundation uses a stratified sampling design on arXiv papers (`cs.CL`, `cs.LG`, `cs.AI`) published between 2020 and 2024.

### 1.1 Sampling Frame & Strata Allocation
- **Total Processed Corpus:** 5,000 reference papers + 5,000 non-reference evaluation frame papers = 10,000 papers.
- **Reference Sample:** 5,000 papers drawn with fixed RNG seed (`seed=42`). Reference papers are strictly isolated to calibrate percentile thresholds and are never presented for human labeling.
- **Evaluation Sampling Frame:** 5,000 non-reference papers scored against the researcher's seed interest profile vector $\mathbf{v}_0$ (`SentenceTransformer('all-MiniLM-L6-v2')`, 384-d, L2-normalized).
- **Sampling Strata Allocation:**
  The evaluation frame is partitioned into five percentile bands based on score percentiles of the reference distribution:

| Stratum | Percentile Band | Population ($N_h$) | Target Sample ($n_h$) | Actual Labeled | Sampling Weight ($w_h = N_h / n_h$) | Default Region |
|---|---|---|---|---|---|---|
| **S1** | $\ge P98$ | 100 | 100 | 84 | 1.1905 | HIGH |
| **S2** | $P90 - P98$ | 400 | 120 | 38 | 10.5263 | BORDERLINE |
| **S3** | $P75 - P90$ | 750 | 100 | 20 | 37.5000 | LOW |
| **S4** | $P50 - P75$ | 1,250 | 80 | 15 | 83.3333 | LOW |
| **S5** | $< P50$ | 2,500 | 100 | 14 | 178.5714 | LOW |
| **Total** | — | **5,000** | **500** | **171** | — | — |

*Note: Sampling weights guarantee exact finite-population reproduction: $\sum_{h=1}^5 n_h w_h = 5,000$.*

### 1.2 Ground Truth Label Distribution
Human labeling used a three-point taxonomy (`relevant`, `somewhat`, `not_relevant`).
- **Primary Binary Mapping:** `relevant = 1`, `somewhat = 0`, `not_relevant = 0` (used for all primary metric reporting and the simulated feedback oracle).
- **Lenient Binary Mapping:** `relevant = 1`, `somewhat = 1`, `not_relevant = 0` (used for sensitivity analysis).

| Split | Relevant ($y=1$) | Somewhat ($y_{\text{lenient}}=1$) | Not Relevant ($y=0$) | Total Labels |
|---|---|---|---|---|
| **Development Split (70%)** | 70 | 25 | 26 | **121** |
| **Frozen Test Split (30%)** | 33 | 8 | 9 | **50** |
| **Total Corpus Ground Truth** | **103** | **33** | **35** | **171** |

**Verification of Quotas:**
- Overall relevant labels: $103 \ge 100$ (Target Met).
- Test split relevant labels: $33 \ge 30$ (Target Met).
- Frozen test split integrity: Split created with `seed=42`. Test set access is strictly audited in `experiments/results/test_access.log`.

---

## 2. Experiment E1: Gate Quality vs. Baselines (Test Split, Weighted)

**Evaluation Protocol:** Evaluated on the frozen test split ($n=50$, representing population $N=1,500$). All metrics are weighted by sampling stratum weights $w_h$. Confidence intervals (95%) are estimated using stratified bootstrap with 1,000 resamples.
- **Source:** [`experiments/results/final_e1_gate_quality.csv`](../../../experiments/results/final_e1_gate_quality.csv)

| Model / Baseline | Matched Vol. | Precision | Recall | F1 Score | PR-AUC (95% CI) | ROC-AUC | LOW Miss Rate | Borderline Share | Borderline Composition (% Rel / % Some / % Not) |
|---|---|---|---|---|---|---|---|---|---|
| **Three-Way Gate (HIGH)** | 30 | **0.600** | 0.317 | **0.414** | **0.557** [0.439, 0.706] | **0.978** | **0.000** | 7.4% | 41.7% / 8.3% / 50.0% |
| **B0: Random Selection** | 30 | 0.048 | 0.179 | 0.076 | 0.063 [—, —] | 0.503 | — | — | — |
| **B1: TF-IDF Cosine** | 30 | 0.397 | **0.428** | 0.412 | 0.501 [—, —] | 0.956 | — | — | — |
| **B2: Seed Embedding Cosine** | 30 | **0.600** | 0.317 | **0.414** | **0.557** [—, —] | **0.978** | — | — | — |
| **B4: TF-IDF + LogReg** | 30 | 0.541 | 0.421 | 0.473 | 0.517 [—, —] | 0.965 | — | — | — |

### Key E1 Findings:
1. **Zero Discard of Relevant Papers (LOW Miss Rate = 0.0%):** The cold-start gate with calibrated thresholds ($P_{\text{high}}=98.0, P_{\text{low}}=90.0$) routed zero relevant test papers to LOW. All relevant papers were either auto-delivered (HIGH) or escalated to human review (BORDERLINE).
2. **Superior Precision over Lexical Baselines:** The embedding gate achieved 0.600 precision compared to 0.397 for TF-IDF cosine (B1) and 0.048 for random retrieval (B0).
3. **Borderline Queue Efficiency:** Only 7.4% of candidate papers fell into the borderline review queue, concentrating ambiguous papers (41.7% relevant, 8.3% somewhat relevant, 50.0% non-relevant) for human attention.

---

## 3. Experiment E2 & E3: Learning Curves and Signal Ablation

**Evaluation Protocol:** Offline simulation with simulated-feedback oracle on labeled development pool. 10 random seeds per sample size. Evaluated on frozen test split.
- **Protocol B (Gate-Selected):** User signals drawn from HIGH papers, reviewer signals from BORDERLINE papers.
- **Source:** [`experiments/results/e3_signal_ablation.csv`](../../../experiments/results/e3_signal_ablation.csv)

| Feedback Count ($n$) | Signal Source | Mean PR-AUC | Std PR-AUC | Mean ROC-AUC | Mean F1 Score | Evaluation Type |
|---|---|---|---|---|---|---|
| **10** | User Only (👍/👎 on HIGH) | 0.717 | 0.010 | 0.874 | 0.282 | Simulated (Protocol B) |
| **10** | Reviewer Only (Accept/Reject on Borderline) | **0.732** | 0.020 | **0.891** | 0.262 | Simulated (Protocol B) |
| **10** | Combined (User + Reviewer) | 0.723 | 0.016 | 0.878 | 0.239 | Simulated (Protocol B) |
| **20** | User Only | 0.717 | 0.010 | 0.875 | 0.272 | Simulated (Protocol B) |
| **20** | Reviewer Only | **0.744** | 0.014 | **0.898** | 0.284 | Simulated (Protocol B) |
| **20** | Combined | 0.733 | 0.014 | 0.885 | 0.263 | Simulated (Protocol B) |
| **30** | User Only (switches to LogReg) | 0.425 | 0.080 | 0.629 | 0.400 | Simulated (Protocol B) |
| **30** | Reviewer Only (switches to LogReg) | 0.584 | 0.056 | 0.750 | **0.471** | Simulated (Protocol B) |
| **30** | Combined (switches to LogReg) | **0.609** | 0.067 | **0.767** | 0.464 | Simulated (Protocol B) |
| **50** | User Only | 0.492 | 0.046 | 0.678 | 0.402 | Simulated (Protocol B) |
| **50** | Reviewer Only | 0.624 | 0.053 | 0.786 | 0.483 | Simulated (Protocol B) |
| **50** | Combined | **0.682** | 0.056 | **0.826** | **0.540** | Simulated (Protocol B) |

### Key E3 Findings:
1. **Value of Borderline Review Signals:** Reviewer decisions on borderline cases provide higher information density than user reactions on high-confidence papers. At $n=20$, reviewer-only feedback achieved PR-AUC 0.744 vs. 0.717 for user-only.
2. **Synergy at Scale ($n=50$):** Combining both feedback streams resolved the selection bias of high-scoring papers, yielding the highest overall F1 score (0.540) and PR-AUC (0.682).

---

## 4. Experiment E4: Preference Adaptation — Centroid vs. Classifier

**Evaluation Protocol:** Comparing Rocchio centroid adaptation ($\alpha=1.0, \beta=0.75, \gamma=0.25$) against supervised Logistic Regression ($C \in \{0.1, 0.3, 1.0\}$, balanced weights) at matched feedback budgets.
- **Source:** [`experiments/results/e4_centroid_update.csv`](../../../experiments/results/e4_centroid_update.csv)

| Feedback Count ($n$) | Model | Mean PR-AUC | Std PR-AUC | Mean ROC-AUC | Mean F1 Score | Evaluation Type |
|---|---|---|---|---|---|---|
| **0** | **B2: Fixed Seed Vector** | 0.696 | 0.000 | 0.860 | 0.102 | Baseline |
| **10** | **B3: Updated Centroid (Rocchio)** | **0.738** | 0.020 | **0.896** | 0.255 | Simulated (Protocol A) |
| **10** | **B5: Classifier (Embedding LogReg)** | 0.600 | 0.102 | 0.797 | **0.503** | Simulated (Protocol A) |
| **20** | **B3: Updated Centroid (Rocchio)** | **0.738** | 0.020 | **0.892** | 0.266 | Simulated (Protocol A) |
| **20** | **B5: Classifier (Embedding LogReg)** | 0.675 | 0.083 | 0.845 | **0.608** | Simulated (Protocol A) |
| **30** | **B3: Updated Centroid (Rocchio)** | **0.741** | 0.011 | **0.896** | 0.288 | Simulated (Protocol A) |
| **30** | **B5: Classifier (Embedding LogReg)** | 0.690 | 0.051 | 0.866 | **0.632** | Simulated (Protocol A) |
| **50** | **B3: Updated Centroid (Rocchio)** | **0.743** | 0.010 | **0.897** | 0.283 | Simulated (Protocol A) |
| **50** | **B5: Classifier (Embedding LogReg)** | 0.726 | 0.042 | 0.893 | **0.678** | Simulated (Protocol A) |

### Key E4 Findings:
1. **Low-Sample Robustness of Centroid:** At small feedback sample sizes ($n \le 20$), Rocchio centroid updating delivers immediate, low-variance ranking improvements (PR-AUC 0.738) without overfitting risk.
2. **Classifier Superiority in Decision Boundary Sharpness:** Once $n \ge 30$, Logistic Regression learns a much sharper decision boundary, outperforming the centroid on F1 score (0.632 vs. 0.288 at $n=30$, and 0.678 vs. 0.283 at $n=50$). This strongly validates the phased architecture (seed $\to$ centroid $\to$ classifier).

---

## 5. Experiment E5: Computational & LLM Summarization Efficiency

**Evaluation Protocol:** Comparing gated execution against an unmitigated "summarize-everything" baseline.
- **Source:** [`experiments/results/e5_efficiency.csv`](../../../experiments/results/e5_efficiency.csv) and [`final_efficiency.csv`](../../../experiments/results/final_efficiency.csv)

### 5.1 Controlled Benchmark (1,000 Candidate Papers)
- **Scored Candidates:** 1,000 papers.
- **Gating Routing Distribution:**
  - HIGH: 20 papers (2.0%) $\to$ Summarized and delivered.
  - BORDERLINE: 80 papers (8.0%) $\to$ 15 reviewed (queue cap), 65 overflow.
  - LOW: 900 papers (90.0%) $\to$ Discarded.
- **LLM Calls:** 20 calls (gated) vs. 1,000 calls (unmitigated).
- **LLM Call Reduction:** **98.0%**.
- **Estimated Cost:** $0.00128 (gated) vs. $0.06375 (unmitigated) $\to$ **98.0% cost reduction**.

### 5.2 Live System Pilot Telemetry
- **Candidate Papers Scored:** 160 papers across 4 evaluated cycles.
- **Summaries Generated:** 0 (due to cold-start scoring distributions and absent production token logs).
- **Measured Token Telemetry:** **Absent** in live database.
- **Formal Status:** As enforced by the Phase 13 evaluation audit, measured live cost savings are designated **`NOT_MEASURABLE`**. The theoretical gating reduction of 1.0 (100% of candidate papers gated) is reported separately with full disclosure.

---

## 6. Experiment E6: Value of Human Review on Borderline Papers

**Evaluation Protocol:** Evaluating decision strategies on borderline candidates ($n=4$ test sample borderline subset with ground truth).
- **Source:** [`experiments/results/e6_human_review_value.csv`](../../../experiments/results/e6_human_review_value.csv)

| Strategy | Papers Delivered | Human Reviews Required | Precision | Recall | F1 Score | Description |
|---|---|---|---|---|---|---|
| **Auto-Reject Borderline (HIGH Only)** | 2 | 0 | **1.000** | 0.667 | 0.800 | Discards all borderline papers without review (Precision-oriented). |
| **Auto-Accept Borderline (HIGH + BORDERLINE)** | 4 | 0 | 0.750 | **1.000** | 0.857 | Delivers all borderline papers without filter (Recall-oriented). |
| **Relevance Gate with Human Review** | 3 | 2 | **1.000** | **1.000** | **1.000** | Escalates borderline papers to human review. |

### Key E6 Findings:
- Human review on borderline candidates successfully recovered false negatives while filtering out false positives, achieving perfect Precision (1.000) and Recall (1.000) on the test subset with minimal review burden (2 reviews).

---

## 7. Pilot Execution & Participant Classification (E7)

### 7.1 Database Routing Decisions Reconciliation (210 Total Records)
All 210 routing decisions in the database export have been audited and reconciled into four mutually exclusive partitions:

| Partition | Record Count | Filter Criteria | Purpose |
|---|---|---|---|
| **Clean Live Pilot Records** | **20** | `mode='live'`, non-dev, non-simulated, non-outage | Live pilot operational analysis |
| **Genuine Pilot Replay Records** | **100** | `mode='replay'`, non-dev, non-simulated | Historical replay simulations |
| **Developer Records** | **50** | `user_id='dev_phase5'` | Pipeline integration verification |
| **Simulated Records** | **40** | `user_id='simulated_researcher'` (`is_simulated=True`) | Feedback oracle & learning curves |
| **Partition Sum** | **210 = 210** | **Exact Match (`is_reconciled = True`)** | Complete audit trail |

### 7.2 Participant Classification Audit
- **`dev_phase5`:** Correctly classified as **Developer Account** (`is_developer = True`). Segregated from pilot analysis.
- **`simulated_researcher`:** Correctly classified as **Simulated Agent** (`is_simulated = True`). Segregated from pilot analysis.
- **`phase5_verify`:** Classified as **Unresolved** in audit.
  - Name: `"Phase 5 Verification"`.
  - Database Flags: `is_simulated = False`, `is_developer = False`.
  - Activity: 3 cycles (1 live, 2 replay) across 2 active days; **0 feedback events**, **0 model versions**, remained on cold-start seed vector.
  - **Exclusion Status:** Excluded from H1/H2 before/after comparative testing under Section 8.6 criteria:
    - Active days: $2 < 5$ (insufficient active days).
    - Effective labels: $0 < 30$ (insufficient training labels).
    - Vector activation: None (remained on version 0).

### 7.3 Pilot Hypotheses Evaluation Status (E7)
- **Hypothesis H1 (Digest Like Rate Post $\ge$ Pre):** `PENDING / EXCLUDED` (0 eligible participants completed vector/model promotion).
- **Hypothesis H2 (Review Burden Does Not Increase):** `PENDING / EXCLUDED` (0 eligible participants completed vector/model promotion).
- **Hypothesis H3 (User Satisfaction Median $\ge 4/5$):** `PENDING` (0 survey responses submitted).
- **Readiness Tier:** **`EVALUATION_PARTIALLY_AVAILABLE`** (`can_freeze_final_numbers = False`).
