# FIGURE AND DIAGRAM SPECIFICATIONS FOR ACADEMIC PAPER

**Repository:** `https://github.com/Sinchana-B-Kurkuri/research-monitoring-agent`\
**Purpose:** Precise LaTeX / TikZ and graphic specifications for all paper figures.

---

## Figure 1: System Architecture and State-Machine Workflow
- **Placement:** Section 3 (System Architecture and Agentic Workflow)
- **Caption:** *Figure 1. End-to-end architecture of the Relevance-Gated Agentic Workflow, illustrating the three-way relevance gate, asynchronous human review escalation, dual-tier preference memory, and gated LLM summarization.*
- **LaTeX / TikZ Code Specification:**
```latex
\begin{figure*}[t]
\centering
\begin{tikzpicture}[node distance=1.5cm, auto,
    block/.style={rectangle, draw, fill=blue!10, text width=2.8cm, text centered, rounded corners, minimum height=1cm, font=\small},
    decision/.style={diamond, draw, fill=orange!15, text width=2.2cm, text centered, font=\small},
    terminal/.style={rectangle, draw, fill=gray!20, text width=2.5cm, text centered, rounded corners, minimum height=0.8cm, font=\small},
    line/.style={draw, -latex', thick}]

    \node [block] (input) {User Profile ($\mathbf{v}_0$) \& arXiv Stream};
    \node [block, right of=input, node distance=3.8cm] (embed) {Dense Embedding ($384$-d, $L_2$-norm)};
    \node [decision, right of=embed, node distance=4.2cm] (gate) {Relevance Gate\\($t_{\text{high}}, t_{\text{low}}$)};

    \node [block, above right of=gate, node distance=3.2cm] (high) {HIGH Route\\Auto-Deliver};
    \node [block, right of=gate, node distance=3.5cm] (border) {BORDERLINE\\Review Queue (Cap 15)};
    \node [terminal, below right of=gate, node distance=3.2cm] (low) {LOW Route\\Discard \& Audit Log};

    \node [block, right of=high, node distance=3.5cm] (summary) {Summary Agent\\(Interest-Framed LLM)};
    \node [block, right of=summary, node distance=3.2cm] (digest) {Daily Digest\\(User $\pmb{+}/\pmb{-}$ Feedback)};

    \node [block, below of=digest, node distance=2.5cm] (pref) {Preference Memory\\(SQL View)};
    \node [block, below of=pref, node distance=2.0cm] (adapt) {Rocchio Centroid $\to$\\LogReg Classifier};

    \path [line] (input) -- (embed);
    \path [line] (embed) -- (gate);
    \path [line] (gate) |- node [above left] {$s \ge t_{\text{high}}$} (high);
    \path [line] (gate) -- node [above] {Ambiguous} (border);
    \path [line] (gate) |- node [below left] {$s \le t_{\text{low}}$} (low);
    \path [line] (high) -- (summary);
    \path [line] (border) |- node [above left] {Accept} (summary);
    \path [line] (summary) -- (digest);
    \path [line] (digest) -- (pref);
    \path [line] (border) |- node [below] {Reject / Expire} (pref);
    \path [line] (pref) -- (adapt);
    \path [line] (adapt) -| node [near end, below] {Updated $\mathbf{v}_{t+1}$ / Thresholds} (gate);
\end{tikzpicture}
\caption{End-to-end architecture of the Relevance-Gated Agentic Workflow.}
\label{fig:architecture}
\end{figure*}
```

---

## Figure 2: Signal Ablation Learning Trajectories (E3)
- **Placement:** Section 6.2 (Learning Curves & Signal Ablation)
- **Caption:** *Figure 2. Precision-Recall AUC (PR-AUC) learning curves across feedback budgets ($n \in \{10, 20, 30, 50\}$) comparing User-Only, Reviewer-Only, and Combined feedback streams under Protocol B (10 random seeds). Error bands indicate $\pm 1$ standard deviation.*
- **Existing Generated Graphic:** [`experiments/results/e3_signal_ablation.png`](../../../experiments/results/e3_signal_ablation.png)
- **Visual Description:**
  - X-axis: Feedback Budget ($n$).
  - Y-axis: Mean PR-AUC on Frozen Test Split.
  - Trajectories: Reviewer-only (green) dominates user-only (red) in the low-sample regime ($n \le 20$). At $n=50$, Combined (blue) achieves the highest PR-AUC (0.682), demonstrating synergistic coverage across boundary and high-confidence regions.

---

## Figure 3: Preference Adaptation Trajectories: Centroid vs. Classifier (E4)
- **Placement:** Section 6.3 (Centroid Update vs. Supervised Classifier)
- **Caption:** *Figure 3. Comparison of ranking quality (PR-AUC) and decision sharpness (F1 Score) between Rocchio Centroid Adaptation (B3) and L2-regularized Logistic Regression (B5) across sample sizes $n \in \{0, 10, 20, 30, 50\}$ (10 random seeds).*
- **Existing Generated Graphic:** [`experiments/results/e4_centroid_update.png`](../../../experiments/results/e4_centroid_update.png)
- **Visual Description:**
  - Left panel: PR-AUC vs. $n$. Centroid provides immediate, low-variance gain from 0.696 to 0.738 at $n=10$.
  - Right panel: F1 score vs. $n$. Logistic regression starts with higher variance at $n=10$ but dramatically overtakes the centroid at $n=50$ (F1 = 0.678 vs. 0.283), justifying the phased deployment strategy.

---

## Figure 4: Computational and LLM Call Reduction (E5)
- **Placement:** Section 6.4 (Computational & Token Efficiency)
- **Caption:** *Figure 4. Comparison of LLM summarization calls and estimated API expenditures between unmitigated "summarize-everything" processing and relevance gating across 1,000 candidate papers.*
- **Existing Generated Graphic:** [`experiments/results/e5_efficiency.png`](../../../experiments/results/e5_efficiency.png)
- **Visual Description:**
  - Bar chart contrasting 1,000 LLM calls ($0.06375) against 20 gated calls ($0.00128), visually illustrating the 98.0% reduction in API calls and financial cost.

---

## Figure 5: Value of Human Escalation on Borderline Papers (E6)
- **Placement:** Section 6.5 (Value of Human Review)
- **Caption:** *Figure 5. Precision, Recall, and F1 trade-offs of Auto-Reject, Auto-Accept, and Human-in-the-Loop review strategies on borderline candidate publications.*
- **Existing Generated Graphic:** [`experiments/results/e6_human_review_value.png`](../../../experiments/results/e6_human_review_value.png)
- **Visual Description:**
  - Clustered column chart showing Precision (1.000), Recall (1.000), and F1 (1.000) for the proposed human-in-the-loop strategy against Auto-Reject (Recall 0.667) and Auto-Accept (Precision 0.750).
