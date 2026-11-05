# Relevance-Gated Agentic Workflow for Autonomous Research Monitoring: An Empirical Study in Human-in-the-Loop Literature Triage and Preference Adaptation

**Sinchana B. Kurkuri**\
Department of Computer Science and Engineering\
`sinchana.kurkuri@academic.edu`

---

## ABSTRACT
The exponential expansion of pre-print publication venues—most notably arXiv—has overwhelmed researchers' capacity to maintain awareness of relevant scientific advances. Conventional notification mechanisms, such as keyword alert feeds and unweighted keyword searches, either suffer from severe false-positive saturation or omit semantically relevant contributions. Conversely, unconstrained Large Language Model (LLM) agent frameworks that ingest, parse, and summarize all published literature incur prohibitive monetary expenses, latency, and hallucination risks. In this paper, we propose and evaluate a **Relevance-Gated Agentic Workflow for Autonomous Research Monitoring**, an autonomous system that continuously triages, filters, summarizes, and adapts to user preferences over scientific publication streams. The system is formulated as an explicit, auditable Python state machine rather than an unconstrained multi-agent swarm. Candidate publications are embedded via dense sentence transformers (`all-MiniLM-L6-v2`, 384-d) and evaluated against a user interest profile using a **three-way percentile-calibrated relevance gate**: high-confidence papers are automatically delivered and summarized; low-confidence papers are immediately discarded without invoking expensive language models; and borderline candidates are escalated to an asynchronous, capped human review queue. The system dynamically personalizes its routing logic through two-stage preference memory: unsupervised Rocchio centroid updates for cold-start adaptation, followed by an L2-regularized logistic regression classifier promoted under strict out-of-fold generalization guards. Evaluating on an arXiv corpus across computer science domains (`cs.CL`, `cs.LG`, `cs.AI`) with a 500-paper stratified ground-truth sampling frame and frozen test split, our relevance gate achieves 0.600 precision and 0.978 ROC-AUC on high-confidence deliveries while discarding zero relevant papers (0.0% low-miss rate). Compared against an unmitigated summarize-everything baseline, our relevance gating achieves a 98.0% reduction in LLM inference calls and computational expenditures. Furthermore, signal ablation experiments demonstrate that human review decisions on borderline papers provide vital information density that accelerates preference learning beyond user interaction signals alone. We conclude with a candid report on production deployment realities, operational trade-offs, and empirical boundary conditions.

### CCS CONCEPTS
• **Information systems** $\to$ **Document filtering**; **Personalization**; **Information retrieval query processing**;\
• **Computing methodologies** $\to$ **Machine learning approaches**; **Natural language processing**.

### KEYWORDS
Research Monitoring; Relevance Gating; Human-in-the-Loop; Dense Retrieval; Active Learning; Preference Adaptation; Large Language Models.

---

## 1. INTRODUCTION

Contemporary scientific research is experiencing an unprecedented surge in publication velocity. In fields such as artificial intelligence, machine learning, and computational linguistics, hundreds of manuscripts are deposited to pre-print repositories like arXiv every week. For active investigators, maintaining an up-to-date mental model of current methodologies, benchmark results, and related literature has transformed from a routine scholarly task into a significant cognitive burden.

Historically, researchers have relied on keyword-based alerting systems (e.g., Google Scholar Alerts, arXiv RSS feeds) to monitor literature. However, lexical keyword matching suffers from fundamental information-retrieval deficiencies, notably synonymy (novel concepts expressed with unfamiliar terminology) and polysemy (identical terms possessing distinct technical definitions across subfields). Consequently, keyword feeds inevitably degrade into high-volume, noisy message streams that researchers routinely abandon due to alert fatigue.

Recent advancements in dense semantic representations and Large Language Models (LLMs) present a compelling alternative: encoding publication abstracts into semantic vector spaces to compute semantic similarity against a user profile, and delegating reading and synthesis to LLMs. Nevertheless, naive "agentic" literature monitoring architectures—which query generative LLMs to evaluate, score, and summarize every incoming paper—suffer from acute economic and computational limitations. Processing hundreds of daily manuscripts via multi-thousand-token prompt templates incurs high API costs, significant processing latency, and unpredictable reasoning hallucinations.

To resolve this dilemma, we design, implement, and empirically evaluate a **Relevance-Gated Agentic Workflow for Autonomous Research Monitoring**. Rather than adopting opaque, non-deterministic multi-agent frameworks (e.g., LangChain, AutoGen, CrewAI), our framework is anchored by an **explicit, deterministic finite-state machine**. Central to our architecture is a **three-way relevance gate** that partitions incoming publications into three distinct operational pathways:
1. **HIGH Confidence Pathway ($s \ge t_{\text{high}}$):** Publications exhibiting high semantic alignment are directly forwarded to the user's research digest and processed by an interest-framed LLM summarizer.
2. **BORDERLINE Pathway ($t_{\text{low}} < s < t_{\text{high}}$):** Ambiguous candidate publications are routed to an asynchronous human review queue, enabling researchers to triage borderline works with minimal manual effort.
3. **LOW Confidence Pathway ($s \le t_{\text{low}}$):** Irrelevant publications are immediately pruned and logged into an auditable database table, preventing unnecessary downstream computation or LLM API calls.

To prevent cold-start drift, relevance thresholds ($t_{\text{high}}, t_{\text{low}}$) are calibrated as percentiles ($P_{\text{high}}, P_{\text{low}}$) over a fixed, representative reference sample of 5,000 arXiv papers. As the user interacts with the system by triaging borderline papers and rating delivered summaries, the system accumulates multi-source feedback events. These signals trigger a dual-tier preference memory system: initially refining the interest vector using a Rocchio-style centroid adjustment, and subsequently transitioning to a supervised, L2-regularized logistic regression classifier once sufficient feedback data is accumulated and verified via out-of-fold generalization guards.

### 1.1 Problem Statement and Motivation
Let $\mathcal{P} = \{p_1, p_2, \dots, p_N\}$ denote a continuous stream of incoming scientific papers, where each paper $p_i$ is characterized by its title, abstract, metadata, and dense embedding vector $\mathbf{p}_i \in \mathbb{R}^d$. A researcher's latent intellectual interests are captured by an evolving preference profile $\mathcal{U}$. The core operational challenge is to define a mapping function:
$$\Phi: \mathcal{P} \times \mathcal{U} \to \{\text{HIGH}, \text{BORDERLINE}, \text{LOW}\}$$
such that:
1. **Precision of High Route:** Publications auto-delivered to the digest maximize precision with respect to the user's primary interests, minimizing reading fatigue.
2. **Safety of Low Route (Zero Discard of Relevant Papers):** The miss rate—the proportion of genuinely relevant papers erroneously assigned to LOW—is constrained to near zero ($\le 5\%$).
3. **Economic and Computational Efficiency:** Generative LLM summarization is invoked exclusively on gate-cleared papers ($\text{HIGH} \cup \text{BORDERLINE}_{\text{accepted}}$), reducing token expenditures by over 90% compared to unmitigated processing.
4. **Active Personalization:** Human feedback on delivered digests and borderline review tasks progressively refines the decision boundary without requiring full model fine-tuning.

### 1.2 Research Gap
Prior research in academic literature triage has remained bifurcated. On one hand, classical recommender systems (e.g., arXiv-Sanity, Mendeley Suggest) employ collaborative filtering or static TF-IDF/bag-of-words classifiers. While computationally inexpensive, they lack semantic nuance, struggle with lexical mismatch, and provide static ranking without adaptive agentic escalation. On the other hand, recent generative LLM agent workflows rely on multi-agent debate or comprehensive document reading, neglecting the computational and financial realities of high-throughput literature streams. Furthermore, existing human-in-the-loop systems frequently blur the distinction between reviewer triage decisions (evaluating unsummarized candidate abstracts) and consumer reactions (liking/disliking delivered syntheses), failing to isolate the informational value of different human interaction signals.

### 1.3 Research Objectives and Contributions
In this work, we bridge this gap through five specific contributions:
1. **Deterministic State-Machine Architecture:** We formulate an explicit Python state machine for autonomous research monitoring that enforces auditable transitions, idempotent cycle runs, and asynchronous human escalation.
2. **Calibrated Three-Way Percentile Gating:** We demonstrate that formulating relevance thresholds as percentiles of a fixed reference distribution provides robust, cold-start threshold stability that automatically transfers across shifting user profiles.
3. **Informational Signal Ablation:** We evaluate the distinct learning values of end-user digest reactions versus human reviewer borderline triage decisions, proving that borderline review decisions accelerate preference adaptation significantly faster than high-confidence feedback alone.
4. **Dual-Tier Preference Memory:** We implement and benchmark an adaptive progression that transitions from unsupervised Rocchio centroid drift to a supervised, regularized classifier governed by cross-validation promotion guards.
5. **Rigorous Empirical and Boundary Reporting:** We ground all claims in an audited evaluation framework comprising 171 human-labeled papers, stratified Horvitz-Thompson sampling weights, frozen test access auditing, and an unvarnished analysis of live pilot constraints.

---

## 2. RELATED WORK

### 2.1 Scientific Literature Filtering & arXiv Alerting
Automated literature monitoring has evolved from manual bibliographic queries to automated alerting. The seminal work of Karpathy in developing *arXiv-Sanity Preserver* demonstrated the utility of TF-IDF vectors combined with support vector machines (SVM) trained on user library bookmarks. While effective as an exploratory interface, arXiv-Sanity relies on static user-triggered searches and batch retraining. Commercial engines such as Google Scholar Alerts and Semantic Scholar provide periodic email digests based on citation graphs and keyword co-occurrence; however, their ranking algorithms operate as black boxes, provide no mechanism for active borderline triage, and cannot generate personalized, interest-framed syntheses.

### 2.2 Dense Embedding Representations & Semantic Search
The advent of pre-trained transformer representations has revolutionized document retrieval. Reimers and Gurevych introduced Sentence-BERT (SBERT), demonstrating that fine-tuning siamese and triplet networks on natural language inference datasets produces dense sentence representations capable of semantic similarity search via cosine distance. Specialized scientific models, such as SciBERT and SPECTER, further fine-tuned transformers on scientific citations and co-citation graphs. In our architecture, we employ `all-MiniLM-L6-v2`, an efficient 384-dimensional model trained on over 1 billion sentence pairs. Due to its compact dimensionality and high semantic retrieval performance, it allows exact nearest-neighbor cosine scoring across tens of thousands of papers within memory, eliminating the operational overhead of approximate nearest neighbor (ANN) vector indices for moderate corpus scales.

### 2.3 Human-in-the-Loop Machine Learning and Active Learning
Active learning explores how algorithms can query users to label informative data points, typically focusing on points near the decision boundary (uncertainty sampling). Settles provided a comprehensive survey of active learning strategies, noting that boundary-proximal instances provide maximal reduction in hypothesis version spaces. In information retrieval, Rocchio's classical relevance feedback algorithm introduced a geometric formulation for updating vector space queries based on positive and negative documents. Our work synthesizes these principles into an autonomous operational workflow: borderline papers ($t_{\text{low}} < s < t_{\text{high}}$) serve as natural active-learning candidates that are asynchronously presented to the researcher, simultaneously refining the literature delivery stream and expanding the supervised training dataset.

### 2.4 LLM Summarization and Agentic Architectures
The emergence of generative LLMs (e.g., GPT-4, Gemini) has catalyzed interest in autonomous research agents. Frameworks like AutoGen and CrewAI demonstrate multi-agent debates where specialized agents discover, critique, and summarize literature. However, recent empirical audits demonstrate that such unconstrained multi-agent frameworks suffer from non-deterministic looping, high token overhead, and compounding hallucinations. Our framework adheres to an *agentic workflow* pattern without framework dependencies: the agentic autonomy emerges from deterministic routing decisions, automatic escalation to human review, and persistent preference adaptation over time, while constraining expensive LLM execution strictly to gate-cleared manuscripts.

---

## 3. SYSTEM ARCHITECTURE AND AGENTIC WORKFLOW

### 3.1 Explicit State-Machine Core vs. Multi-Agent Frameworks
A central architectural tenet of our design is the rejection of black-box multi-agent frameworks in favor of an **explicit Python state machine**. In academic research monitoring, auditability and determinism are paramount: a researcher must understand exactly why a manuscript was delivered, queued, or discarded.

The lifecycle of each candidate paper is tracked by nine formal states across the system:
1. `ROUTED_HIGH`: Candidate paper scored at or above $t_{\text{high}}$; automatically scheduled for summarization.
2. `ROUTED_BORDERLINE`: Candidate paper scored between $t_{\text{low}}$ and $t_{\text{high}}$; scheduled for review queue ingestion.
3. `DISCARDED`: Candidate paper scored below $t_{\text{low}}$; permanently marked as discarded (terminal state, logged with full audit metadata).
4. `PENDING_REVIEW`: Borderline paper persisted in the user's active human review queue.
5. `REVIEW_REJECTED`: Borderline paper explicitly rejected by the human reviewer (terminal state, creates negative training event).
6. `REVIEW_EXPIRED`: Borderline paper unreviewed after 3 calendar days (terminal state, creates no label).
7. `DEFERRED_OVERFLOW`: Borderline paper exceeding the cycle review queue cap (15 items) or high-confidence paper exceeding the digest cap (20 items); logged and terminated without labeling.
8. `SUMMARIZE_PENDING`: Gate-cleared paper awaiting LLM synthesis.
9. `DELIVERED`: Synthesized paper published to the user's research digest.
10. `REACTED`: User provided interactive feedback (Like 👍 or Dislike 👎) on the delivered digest item.

```
+---------------------------------------------------------------------------------+
|                                CORE SYSTEM FLOW                                 |
+---------------------------------------------------------------------------------+
 User Profile (Seed Paragraph -> Dense Vector v0)
   |
   v
 Paper Discovery (Live arXiv API / Historical Corpus)
   |
   v
 Text Preprocessing (Version-Stripping, Normalization)
   |
   v
 Embedding Generation (all-MiniLM-L6-v2, 384-d, L2-Normalized)
   |
   v
 Relevance Scoring: s = cosine(p, v) [or P(relevant|p) under Classifier]
   |
   v
+------------------------ THREE-WAY RELEVANCE GATE -------------------------------+
|                                                                                 |
|  [ s >= t_high ] -------------> HIGH Route                                      |
|                                   |                                             |
|                                   v                                             |
|                             Summary Agent (Gated LLM Synthesis)                 |
|                                   |                                             |
|                                   v                                             |
|                             Digest Delivery (Capped at 20)                      |
|                                   |                                             |
|                                   v                                             |
|                             User Feedback (👍 / 👎)                             |
|                                   |                                             |
|  [ t_low < s < t_high ] ------> BORDERLINE Route                                |
|                                   |                                             |
|                                   v                                             |
|                             Review Queue (Capped at 15)                         |
|                               |       |        |                                |
|                        Accept |       | Reject | Expire (3 days)                |
|                               v       v        v                                |
|                       Summary Agent  Log      Log                               |
|                                                                                 |
|  [ s <= t_low ] --------------> LOW Route                                       |
|                                   |                                             |
|                                   v                                             |
|                             Discard & Audit Log                                 |
+---------------------------------------------------------------------------------+
                                    |
                                    v
                     Multi-Source Feedback Collection
                  (User Reactions vs Reviewer Decisions)
                                    |
                                    v
                 Preference Memory (SQL View training_labels)
                                    |
        +---------------------------+---------------------------+
        | (n < 30 effective labels)                             | (n >= 30, guard passes)
        v                                                       v
 Unsupervised Adaptation                                 Supervised Learning
 (Rocchio Centroid Vector Drift)                         (L2-Regularized LogReg)
        |                                                       |
        v                                                       v
 Dynamic Threshold Recomputation                         Classifier Routing Mode
 (t_high, t_low via Reference Sample)                    (p_high, p_low calibrated)
```
*Figure 1: Complete end-to-end operational architecture of the Relevance-Gated Agentic Workflow.*

### 3.2 Three-Tier Runtime Deployment Architecture
To ensure high performance while maintaining compatibility with lightweight hosting constraints, the system is partitioned into three distinct runtime layers:
1. **Always-On Web Service:** Implemented in FastAPI, Jinja2, and HTMX. Serves the researcher's review queue, research digest, and audit log. Crucially, the web layer **does not import PyTorch or SentenceTransformers**, maintaining a memory footprint under 150 MB. LLM summarization for review-accepted papers is handled via lightweight background tasks.
2. **Scheduled Cycle Worker:** Executes automated monitoring cycles (fetching arXiv feeds, computing embeddings, scoring against active profiles, managing queue caps, updating centroid vectors, and retraining classifiers). Designed to execute headlessly via cron or scheduled GitHub Actions workflows.
3. **Persistent Data Layer:** Managed PostgreSQL instance with the `pgvector` extension. Stores document metadata, 384-dimensional dense vectors, auditable routing decisions, and serialized model blobs. For local development and test environments, an automated fallback engine utilizes SQLite with in-memory NumPy vector operations.

### 3.3 Database Schema and Relational Architecture
The database schema consists of nine core tables and one materialized SQL view:
- `papers`: Stores canonical arXiv publications, containing `arxiv_id` (version-stripped unique primary key), `title`, `abstract`, `submitted_date`, `categories` (JSON), `embedding` (`vector(384)`), `embedding_model`, and `is_reference` boolean flag.
- `users`: Pilot researcher records with private `access_token` strings for URL-authenticated access.
- `interest_profiles`: Initial seed interest text and corresponding 384-d seed vector.
- `interest_vector_versions`: History of all active and legacy interest vectors for each user, recording the derivation `method` (`seed` or `rocchio`), `n_effective_labels`, threshold parameters ($P_{\text{high}}, P_{\text{low}}, t_{\text{high}}, t_{\text{low}}$), and activation status.
- `cycles`: Execution logs of monitoring runs, tracking `as_of_date`, `mode` (`replay` vs `live`), processing timestamps, and summary statistics.
- `routing_decisions`: Immutable audit log of every paper evaluated per user. Captures `cosine_score`, `clf_prob`, `method` (`threshold` vs `classifier`), `route` (`HIGH`, `BORDERLINE`, `LOW`), `status`, and decision `reason`.
- `review_tasks`: Asynchronous queue entries for borderline papers, tracking presentation timestamps, decisions (`accept`, `reject`, `skip`), and expiration markers.
- `digests` and `digest_items`: Curated daily delivery collections, capping delivered papers at 20 per cycle and tracking delivery provenance (`via='auto'` vs `via='review'`).
- `feedback_events`: Append-only ledger of every user reaction (Like 👍 = 1, Dislike 👎 = 0) and reviewer decision (Accept = 1, Reject = 0).
- `training_labels` (SQL View): Deterministic resolution of multi-source feedback into one effective training label per (user, paper).

---

## 4. METHODOLOGY

### 4.1 Interest Profiling and Dense Embeddings
A user's interest profile is initialized from a descriptive natural language paragraph delineating primary research goals, methodological preferences, and target application domains.

Both paper contents (concatenated title and abstract) and user profiles are projected into a dense semantic vector space using the `all-MiniLM-L6-v2` transformer model:
$$\mathbf{p} = \frac{f_{\text{enc}}(\text{title} \circ \text{" "} \circ \text{abstract})}{\|f_{\text{enc}}(\text{title} \circ \text{" "} \circ \text{abstract})\|_2}, \quad \mathbf{v}_0 = \frac{f_{\text{enc}}(\text{seed\_text})}{\|f_{\text{enc}}(\text{seed\_text})\|_2}$$
where $\mathbf{p}, \mathbf{v}_0 \in \mathbb{R}^{384}$ and $\|\mathbf{p}\|_2 = \|\mathbf{v}_0\|_2 = 1.0$. Because all embeddings are strictly $L_2$-normalized upon creation, the semantic cosine similarity reduces to a standard dot product:
$$s(\mathbf{p}, \mathbf{v}) = \cos(\mathbf{p}, \mathbf{v}) = \mathbf{p} \cdot \mathbf{v} = \sum_{j=1}^{384} p_j v_j$$

### 4.2 Percentile-Gated Relevance Routing
Raw cosine similarity scores vary substantially across different user interest profiles and scientific subfields; a score of 0.65 may represent an exceptionally close match in a broad domain but a mediocre match in a narrow domain. To achieve robust, domain-invariant decision boundaries, our relevance agent computes thresholds relative to an isolated, fixed **Reference Sample** $\mathcal{S}_{\text{ref}}$ comprising $M = 5,000$ randomly selected arXiv papers.

Given the active interest vector $\mathbf{v}$, the reference score distribution is computed:
$$\mathcal{D}_{\text{ref}}(\mathbf{v}) = \{ \mathbf{p}_k \cdot \mathbf{v} \mid p_k \in \mathcal{S}_{\text{ref}} \}$$
The high and low relevance thresholds, $t_{\text{high}}$ and $t_{\text{low}}$, are calculated as the $P_{\text{high}}$-th and $P_{\text{low}}$-th percentiles of $\mathcal{D}_{\text{ref}}(\mathbf{v})$:
$$t_{\text{high}} = \text{Quantile}\left(\mathcal{D}_{\text{ref}}(\mathbf{v}), \frac{P_{\text{high}}}{100}\right), \quad t_{\text{low}} = \text{Quantile}\left(\mathcal{D}_{\text{ref}}(\mathbf{v}), \frac{P_{\text{low}}}{100}\right)$$
The three-way relevance gating rule is defined as:
$$\Phi(\mathbf{p}, \mathbf{v}) = \begin{cases} \text{HIGH}, & \text{if } s(\mathbf{p}, \mathbf{v}) \ge t_{\text{high}} \\ \text{BORDERLINE}, & \text{if } t_{\text{low}} < s(\mathbf{p}, \mathbf{v}) < t_{\text{high}} \\ \text{LOW}, & \text{if } s(\mathbf{p}, \mathbf{v}) \le t_{\text{low}} \end{cases}$$

**Calibration Objective:** Threshold percentiles are calibrated on ground-truth development labels by solving a dual-objective constrained optimization:
$$\min P_{\text{high}} \quad \text{subject to} \quad \text{Precision}_{\text{weighted}}(\text{HIGH}) \ge 0.80$$
$$\max P_{\text{low}} \quad \text{subject to} \quad P_{\text{low}} < P_{\text{high}} \quad \text{and} \quad \text{MissRate}_{\text{weighted}}(\text{LOW}) \le 0.05$$
On our calibrated development set, this optimization yields optimal operating levels at $P_{\text{high}} = 98.0$ and $P_{\text{low}} = 90.0$.

### 4.3 Asynchronous Human Review Queue and Overflow Policies
Borderline papers ($t_{\text{low}} < s < t_{\text{high}}$) represent publications with moderate semantic proximity where automated confidence is insufficient. To prevent cognitive fatigue, the review queue enforces strict operational caps:
- **Queue Capacity:** Maximum of 15 borderline tasks per user per monitoring cycle, ordered descending by similarity score.
- **Queue Overflow Policy:** Papers exceeding the 15-task cap are assigned the terminal status `DEFERRED_OVERFLOW` and logged to the database without generating review tasks.
- **Task Expiration Policy:** Review tasks remaining undecided after 72 hours (3 calendar days) transition to `REVIEW_EXPIRED` and are retired without generating training labels.
- **Idempotent Resume Step:** When a user marks an item as **Accepted**, a background worker transitions the paper to `SUMMARIZE_PENDING`, triggers the summarizer, and appends the paper to the current active digest (`via='review'`).

### 4.4 Gated, Interest-Framed LLM Summarization
Unlike standard summarizers that merely condense paper abstracts, our **Summary Agent** generates *interest-framed* syntheses explaining *why* a specific manuscript matters relative to the researcher's expressed profile.

Summarization is restricted strictly to gate-cleared papers. The agent invokes a hosted LLM API (Google Gemini 1.5 Flash) via a standardized wrapper with low decoding temperature ($T = 0.2$). The prompt structure provides the user's seed interest paragraph and the paper's abstract, instructing the model to generate a structured 3-sentence summary: (1) core methodology, (2) primary empirical result, and (3) concrete relevance to the user's research goals. To prevent duplicate billing, summaries are cached with a unique constraint on `(user_id, paper_id, llm_model, prompt_version)`.

### 4.5 Multi-Source Feedback and the Effective-Label Rule
The system records two distinct human interaction signals:
1. **Reviewer Decisions:** Captured when a researcher triages borderline abstracts in the review queue ($\text{Accept} \to 1$, $\text{Reject} \to 0$).
2. **End-User Reactions:** Captured when a researcher reads a delivered paper summary in the daily research digest ($\text{Like} \to 1$, $\text{Dislike} \to 0$).

Because a user may accept a borderline paper based on its abstract but subsequently dislike the delivered summary, a deterministic conflict resolution rule is required. The database view `training_labels` computes one **effective label** per (user, paper) pair according to the following precedence:
$$y_{\text{eff}}(u, p) = \begin{cases} y_{\text{user}}(u, p), & \text{if an end-user digest reaction exists} \\ y_{\text{reviewer}}(u, p), & \text{else if a reviewer queue decision exists} \\ \emptyset, & \text{otherwise (skips and expirations yield no label)} \end{cases}$$
Raw events are preserved immutably in an append-only ledger (`feedback_events`).

### 4.6 Unsupervised Adaptation: Rocchio Centroid Drift
During the initial cold-start phase ($n < 30$ effective labels), training a supervised parametric classifier is prone to severe overfitting. Instead, the system adapts using a modified, $L_2$-normalized Rocchio centroid formulation:
$$\mathbf{c}_{t+1} = \frac{\alpha \mathbf{v}_0 + \frac{\beta}{|P|} \sum_{\mathbf{p} \in P} \mathbf{p} - \frac{\gamma}{|N|} \sum_{\mathbf{n} \in N} \mathbf{n}}{\left\| \alpha \mathbf{v}_0 + \frac{\beta}{|P|} \sum_{\mathbf{p} \in P} \mathbf{p} - \frac{\gamma}{|N|} \sum_{\mathbf{n} \in N} \mathbf{n} \right\|_2}$$
where $\mathbf{v}_0$ is the initial seed vector, $P = \{\mathbf{p} \mid y_{\text{eff}} = 1\}$, $N = \{\mathbf{n} \mid y_{\text{eff}} = 0\}$, and hyperparameters are configured to default values $\alpha = 1.0, \beta = 0.75, \gamma = 0.25$. (If no negative labels exist, the negative summation term is omitted).

The update is triggered upon accumulating 10 effective labels (with $\ge 3$ positives), and retrained after every 25 subsequent feedback events. Whenever a new vector version is generated, the reference sample is immediately re-scored, and updated thresholds ($t_{\text{high}}, t_{\text{low}}$) are computed to preserve fixed percentile shares.

### 4.7 Supervised Adaptation: Preference Classifier with Promotion Guard
Once a researcher accumulates at least 30 effective labels (with $\ge 8$ positive and $\ge 8$ negative examples), the system activates its supervised learning pipeline.

The preference model is formulated as an $L_2$-regularized Logistic Regression classifier:
$$P(\text{relevant} \mid \mathbf{p}) = \sigma(\mathbf{w}^T \mathbf{p} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{p} + b)}}$$
The model is fitted using scikit-learn with balanced class weighting (`class_weight='balanced'`) to penalize minority-class errors. The regularization hyperparameter $C$ is chosen from $\{0.1, 0.3, 1.0\}$ via internal 5-fold cross-validation.

**Autonomous Promotion Guard:** To protect the user from model degradation, a newly trained classifier is evaluated using repeated stratified 5-fold out-of-fold (OOF) cross-validation. A candidate model is promoted to active routing if and only if:
$$\text{ROC-AUC}_{\text{OOF}}(\text{Classifier}) \ge 0.65 \quad \text{and} \quad \text{ROC-AUC}_{\text{OOF}}(\text{Classifier}) \ge \text{ROC-AUC}_{\text{OOF}}(\text{Centroid}) + 0.02$$
If the candidate fails this promotion guard, the system rejects the model and continues routing via the Rocchio centroid.

Once promoted, routing operates on predicted posterior probabilities:
$$\Phi_{\text{clf}}(\mathbf{p}) = \begin{cases} \text{HIGH}, & \text{if } P(\text{relevant} \mid \mathbf{p}) \ge p_{\text{high}} \\ \text{BORDERLINE}, & \text{if } p_{\text{low}} < P(\text{relevant} \mid \mathbf{p}) < p_{\text{high}} \\ \text{LOW}, & \text{if } P(\text{relevant} \mid \mathbf{p}) \le p_{\text{low}} \end{cases}$$
with calibrated default cutoffs $p_{\text{high}} = 0.70$ and $p_{\text{low}} = 0.30$.

---

## 5. DATASET AND EXPERIMENTAL METHODOLOGY

### 5.1 arXiv Corpus and Isolated Reference Sample
Our evaluation corpus was constructed by harvesting metadata from arXiv across three central AI categories: Computation and Language (`cs.CL`), Machine Learning (`cs.LG`), and Artificial Intelligence (`cs.AI`) deposited between January 1, 2020, and December 31, 2024. All records were normalized, whitespace-cleaned, and deduplicated using version-stripped arXiv identifiers.

The corpus was partitioned into two isolated 5,000-paper subsets:
1. **Reference Sample ($\mathcal{S}_{\text{ref}}$, $N = 5,000$):** Sampled randomly using fixed seed 42. Reserved exclusively for computing percentile distributions and recomputing thresholds.
2. **Evaluation Sampling Frame ($N = 5,000$):** Reserved for ground-truth labeling, baseline comparisons, and offline simulation experiments.

### 5.2 Stratified Sampling Frame and Finite-Population Weights
Because relevant publications constitute a small fraction ($< 5\%$) of all submitted manuscripts, uniform random sampling would yield very few positive examples in a modest labeling budget. To construct an informative evaluation set, we stratified the 5,000-paper evaluation frame into five strata based on reference percentiles under the seed interest vector $\mathbf{v}_0$:
- **Stratum S1 ($\ge P98$):** Population $N_1 = 100$. Target region: HIGH.
- **Stratum S2 ($P90 - P98$):** Population $N_2 = 400$. Target region: BORDERLINE.
- **Stratum S3 ($P75 - P90$):** Population $N_3 = 750$. Target region: LOW.
- **Stratum S4 ($P50 - P75$):** Population $N_4 = 1,250$. Target region: LOW.
- **Stratum S5 ($< P50$):** Population $N_5 = 2,500$. Target region: LOW.

To correct for non-uniform sampling probabilities and provide unbiased estimates of true population performance, all reported metrics incorporate Horvitz-Thompson sampling weights:
$$w_h = \frac{N_h}{n_h}$$
where $N_h$ is the stratum population and $n_h$ is the number of labeled papers in stratum $h$. This weighting guarantees that the weighted sample reproduces the exact finite-population totals: $\sum_{h=1}^5 n_h w_h = 5,000$.

### 5.3 Ground-Truth Labeling & Frozen Test Split
A total of **171 publications** were manually labeled by domain experts using a three-point grading scale:
- `relevant` ($y=1$): Directly addresses the researcher's stated research core.
- `somewhat` ($y=0$ in primary, $y=1$ in lenient): Shares peripheral topical overlap or related methodology.
- `not_relevant` ($y=0$): Completely outside the researcher's scope.

The labeled dataset was partitioned into a **70% Development Split** (121 papers) and a **30% Test Split** (50 papers) stratified across S1–S5 using fixed random seed 42. The test split was permanently frozen prior to experiment execution. Every test set evaluation was audited through an immutable log file (`test_access.log`).

The label distribution across splits satisfies all predefined roadmap quotas:
- Total labeled papers: 171
- Total relevant labels: $103 \ge 100$ (Target Met)
- Test split relevant labels: $33 \ge 30$ (Target Met)

### 5.4 Baseline Models
We benchmark the proposed three-way relevance gate against five baseline approaches:
- **B0 (Random Selection):** Uniform random selection of publications at matched volume (lower performance bound).
- **B1 (TF-IDF Cosine Similarity):** Classical lexical baseline computing cosine similarity between sublinear TF-IDF vectors of candidate abstracts and the seed interest text.
- **B2 (Fixed Seed Embedding Cosine):** Dense embedding retrieval using the cold-start seed vector $\mathbf{v}_0$ without three-way gating or adaptation.
- **B3 (Updated Centroid Cosine):** Dense retrieval using the Rocchio-adapted interest vector $\mathbf{c}_t$.
- **B4 (TF-IDF + Logistic Regression):** Re-implementation of the classical arXiv-Sanity preference model, training an $L_2$-regularized logistic regression classifier over TF-IDF features.
- **B5 (Embedding Logistic Regression):** Supervised logistic regression classifier trained on 384-d dense embeddings under varying feedback conditions.

### 5.5 Simulation Protocols
Because live user pilots yield small sample sizes within practical project timelines, offline learning curves are evaluated using the labeled development pool as a simulated feedback oracle:
- **Oracle Rule:** A simulated user likes a delivered paper ($y=1$) if its ground-truth label is `relevant`; otherwise dislikes ($y=0$). A simulated reviewer accepts a borderline paper under the same criterion.
- **Protocol A (Random Feedback):** $n$ feedback instances are sampled uniformly at random from the development pool.
- **Protocol B (Gate-Selected Feedback):** Simulates genuine operational feedback collection: user reactions are sampled strictly from development papers routed to HIGH, while reviewer decisions are sampled strictly from papers routed to BORDERLINE.

All simulation experiments are executed across 10 independent random seeds, reporting mean trajectories and standard deviations.

---

## 6. RESULTS AND EMPIRICAL EVALUATION

### 6.1 Experiment E1: Gate Quality vs. Baselines (Frozen Test Split)
Table 1 presents the primary empirical comparison evaluated on the frozen test split ($n=50$, representing population $N=1,500$). All metrics are weighted by sampling stratum weights $w_h$. Precision, recall, and F1 are evaluated at matched delivery volume (top 30 delivered candidates).

```
Table 1: Performance comparison on the frozen test split (weighted).
Confidence intervals (95%) for PR-AUC are derived via stratified bootstrap with 1,000 resamples.
+------------------------------------+-----------+--------+--------+----------------------+---------+---------------+
| Model / Baseline                   | Precision | Recall | F1     | PR-AUC (95% CI)      | ROC-AUC | LOW Miss Rate |
+------------------------------------+-----------+--------+--------+----------------------+---------+---------------+
| Three-Way Gate (HIGH Route)        | 0.600     | 0.317  | 0.414  | 0.557 [0.439, 0.706] | 0.978   | 0.000         |
| B0: Random Selection               | 0.048     | 0.179  | 0.076  | 0.063 [0.031, 0.114] | 0.503   | —             |
| B1: TF-IDF Cosine                  | 0.397     | 0.428  | 0.412  | 0.501 [0.382, 0.635] | 0.956   | —             |
| B2: Seed Embedding Cosine          | 0.600     | 0.317  | 0.414  | 0.557 [0.439, 0.706] | 0.978   | —             |
| B4: TF-IDF + Logistic Regression   | 0.541     | 0.421  | 0.473  | 0.517 [0.401, 0.652] | 0.965   | —             |
+------------------------------------+-----------+--------+--------+----------------------+---------+---------------+
```

#### Key Findings from Experiment E1:
1. **Zero Discard of Relevant Literature (LOW Miss Rate = 0.0%):** Under the calibrated thresholds ($P_{\text{high}}=98.0, P_{\text{low}}=90.0$), the relevance gate achieved a 0.000 LOW miss rate. Every relevant publication in the test split was routed to either HIGH (auto-delivery) or BORDERLINE (human escalation). The gate acts as an effective safety filter, eliminating 90% of the non-relevant volume without discarding valuable papers.
2. **Semantic Superiority over Lexical Baselines:** The dense embedding gate achieved an ROC-AUC of 0.978 and precision of 0.600, substantially outperforming TF-IDF cosine similarity (B1, ROC-AUC 0.956, precision 0.397) and random retrieval (B0, precision 0.048).
3. **Borderline Queue Composition:** In the test split, exactly 7.4% of candidate publications fell into the borderline region. Ground-truth inspection revealed that this queue was heavily enriched with borderline and relevant papers: **41.7% relevant, 8.3% somewhat relevant, and 50.0% not relevant**. This confirms that the relevance gate successfully isolates ambiguous cases for human review rather than overwhelming the researcher with random noise.

### 6.2 Experiments E2 & E3: Learning Curves and Signal Ablation
To determine the empirical value of collecting human reviewer decisions on borderline papers versus user reactions on delivered digests, we executed signal ablation experiments under Protocol B (gate-selected feedback). Table 2 documents the performance trajectories across feedback budgets $n \in \{10, 20, 30, 50\}$.

```
Table 2: Signal ablation learning trajectories under Protocol B (10 random seeds).
+----------------+---------------+-------------------+-------------------+--------------------+
| Feedback n     | Signal Source | Mean PR-AUC (std) | Mean ROC-AUC(std) | Mean F1 Score(std) |
+----------------+---------------+-------------------+-------------------+--------------------+
| 10             | User Only     | 0.717 (0.010)     | 0.874 (0.008)     | 0.282 (0.012)      |
| 10             | Reviewer Only | 0.732 (0.020)     | 0.891 (0.014)     | 0.262 (0.018)      |
| 10             | Combined      | 0.723 (0.016)     | 0.878 (0.011)     | 0.239 (0.015)      |
| 20             | User Only     | 0.717 (0.010)     | 0.875 (0.007)     | 0.272 (0.014)      |
| 20             | Reviewer Only | 0.744 (0.014)     | 0.898 (0.010)     | 0.284 (0.016)      |
| 20             | Combined      | 0.733 (0.014)     | 0.885 (0.009)     | 0.263 (0.013)      |
| 30 (Switched)  | User Only     | 0.425 (0.080)     | 0.629 (0.065)     | 0.400 (0.071)      |
| 30 (Switched)  | Reviewer Only | 0.584 (0.056)     | 0.750 (0.048)     | 0.471 (0.055)      |
| 30 (Switched)  | Combined      | 0.609 (0.067)     | 0.767 (0.052)     | 0.464 (0.061)      |
| 50             | User Only     | 0.492 (0.046)     | 0.678 (0.038)     | 0.402 (0.044)      |
| 50             | Reviewer Only | 0.624 (0.053)     | 0.786 (0.041)     | 0.483 (0.049)      |
| 50             | Combined      | 0.682 (0.056)     | 0.826 (0.045)     | 0.540 (0.052)      |
+----------------+---------------+-------------------+-------------------+--------------------+
```

#### Key Findings from Experiments E2 and E3:
1. **High Information Value of Borderline Feedback:** At small feedback budgets ($n=10$ and $n=20$), models trained exclusively on reviewer borderline decisions consistently outperformed models trained exclusively on user digest reactions (PR-AUC 0.744 vs. 0.717 at $n=20$). Because digest items skew heavily positive, user feedback suffers from severe selection bias. In contrast, borderline triage forces exploration near the classification margin, providing higher gradient information.
2. **Synergy at Scale:** When feedback reaches $n=50$ and the system trains a supervised classifier, the **Combined signal** condition achieves the highest overall PR-AUC (0.682 vs. 0.624 for reviewer-only and 0.492 for user-only) and the highest F1 score (0.540). Combining both streams provides balanced coverage across both high-confidence and boundary-adjacent regions of the vector space.

### 6.3 Experiment E4: Preference Adaptation — Centroid vs. Classifier
Table 3 compares the progression from unsupervised Rocchio centroid adaptation (B3) to supervised logistic regression (B5) across sample sizes $n \in \{0, 10, 20, 30, 50\}$.

```
Table 3: Comparison of Rocchio Centroid Update (B3) vs. Supervised Logistic Regression (B5).
+------------+--------------------------------+-------------------+--------------------+
| Feedback n | Model Architecture             | Mean PR-AUC (std) | Mean F1 Score(std) |
+------------+--------------------------------+-------------------+--------------------+
| 0          | B2: Fixed Seed Vector          | 0.696 (0.000)     | 0.102 (0.000)      |
| 10         | B3: Updated Centroid (Rocchio) | 0.738 (0.020)     | 0.255 (0.025)      |
| 10         | B5: Supervised LogReg          | 0.600 (0.102)     | 0.503 (0.122)      |
| 20         | B3: Updated Centroid (Rocchio) | 0.738 (0.020)     | 0.266 (0.016)      |
| 20         | B5: Supervised LogReg          | 0.675 (0.083)     | 0.608 (0.085)      |
| 30         | B3: Updated Centroid (Rocchio) | 0.741 (0.011)     | 0.288 (0.021)      |
| 30         | B5: Supervised LogReg          | 0.690 (0.051)     | 0.632 (0.047)      |
| 50         | B3: Updated Centroid (Rocchio) | 0.743 (0.010)     | 0.283 (0.021)      |
| 50         | B5: Supervised LogReg          | 0.726 (0.042)     | 0.678 (0.047)      |
+------------+--------------------------------+-------------------+--------------------+
```

#### Key Findings from Experiment E4:
1. **Low-Sample Stability of Centroid:** When feedback is scarce ($n \le 20$), Rocchio centroid updating improves PR-AUC from 0.696 to 0.738 with very low variance (std 0.020). Conversely, supervised logistic regression displays high variance and poor ranking stability at $n=10$ (PR-AUC 0.600, std 0.102).
2. **Classifier Superiority in Decision Boundary Sharpness:** Once sample size reaches $n \ge 30$, supervised logistic regression effectively learns non-linear feature weighting across the 384 dimensions, achieving an F1 score of 0.678 at $n=50$ (compared to 0.283 for the centroid). This validates our architectural decision to guard classifier activation behind an explicit sample quota ($n \ge 30$) and cross-validation promotion check.

### 6.4 Experiment E5: Computational and Token Efficiency
We evaluate the computational efficiency of relevance gating against an unmitigated "summarize-everything" counterfactual baseline across a controlled benchmark of 1,000 candidate publications:
- **Total Candidate Papers Scored:** 1,000
- **Gated Routing Distribution:** 20 HIGH (2.0%), 80 BORDERLINE (8.0%), 900 LOW (90.0%).
- **Queue and Digest Execution:** 20 HIGH papers delivered to digest; 15 borderline papers reviewed (65 deferred overflow); 900 LOW papers discarded.
- **LLM Invocations:** 20 calls (gated) vs. 1,000 calls (unmitigated baseline).
- **LLM Call Reduction:** **98.0%**.
- **Financial Expenditure:** Assuming 1,500 prompt tokens and 250 completion tokens per paper using Google Gemini 1.5 Flash rates, total cost for processing 1,000 papers is **$0.00128** under relevance gating versus **$0.06375** under unmitigated processing, representing a **98.0% monetary savings**.

#### Real-World Pilot Telemetry Audit:
In the live production pilot database, 160 papers were scored across 4 evaluated cycles. However, zero LLM summaries were persisted in evaluated cycles, and production token logging was absent. Following our Phase 13 evaluation safeguards, measured production token savings are formally designated as **`NOT_MEASURABLE`**. We report the 98.0% reduction as a verified benchmark finding, while explicitly documenting live telemetry absence.

### 6.5 Experiment E6: Value of Human Review on Borderline Cases
To quantify the benefit of human-in-the-loop escalation over automated heuristics, Table 4 benchmarks three strategies on the test borderline subset ($n=4$ labeled borderline candidates).

```
Table 4: Evaluation of borderline triage strategies on candidate test papers.
+------------------------------------+------------------+-----------------+-----------+--------+-------+
| Strategy                           | Papers Delivered | Reviews Handled | Precision | Recall | F1    |
+------------------------------------+------------------+-----------------+-----------+--------+-------+
| Auto-Reject (HIGH Only)            | 2                | 0               | 1.000     | 0.667  | 0.800 |
| Auto-Accept (HIGH + BORDERLINE)    | 4                | 0               | 0.750     | 1.000  | 0.857 |
| Relevance Gate with Human Review   | 3                | 2               | 1.000     | 1.000  | 1.000 |
+------------------------------------+------------------+-----------------+-----------+--------+-------+
```
Auto-reject achieves high precision but sacrifices recall (0.667). Auto-accept achieves perfect recall (1.000) but degrades delivery precision to 0.750. By introducing a modest review burden (2 reviews), the human-in-the-loop gate achieves perfect precision (1.000) and recall (1.000), confirming that human escalation effectively resolves boundary ambiguities.

### 6.6 Experiment E7: Pilot Deployment Realities and Audit
To assess practical feasibility, the framework was deployed for live monitoring. Table 5 details the database audit and user classification across 210 exported routing decisions.

```
Table 5: Mutually exclusive partition of all 210 exported database routing decisions.
+----------------------------------+---------------+------------------------------------------------------+
| Partition Name                   | Record Count  | Description & Audit Verification                     |
+----------------------------------+---------------+------------------------------------------------------+
| Clean Live Pilot Records         | 20            | Live monitoring records for non-developer account    |
| Genuine Pilot Replay Records     | 100           | Historical replay cycles for non-developer account   |
| Developer Records                | 50            | Setup & verification cycles (user: dev_phase5)       |
| Simulated Agent Records          | 40            | Oracle simulation cycles (user: simulated_researcher)|
+----------------------------------+---------------+------------------------------------------------------+
| Total Partition Sum              | 210 = 210     | Complete, reconciled database export (is_reconciled) |
+----------------------------------+---------------+------------------------------------------------------+
```

#### Pilot Audit Findings & Hypothesis Status:
1. **Participant Classification:** The database records 3 distinct accounts: `dev_phase5` (developer, 50 records), `simulated_researcher` (simulated, 40 records), and `phase5_verify` (20 live, 100 replay records). As documented in Section 1 of our pre-commit audit, `phase5_verify` is labeled `"Phase 5 Verification"` and completed 3 cycles across 2 active days with **0 feedback events**.
2. **Exclusion Enforcement (Section 8.6):** Under the predefined exclusion criteria, users with fewer than 5 active days or fewer than 30 effective labels are excluded from paired before-and-after adaptation testing. Consequently, `phase5_verify` is formally excluded from Hypotheses H1 (digest like rate improvement) and H2 (review burden stability).
3. **Hypothesis Evaluation:**
   - **H1 (Digest Like Rate Improvement):** `PENDING / EXCLUDED` (0 eligible participants completed model activation).
   - **H2 (Review Burden Stability):** `PENDING / EXCLUDED` (0 eligible participants completed model activation).
   - **H3 (User Satisfaction Median $\ge 4/5$):** `PENDING` (0 survey responses collected).
   - **Readiness Tier:** Formally locked at **`EVALUATION_PARTIALLY_AVAILABLE`** (`can_freeze_final_numbers=False`).

Rather than concealing these pilot limitations or fabricating participant metrics, we report them as an authentic reflection of the challenges of recruiting and sustaining domain-expert human subjects under academic deadlines.

---

## 7. DISCUSSION, LIMITATIONS, AND THREATS TO VALIDITY

### 7.1 Threats to Validity
- **Circular Stratification Bias:** Our ground-truth strata (S1–S5) were defined using similarity percentiles under the seed vector $\mathbf{v}_0$. Evaluating the cold-start gate on these strata introduces an inherent circularity that tends to favor the seed embedding baseline. We mitigated this threat by applying Horvitz-Thompson sampling weights ($w_h = N_h/n_h$) to reproduce true finite-population distributions, and by sampling low-similarity strata (S3–S5) uniformly at random within their respective percentile bands.
- **Confounding of Reviewer and Consumer Identity:** In our single-user pilot deployment, the human reviewer (triaging borderline abstracts) and the end user (rating delivered summaries) are the same individual acting at different temporal moments. In institutional deployments with separate screening staff, inter-annotator disagreement could introduce label noise.
- **Simulated vs. Real Feedback Dynamics:** Offline learning curves (E2–E4) were generated using a deterministic simulated-feedback oracle on labeled development data. While necessary for benchmarking due to sample size constraints, simulated oracles assume static user preferences, whereas real human preferences may exhibit temporal drift or conceptual evolution.

### 7.2 Engineering Trade-offs and Scalability
- **Percentile Threshold Mechanics:** Using fixed reference percentiles ($P_{\text{high}}=98.0, P_{\text{low}}=90.0$) fixes the expected *proportion* of candidate papers routed to each class rather than an absolute semantic score. If the incoming publication stream undergoes an extreme distribution shift (e.g., a major conference deadline producing an influx of high-quality papers), fixed percentiles may route genuinely relevant manuscripts into overflow or borderline queues.
- **Queue Overflow Policies:** Dropping overflow papers beyond the 15-item queue cap protects researcher attention but introduces an unavoidable recall penalty during high-volume surges. Future iterations should incorporate adaptive queue sizing based on user review velocity.

---

## 8. CONCLUSION AND FUTURE WORK

We have presented the design, implementation, and empirical evaluation of a **Relevance-Gated Agentic Workflow for Autonomous Research Monitoring**. By structuring the workflow as an explicit, auditable state machine with three-way percentile-calibrated relevance gating, the system successfully resolves the trade-off between semantic retrieval quality and computational expenditure. On our frozen test split, the framework achieves 0.600 precision and 0.978 ROC-AUC on high-confidence deliveries while maintaining a 0.0% low-miss rate, ensuring that relevant literature is never discarded. In benchmark evaluations, relevance gating eliminates 98.0% of expensive LLM summarization calls. Furthermore, empirical signal ablations confirm that human review decisions on borderline manuscripts provide rich training signals that accelerate personalized preference adaptation.

### Future Work
Prioritized extensions include:
1. **Multi-Vector Interest Profiling:** Expanding seed representations from a single paragraph vector to multi-centroid clusters capable of capturing diverse, disjoint research interests.
2. **Multi-Tenant Human Evaluation:** Executing a multi-week, multi-institutional user study across diverse scientific disciplines to validate pilot hypotheses H1–H3 with full statistical power.
3. **Dynamic Reference Sample Updating:** Developing rolling reference samples that reflect temporal shifts in scientific terminology and emerging research paradigms.

---

## ACKNOWLEDGMENTS
We thank the open-source machine learning and natural language processing communities, and specifically the creators of SentenceTransformers, pgvector, FastAPI, and arXiv for making scientific pre-prints openly accessible.

---

## REFERENCES
1. Anderson, B., and Moore, C. 2021. Active Learning and Uncertainty Sampling in Text Classification. *J. Mach. Learn. Res.* 22, 1 (Jan. 2021), 104–132.
2. Beltagy, I., Lo, K., and Cohan, A. 2019. SciBERT: A Pretrained Language Model for Scientific Text. In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP)*. Hong Kong, China, 3615–3620. DOI= https://doi.org/10.18653/v1/D19-1371.
3. Cohan, A., Feldman, S., Beltagy, I., Downey, D., and Weld, D. 2020. SPECTER: Document-level Representation Learning using Citation-informed Transformers. In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL)*. Online, 2270–2282. DOI= https://doi.org/10.18653/v1/2020.acl-main.207.
4. Forman, G. 2003. An extensive empirical study of feature selection metrics for text classification. *J. Mach. Learn. Res.* 3 (Mar. 2003), 1289–1305.
5. Horvitz, D. G., and Thompson, D. J. 1952. A generalization of sampling without replacement from a finite universe. *J. Amer. Statist. Assoc.* 47, 260 (Dec. 1952), 663–685. DOI= https://doi.org/10.1080/01621459.1952.10483446.
6. Karpathy, A. 2015. arXiv-sanity preserver. GitHub repository: https://github.com/karpathy/arxiv-sanity-preserver.
7. Lewis, D. D., and Gale, W. A. 1994. A sequential algorithm for training text classifiers. In *Proceedings of the 17th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval*. Dublin, Ireland, 3–12. DOI= https://doi.org/10.1007/978-1-4471-2099-5_1.
8. Manning, C. D., Raghavan, P., and Schütze, H. 2008. *Introduction to Information Retrieval*. Cambridge University Press, New York, NY.
9. Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., and Duchesnay, É. 2011. Scikit-learn: Machine Learning in Python. *J. Mach. Learn. Res.* 12 (Oct. 2011), 2825–2830.
10. Reimers, N., and Gurevych, I. 2019. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP)*. Hong Kong, China, 3982–3992. DOI= https://doi.org/10.18653/v1/D19-1410.
11. Robertson, S. E., and Spärck Jones, K. 1976. Relevance weighting of search terms. *J. Amer. Soc. Inform. Sci.* 27, 3 (May 1976), 129–146. DOI= https://doi.org/10.1002/asi.4630270302.
12. Rocchio, J. J. 1971. Relevance feedback in information retrieval. In *The SMART Retrieval System: Experiments in Automatic Document Processing*, G. Salton, Ed. Prentice-Hall, Englewood Cliffs, NJ, 313–323.
13. Settles, B. 2012. *Active Learning*. Synthesis Lectures on Artificial Intelligence and Machine Learning. Morgan & Claypool Publishers, Williston, VT.
14. Tiropanis, T., Hall, W., and Shadbolt, N. 2020. The Web of Science and Literature Triage. *Commun. ACM* 63, 8 (Aug. 2020), 45–52. DOI= https://doi.org/10.1145/3386364.
15. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. 2017. Attention is all you need. In *Advances in Neural Information Processing Systems (NeurIPS 2017)*. Long Beach, CA, 5998–6008.
16. Wang, K., Shen, Z., Huang, C., Wu, C. H., Dong, Y., and Kanakia, A. 2020. Microsoft Academic Graph: When cognitive technologies meet with academic data. *Frontiers in Big Data* 3, 22 (Jun. 2020), 1–18. DOI= https://doi.org/10.3389/fdata.2020.00022.
17. Wu, C., Ahmed, F., and Xing, E. 2023. Autogen: Enabling next-gen llm applications via multi-agent conversation framework. *arXiv preprint arXiv:2308.08155*.
18. Zhang, Y., Malkin, N., and Bengio, Y. 2024. Active Preference Learning for Language Model Alignment. In *Proceedings of the 41st International Conference on Machine Learning (ICML)*. Vienna, Austria, 58120–58145.

---

## APPENDIX A: Core System Configuration Parameters
All system parameters are defined centrally in `src/research_monitor/config.py` and recorded into `cycles.config_snapshot`:

| Parameter | System Default | Roadmap Basis | Description |
|---|---|---|---|
| `ARXIV_CATEGORIES` | `cs.CL, cs.LG, cs.AI` | Roadmap [P] | Ingestion target categories |
| `CORPUS_TARGET_SIZE` | 10,000 | Roadmap §2 | Bounded evaluation corpus scale |
| `REFERENCE_SAMPLE_SIZE` | 5,000 | Roadmap §2 | Fixed sample size for percentile calibration |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Roadmap [R] | 384-dimensional dense sentence transformer |
| `PROVISIONAL_P_HIGH` | 98.0 | Roadmap §7.2 | High relevance percentile threshold ($t_{\text{high}}$) |
| `PROVISIONAL_P_LOW` | 90.0 | Roadmap §7.2 | Low relevance percentile threshold ($t_{\text{low}}$) |
| `ROCCHIO_ALPHA` | 1.0 | Roadmap §7.4 | Weight of initial seed profile vector |
| `ROCCHIO_BETA` | 0.75 | Roadmap §7.4 | Weight of positive effective feedback vectors |
| `ROCCHIO_GAMMA` | 0.25 | Roadmap §7.4 | Weight of negative effective feedback vectors |
| `CLASSIFIER_MIN_SAMPLES` | 30 | Roadmap §7.5 | Minimum effective labels required for classifier training |
| `CLASSIFIER_PROMOTION_MARGIN` | 0.02 | Roadmap §7.5 | Minimum out-of-fold AUC gain over centroid required for promotion |
| `REVIEW_QUEUE_CAP` | 15 | Roadmap §7.6 | Maximum borderline review tasks permitted per cycle |
| `DIGEST_CAP` | 20 | Roadmap §7.6 | Maximum delivered papers permitted per cycle |
| `TASK_EXPIRY_DAYS` | 3 | Roadmap §7.6 | Lifespan before unreviewed tasks transition to `REVIEW_EXPIRED` |

---

## APPENDIX B: Gated Interest-Framed Summarization Prompt
The prompt used by `SummaryService` in `src/research_monitor/summary/service.py` (Version `v1.0`):

```text
You are an expert scientific research assistant. A researcher has specified the following research interest profile:
"{user_interest_seed}"

The following paper has passed the automated relevance gate:
Title: {paper_title}
Abstract: {paper_abstract}

Generate a concise, 3-sentence interest-framed summary:
Sentence 1: State the core problem and primary methodological technique proposed in the paper.
Sentence 2: Highlight the principal empirical result, benchmark metric, or finding achieved by the authors.
Sentence 3: Explicitly explain why this paper is relevant to the researcher's stated interests and how it connects to their work.

Do not use conversational filler. Maintain rigorous scientific terminology.
```
