# Beyond the Keyword Match: A Human-Centric AI Approach to Resume Screening

Applicant Tracking Systems (ATS) historically rely on rigid, lexical keyword matching, frequently discarding highly qualified candidates who lack exact phrase alignments. This project resolves this structural inefficiency by engineering an automated, human-centric resume screening pipeline grounded in deep Natural Language Processing (NLP) and Learning-to-Rank (LTR) algorithms.

**Author:** Amber Agrawal
**Institution:** Symbiosis Statistical Institute (MSc Applied Statistics)
**Affiliation:** IDEAS Foundation, ISI Kolkata

---

## 📊 Project Overview

Conventional recruitment infrastructures exhibit severe algorithmic vulnerabilities by relying almost exclusively on exact lexical overlap (TF-IDF/Boolean search). 

By decoupling candidate evaluation from rigid boolean queries, this project prioritizes semantic relevance. We evaluated asymmetric bi-encoder representations—specifically leveraging **Sentence-BERT (SBERT)** for dense semantic vectorization—against baseline statistical term frequencies. 

### Key Achievements:
* **Dense > Lexical:** SBERT embeddings boosted ROC-AUC separation from a baseline of `0.6187` to `0.8121`.
* **Zero-Shot Generalization:** Implemented strict `GroupShuffleSplit` logic based on job titles to mathematically prevent data leakage.
* **State-of-the-Art Sorting:** Shifted from pointwise regression to listwise ranking. The final `LGBMRanker` achieved a peak **NDCG@10 of 0.9398**, perfectly mirroring human shortlisting cognition.

---

## 🧰 Technology Stack

* **Data Wrangling:** `pandas`, `numpy`
* **Linguistic NLP:** `spaCy` (Tokenization, Lemmatization, Custom Stop-words)
* **Vectorization:** `scikit-learn` (TF-IDF), `sentence-transformers` (SBERT)
* **Machine Learning:** `HistGradientBoosting`, `LogisticRegression`
* **Learning-to-Rank (LTR):** `xgboost` (XGBRanker), `lightgbm` (LGBMRanker)

---

## 📂 Data Access

The empirical foundation of this study relies on an open-source recruitment dataset curated by Neuralframe AI. 
* **Raw Dataset:** [Kaggle - Resume Dataset](https://www.kaggle.com/datasets/saugataroyarghya/resume-dataset) (9,544 records, 35 attributes).

> *Note: Due to GitHub's file size limits on dense vector arrays, the final processed `.npy` embedding matrices and model `.pkl` files are hosted externally. Please refer to the Jupyter Notebooks to see the full transformation pipeline.*

---

## ⚙️ Architecture & Methodology

The pipeline was executed across four modular phases to preserve structural integrity and prevent training-serving skew:

1. **Data Sanitization:** Stripped unstructured resumes of generic corporate filler while explicitly sheltering critical technical acronyms.
2. **Feature Engineering:** Mathematically formulated a non-linear Experience Penalty ($\Delta E$) to model recruiter tenure bias.
3. **Group-Aware Isolation:** Utilized a strict `GroupShuffleSplit` to ensure models evaluated completely disjoint job requests.
4. **Tri-Framing Benchmarking:** 
    * *Phase 1:* Continuous Regression (Pointwise scoring)
    * *Phase 2:* Binary Classification (Threshold filtering)
    * *Phase 3:* Learning-to-Rank (Listwise optimization)

### Pipeline Flowchart
*(Refer to the project documentation for the complete architectural diagram)*

---

## 📈 Core Conclusions

1. **Semantic Understanding:** Transformer-generated embeddings eclipse n-gram statistical parsing, capturing latent structural skill equivalencies.
2. **Ranking Paradigm Superiority:** Predicting absolute scores fundamentally contradicts real-world HR triage logic. Listwise gradient-boosted ranking dynamically solves the core recruitment query.
3. **Validation Purity:** Isolating candidate pools via job groups is mathematically mandatory to ensure true zero-shot evaluation on future job requests.

---

## 🚀 Future Roadmap

* **Bayesian Probability Integration:** Reframe the tenure penalty determinism into a Bayesian inference model to extract posterior match distributions.
* **Counterfactual Bias Mitigation:** Investigate stratified counterfactual sampling to prevent algorithmic weighting of legacy corporate brand names.
* **Generative Justification Output:** Overlay a lightweight LLM strictly as a post-ranking justification layer for HR teams.
