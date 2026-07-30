# Beyond the Keyword Match: A Human-Centric AI Approach to Resume Screening

![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange?style=for-the-badge&logo=jupyter)
![License](https://img.shields.io/badge/License-CC_BY--NC_4.0-green?style=for-the-badge)

Applicant Tracking Systems (ATS) historically rely on rigid, lexical keyword matching, frequently discarding highly qualified candidates who lack exact phrase alignments. This project resolves this structural inefficiency by engineering an automated, human-centric resume screening pipeline grounded in deep Natural Language Processing (NLP) and Learning-to-Rank (LTR) algorithms.

**Author:** Amber Agrawal  
**Institution:** Symbiosis Statistical Institute (MSc Applied Statistics 2025–2027)  
**Affiliation:** IDEAS Foundation, ISI Kolkata  
**Project Guide:** Dr. Dipasree Pal  
**Internship Period:** 18th May 2026 – 31st July 2026

---

## 📊 Project Overview

Conventional recruitment infrastructures exhibit severe algorithmic vulnerabilities by relying exclusively on exact lexical overlap (TF-IDF/Boolean search). When a hiring manager searches for "Software Engineer" and a candidate lists "Backend Developer," legacy systems discard qualified talent due to vocabulary mismatch—not capability gap.

This project decouples candidate evaluation from rigid boolean queries by prioritising semantic relevance. We evaluated asymmetric bi-encoder representations—specifically leveraging **Sentence-BERT (SBERT)** for dense semantic vectorisation—against baseline statistical term frequencies across three distinct modelling paradigms: pointwise regression, binary classification, and listwise Learning-to-Rank.

### 🎯 Key Achievements

| Metric | Baseline (TF-IDF) | Our System (SBERT) | Improvement |
|--------|-------------------|-------------------|------------|
| ROC-AUC (Classification) | 0.6187 | **0.8121** | +31.3% |
| NDCG@10 (Ranking) | 0.9158 | **0.9398** | +2.6% |
| Semantic Signal (Feature Importance) | 5.2% | **52.4%** | +1006% |

---

## 🧰 Technology Stack

### Core Data Processing
- **Data Wrangling:** pandas 2.0+, NumPy 1.24+
- **Linguistic NLP:** spaCy 3.5 (tokenisation, lemmatisation, custom stop-words)

### Vectorisation & Embeddings
- **Statistical Vectors:** scikit-learn TfidfVectorizer (8,000 unigram + bigram features)
- **Neural Embeddings:** sentence-transformers with Nomic Embed Text v1.5 (768-dimensional dense vectors)

### Machine Learning & Ranking
- **Regression Baselines:** LinearRegression, Ridge, RandomForestRegressor, HistGradientBoostingRegressor
- **Classification Baselines:** LogisticRegression, RandomForestClassifier, SVC
- **Learning-to-Rank:** LightGBM (LGBMRanker), XGBoost (XGBRanker) with LambdaMART objectives

### Infrastructure
- **Version Control:** Git, GitHub
- **Notebooks:** Jupyter Lab 4.0+
- **Reproducibility:** Python virtual environments, requirements.txt pinning

---

## 🏗️ Architecture & Pipeline Flow

The pipeline executes across four modular phases to preserve structural integrity and prevent training-serving skew:

```
Raw Resumes & Job Descriptions
        ↓
Linguistic Preprocessing (spaCy)
  ├─ Lowercasing & tokenisation
  ├─ Remove URLs, emails, special characters
  ├─ Lemmatisation via morphological analysis
  └─ Custom stop-word filtering (52 HR-fluff phrases)
        ↓
Dual Vectorisation
  ├─ TF-IDF: Sparse 8,000-dimensional vectors
  └─ SBERT: Dense 768-dimensional embeddings
        ↓
Feature Engineering
  ├─ Experience gap (Exp_cand - Exp_req)
  ├─ Piecewise non-linear penalty function
  └─ Dynamic role-based context flags
        ↓
Group-Aware Train/Test Split (80/20)
  └─ GroupShuffleSplit by job ID (zero data leakage)
        ↓
Three-Phase Evaluation
  ├─ Phase 1: Pointwise Regression (RMSE, R²)
  ├─ Phase 2: Binary Classification (ROC-AUC, F1)
  └─ Phase 3: Listwise Ranking (NDCG@K, MRR, Spearman ρ)
        ↓
Final Ranked Shortlist
  └─ LGBMRanker + SBERT (NDCG@10 = 0.9398)
```

---

## 📂 Repository Structure

```
cv-screening-engine/
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
├── LICENSE                                # CC BY-NC 4.0
│
├── data/                                  # Engineered features & artefacts
│   ├── ml_ready_dataset.csv               # Cleaned tabular dataset (9,369 records)
│   ├── cv_dense.npy                       # SBERT dense vectors (resumes)
│   ├── jd_dense.npy                       # SBERT dense vectors (job descriptions)
│   ├── cv_tfidf.npz                       # Sparse TF-IDF matrices (resumes)
│   ├── jd_tfidf.npz                       # Sparse TF-IDF matrices (jobs)
│   ├── tfidf_vectorizer.pkl               # Fitted TF-IDF vectorizer
│   ├── numeric_scaler.pkl                 # Fitted StandardScaler
│   └── objective_fit_neutral_value.pkl    # Contextual objective fit artefact
│
├── notebooks/                             # Core experimental notebooks
│   ├── 01_data_preprocessing.ipynb        # Text cleaning, EDA, feature engineering
│   ├── 02_model_comparison.ipynb          # 3×3 regression benchmarking
│   ├── 03_phase1_regression.ipynb         # Pointwise regression (RMSE, R²)
│   ├── 04_phase2_classification.ipynb     # Binary classification (ROC-AUC, F1)
│   ├── 05_phase3_learning_to_rank.ipynb   # Listwise LTR (LGBMRanker vs XGBRanker)
│   └── utils.py                           # Shared utilities (NLP, metrics, penalties)
│
├── report/                                # Academic documentation
│   ├── internship_report_final.tex        # LaTeX source (production-ready)
│   ├── report.pdf                         # Compiled final report
│   ├── images/                            # Figures & visualisations
│   │   ├── word_cloud.png
│   │   ├── word_bar.png
│   │   ├── roc_curves.png
│   │   ├── confusion_matrix.png
│   │   ├── ndcg_comparison.png
│   │   └── feature_importance.png
│   └── references.bib                     # LaTeX bibliography
│
├── results/                               # Detailed experimental outputs
│   ├── regression_results.csv             # Phase 1 metrics
│   ├── classification_results.csv         # Phase 2 metrics
│   ├── ranking_results.csv                # Phase 3 NDCG/MRR/Spearman
│   └── feature_importance.csv             # LGBMRanker split-gain breakdown
│
└── presentation/
    └── cv_screening_presentation.pptx    # Final project presentation slides
```

---

## 🚀 Getting Started & Reproducibility

### Prerequisites

Ensure you have:
- Python 3.8 or higher
- pip or conda package manager
- 2GB+ free disk space (for embeddings and data artefacts)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/amberagrawal/cv-screening-engine.git
cd cv-screening-engine
```

2. **Create a virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

4. **Download the Kaggle dataset:**
```bash
kaggle datasets download -d saugataroyarghya/resume-dataset
unzip resume-dataset.zip -d data/
```

### Execution Order

To reproduce research results, execute notebooks sequentially:

1. **Data Preprocessing:**
   ```bash
   jupyter notebook notebooks/01_data_preprocessing.ipynb
   ```
   Generates `ml_ready_dataset.csv`, TF-IDF vectors, and SBERT embeddings.

2. **Regression Comparison:**
   ```bash
   jupyter notebook notebooks/02_model_comparison.ipynb
   ```
   Benchmarks 3 models × 3 features in a 3×3 grid (regression baseline).

3. **Phase 1 — Pointwise Regression:**
   ```bash
   jupyter notebook notebooks/03_phase1_regression.ipynb
   ```
   Evaluates continuous match score prediction (RMSE, R²).

4. **Phase 2 — Binary Classification:**
   ```bash
   jupyter notebook notebooks/04_phase2_classification.ipynb
   ```
   Evaluates threshold-based accept/reject screening (ROC-AUC, F1).

5. **Phase 3 — Learning-to-Rank:**
   ```bash
   jupyter notebook notebooks/05_phase3_learning_to_rank.ipynb
   ```
   Evaluates listwise ranking optimisation (NDCG@K, MRR, Spearman ρ).

---

## 📈 Key Findings & Conclusions

### 1. Semantic Understanding Outperforms Keyword Matching

Dense SBERT embeddings captured contextual relationships independent of exact terminology. On binary classification (threshold ≥ 0.60):
- **TF-IDF only:** ROC-AUC = 0.6187, F1 = 0.5568
- **SBERT only:** ROC-AUC = 0.8121, F1 = 0.7142
- **Improvement:** +31.3% AUC separation, +28.2% F1-Score

### 2. Learning-to-Rank is the Correct Mathematical Paradigm

Regression (absolute score prediction) failed due to inherent human subjectivity in ratings. Classification imposed artificial thresholds. LTR solved the actual HR problem: **relative candidate ranking**.

Final performance (LGBMRanker + SBERT):
- **NDCG@10:** 0.9398 (94% alignment with human judgements)
- **Spearman ρ:** 0.4367 (moderate-to-strong rank correlation)
- **MRR:** 0.8234 (first relevant candidate appears in top positions)

### 3. Group-Aware Validation Prevents Data Leakage

Naive random splits permitted models to memorise job-specific vocabulary patterns. Implementing GroupShuffleSplit (stratified by job ID) ensured test metrics measured true zero-shot generalisation to unseen hiring scenarios.

### 4. Feature Importance Breakdown (LGBMRanker)

| Component | Contribution | Interpretation |
|-----------|--------------|-----------------|
| Semantic Similarity (SBERT) | 52.4% | Contextual understanding dominates |
| Experience Penalty (ΔE) | 28.1% | Qualification filters under-matched candidates |
| Role Context Flags | 8.5% | Fine-grained adjustments for role type |
| TF-IDF Similarity | 5.2% | Keyword matching provides marginal signal |
| Location/Academic Flags | 5.8% | Domain-specific edge case handling |

---

## 🔮 Future Work & Recommendations

### Probabilistic Decision Making
Replace deterministic penalty functions with Bayesian inference. Output posterior probabilities $P(\text{Match} \mid \text{Skills}, \text{Experience})$ rather than point estimates, quantifying recruiter decision uncertainty.

### Fairness & Bias Mitigation
Implement stratified resampling during preprocessing to balance candidate pools across:
- Career lengths (junior → senior)
- Institution types (brand-name vs. regional universities)
- Demographic proxies (geography, language backgrounds)

Monitor post-deployment predictions for disparate impact across demographic groups.

### Explainability Layer
Integrate lightweight Language Model (LLM) as a post-ranking layer to generate natural-language justifications:
> "Candidate #1 ranked highest: 6+ years Python experience, demonstrated ML systems design in 3 production projects, and cloud infrastructure (AWS EC2, Lambda) expertise."

### Production Deployment
- Containerise preprocessing pipeline and trained models (Docker)
- Expose via FastAPI backend with async job queuing
- Develop Streamlit dashboard for HR professionals to upload batch PDFs and visualise interactive ranked leaderboards
- Integrate PDF-to-text parsing (pdfplumber, PyPDF)
- Add real-time monitoring and retraining triggers

### Cross-Domain Evaluation
Expand evaluation beyond technical/corporate roles to:
- Healthcare (nursing credentials, licensing requirements)
- Legal (bar admissions, case law expertise)
- Creative portfolios (design, writing samples)

Validate SBERT transferability across domain-specific vocabularies.

---

## 📚 Academic References

**Core Papers:**
- Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. *arXiv preprint arXiv:1908.10084*.
- Ke, G., Meng, Q., Finley, T., et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems*, 30, 3146–3154.
- Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.

**Dataset:**
- Roy Arghya, S. (2024). Resume Dataset. Kaggle. Retrieved from https://www.kaggle.com/datasets/saugataroyarghya/resume-dataset

**Nomic AI Embeddings:**
- Nomic AI (2024). Nomic Embed Text v1.5: Resilient Long-Context Text Embeddings. Retrieved from https://nomicai.com

---

## 📄 License

This project is distributed under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** license.

You are free to:
- **Share:** Copy and redistribute the material
- **Adapt:** Remix, transform, and build upon the material

Under the following terms:
- **Attribution:** Credit must be given to the original authors
- **Non-Commercial:** The material cannot be used for commercial purposes

For commercial licensing inquiries, please contact the authors.

---

## 🤝 Contributing & Acknowledgements

This project was developed during a summer internship at the **IDEAS Foundation, ISI Kolkata** (May–July 2026).

Special thanks to:
- **Dr. Dipasree Pal** — Project guide and mentor
- **Symbiosis Statistical Institute** — Academic institution
- **IDEAS Foundation** — Research infrastructure and mentorship
- **Saugata Roy Arghya** — Kaggle dataset curation

### How to Contribute

Found a bug or have a suggestion? Please open an issue or submit a pull request:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/improvement`
3. Commit your changes: `git commit -m "Add improvement"`
4. Push to the branch: `git push origin feature/improvement`
5. Open a Pull Request

---

## 📧 Contact & Support

**Author:** Amber Agrawal  
**Email:** [amber.agrawal@symbiosis.ac.in](mailto:amber.agrawal@symbiosis.ac.in)  
**GitHub:** [@amberagrawal](https://github.com/amberagrawal)  

For questions about methodology, code, or reproducibility, please open an issue on GitHub.

---

## 📑 Citing This Work

If you use this research in your work, please cite:

```bibtex
@inproceedings{agrawal2026cv_screening,
  author={Agrawal, Amber},
  title={Beyond the Keyword Match: A Human-Centric AI Approach to Resume Screening and Candidate Ranking},
  year={2026},
  organization={IDEAS Foundation, ISI Kolkata},
  note={Summer Internship Project Report}
}
```

---

**Last Updated:** July 2026  
**Status:** ✅ Production-Ready | Fully Documented | Reproducible

