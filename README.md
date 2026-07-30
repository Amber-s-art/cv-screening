# Beyond the Keyword Match: A Human-Centric AI Approach to Resume Screening

[![Canva Presentation](https://img.shields.io/badge/Canva-View_Presentation-8B4513?style=for-the-badge&logo=canva)](https://canva.link/rn4i541k6vbyec3)

Applicant Tracking Systems (ATS) historically rely on rigid, lexical keyword matching, frequently discarding highly qualified candidates who lack exact phrase alignments. This project resolves this structural inefficiency by engineering an automated, human-centric resume screening pipeline grounded in deep Natural Language Processing (NLP) and Learning-to-Rank (LTR) algorithms.

**Author:** Amber Agrawal  
**Institution:** Symbiosis Statistical Institute (MSc Applied Statistics 2025–2027)  
**Affiliation:** IDEAS Foundation, ISI Kolkata  
**Project Guide:** Dr. Dipasree Pal

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

## ⚙️ Architecture & Pipeline Flow

The pipeline was executed across four modular phases to preserve structural integrity and prevent training-serving skew. Below is the architectural flow of the data from raw text to ranked shortlist:

```mermaid
graph TD
    A[Raw Resumes & Job Descriptions] --> B(Linguistic Preprocessing via spaCy)
    B --> C{Vectorization Generation}
    
    C -->|TF-IDF| D[Sparse Lexical Space]
    C -->|SBERT| E[Dense Semantic Space]
    
    D --> F[Feature Concatenation]
    E --> F
    
    G[Recruiter Intuition] -->|Asymmetric Experience Penalty| F
    
    F --> H{GroupShuffleSplit}
    
    H -->|Zero-Shot Validation| I[Phase 1: Pointwise Regression]
    H -->|Zero-Shot Validation| J[Phase 2: Binary Classification]
    H -->|Zero-Shot Validation| K[Phase 3: Learning-to-Rank]
    
    K -->|LGBMRanker| L((Dynamic Ranked Shortlist))
