# Beyond the Keyword Match: A Human-Centric AI Approach to Resume Screening

[![Canva Presentation](https://img.shields.io/badge/Canva-View_Presentation-8B4513?style=for-the-badge&logo=canva)](https://canva.link/rn4i541k6vbyec3)
![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange?style=for-the-badge&logo=jupyter)

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

The pipeline was executed across modular phases to preserve structural integrity and prevent training-serving skew. Below is the architectural flow of the data from raw text to ranked shortlist:

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

## 📂 Repository File Placement

The repository is structured to ensure complete reproducibility of the academic report and presentation.

cv-screening/
├── README.md                           # Project documentation
├── finnal internship presentation.mp4  # Video recording of the final project presentation
├── requirements.txt                    # Python dependencies
├── data/                               # Engineered features, embeddings, and pickled models
│   ├── ml_ready_dataset.csv            # Cleaned and processed tabular dataset ready for modeling
│   ├── cv_dense.npy                    # SBERT dense semantic vectors for Resumes
│   ├── jd_dense.npy                    # SBERT dense semantic vectors for Job Descriptions
│   ├── cv_tfidf.npz                    # Sparse TF-IDF matrices for Resumes
│   ├── jd_tfidf.npz                    # Sparse TF-IDF matrices for Job Descriptions
│   ├── tfidf_vectorizer.pkl            # Pickled TF-IDF vectorizer model
│   └── objective_fit_neutral_value.pkl # Pickled contextual objective fit model/data
├── notebook/                           # Core experimental notebooks & modular utilities
│   ├── 01_data_preprocessing.ipynb     # Text cleaning, EDA, and feature engineering
│   ├── 03_phase1_regression.ipynb      # Pointwise regression benchmarking
│   ├── 04_phase2_classification.ipynb  # Binary threshold classification
│   ├── 05_phase3_learning_to_rank.ipynb# Listwise LTR (LGBMRanker vs XGBRanker)
│   └── utils.py                        # Shared helper functions (NLP, metrics, etc.)
├── project report/                     # Academic documentation and LaTeX source
│   ├── images/                         # LaTeX document assets and figures
│   ├── main.tex                        # LaTeX source code for the comprehensive report
│   └── report.pdf                      # Final compiled project report
└── results/                            # Detailed outcome analysis and outputs
    ├── clasification prob.docx         # Classification results and confusion matrix analysis
    ├── regresion prob.docx             # Pointwise regression analysis outputs
    └── rank problem.docx               # LTR metrics (NDCG) and split-gain performance breakdowns

🛠️ Getting Started & Reproducibility
To replicate the environment and run the pipeline locally, follow these steps:

1. Prerequisites
Ensure you have Python 3.8+ installed. You will also need to download the English language model for spaCy.

2. Installation
Clone this repository and install the required dependencies:

Bash
git clone [https://github.com/Amber-s-art/cv-screening.git](https://github.com/Amber-s-art/cv-screening.git)
cd cv-screening
pip install -r requirements.txt
python -m spacy download en_core_web_md
3. Execution Order
To reproduce the research results, execute the Jupyter Notebooks in the notebook/ directory in the following sequential order:

Run 01_data_preprocessing.ipynb to generate the clean text corpus.

(If generating vectors locally) Run the vectorization scripts/cells to output the .npy and .npz files to the data/ folder.

Run 03_phase1_regression.ipynb, 04_phase2_classification.ipynb, and 05_phase3_learning_to_rank.ipynb to view the modeling outcomes.

📈 Core Conclusions
Semantic Understanding: Transformer-generated embeddings completely eclipse n-gram statistical parsing, capturing latent structural skill equivalencies that keyword matchers miss.

Ranking Paradigm Superiority: Predicting absolute scores fundamentally contradicts real-world HR triage logic. Listwise gradient-boosted ranking dynamically sorts candidates relative to the competition, solving the core recruitment query.

Validation Purity: Isolating candidate pools via job groups (GroupShuffleSplit) is mathematically mandatory to ensure true zero-shot evaluation on future, unseen job requests.

🚀 Future Roadmap
Bayesian Probability Integration: Reframe the tenure penalty determinism into a Bayesian inference model to extract posterior match distributions rather than rigid limits.

Counterfactual Bias Mitigation: Investigate stratified counterfactual sampling to prevent algorithmic weighting of legacy corporate brand names.

Full-Stack Web Deployment: Integrate robust PDF-to-text parsing and build an accessible web interface. This will allow HR teams to instantly batch-upload and rank resumes, while giving students a tool to score and optimize their CVs against target job descriptions.

🤝 Contact & Acknowledgements
This project was developed during a summer internship at the IDEAS Foundation, ISI Kolkata. Special thanks to Dr. Dipasree Pal for her guidance and mentorship throughout the research.

If you have any questions about the methodology or code, feel free to open an issue or reach out via GitHub.