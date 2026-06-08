# 🪙 Optimized Hierarchical Intent Classification for Financial Service Queries

An optimized, production-ready Natural Language Processing (NLP) pipeline built entirely from scratch to categorize complex, fine-grained customer inquiries in the banking and finance sector. This system implements a custom linguistic pipeline and a mathematical feature execution engine from scratch, achieving high-accuracy tracking across dozens of distinct financial intents without relying on heavy deep learning transformers.

📋 Table of Contents

* [Overview](https://www.google.com/search?q=%23-overview)
* [Key Features](https://www.google.com/search?q=%23-key-features)
* [Dataset & Classes](https://www.google.com/search?q=%23-dataset--classes)
* [Project Architecture](https://www.google.com/search?q=%23-project-architecture)
* [Algorithmic Architecture](https://www.google.com/search?q=%23-algorithmic-architecture)
* [Mathematical Pipeline](https://www.google.com/search?q=%23-mathematical-pipeline)
* [Results & Metrics](https://www.google.com/search?q=%23-results--metrics)
* [Getting Started](https://www.google.com/search?q=%23-getting-started)
* [Project Structure](https://www.google.com/search?q=%23-project-structure)
* [Technologies Used](https://www.google.com/search?q=%23-technologies-used)
* [Contributing](https://www.google.com/search?q=%23-contributing)
* [License](https://www.google.com/search?q=%23-license)

---

## 🔍 Overview

In the financial technology industry, routing customer queries safely and accurately is vital to automation and risk assessment. Traditional black-box deep learning architectures require substantial computing resources and lack immediate mathematical auditability.

This project implements an end-to-end, lightweight machine learning pipeline that:

* Preprocesses noisy banking conversational queries using a deterministic, rule-based linguistic engine.
* Extracts compound text contextual features using engineered Unigrams and Bigrams.
* Computes dynamic feature importance scores across the corpus using a customized Inverse Document Frequency (IDF) matrix.
* Trains a continuous-feature Multinomial Naive Bayes classifier from scratch.
* Diagnoses model mistakes at scale using semantic multi-class confusion matrix chunking.

---

## ✨ Key Features

| Feature | Description |
| --- | --- |
| 🧠 **Engineered From Scratch** | Core modeling, text vectorization, and scoring pipelines implemented completely independent of heavy framework abstractions. |
| 🔀 **Compound N-Grams** | Generates overlapping word pairings (Bigrams) to catch conversational context (e.g., `"still_wait"`, `"card_arrive"`). |
| 🔤 **Custom Lemmatizer** | Maps irregular corporate and financial verbs/nouns (e.g., *spent → spend*, *withdrew → withdraw*, *declined → decline*) to clean base states. |
| 📈 **Sublinear Feature Scaling** | Uses logarithmic term frequency smoothing to suppress repetitive text noise and stabilize predictive scoring boundaries. |
| 📋 **Granular Analytics** | Slices the massive 77-class validation matrix into readable $10 \times 10$ heatmaps for distinct system debugging. |
| ⚡ **Highly Performant** | Executes training and multi-class batch evaluation in seconds on raw CPU frameworks. |

---

## 🦠 Dataset & Classes

The system processes the benchmark **Banking77** dataset (`mteb/banking77`) pulled live via Hugging Face. The data comprises **13,069 customer service queries** mapped across **77 fine-grained intents**.

### Example Intent Segments

* **Cards & Assets:** `card_activation`, `card_linking`, `lost_or_stolen_card`, `card_delivery_estimate`
* **Transactions & Fees:** `beneficiary_not_allowed`, `cash_withdrawal_charge`, `declined_transfer`, `wrong_amount_of_cash`
* **Account Configuration:** `activate_revolving_cards`, `edit_personal_details`, `verify_identity`, `passcode_forgotten`

---

## 🏗️ Project Architecture

```
┌─────────────────────────────────────────────────────┐
│             Hugging Face Banking77 Data             │
│            (Train: 9,993  |  Test: 3,076)           │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│          Advanced Custom Linguistic Engine          │
│    • Suffix Strip: -ies, -ing, -ed, -est, -ly       │
│    • Irregular Financial Verb Normalization          │
│    • Extract Feature Vectors (Unigrams + Bigrams)   │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│           Mathematical Vectorization Matrix         │
│    • Dynamic Multi-Corpus Document Frequency        │
│    • Continuous Sublinear TF-IDF Extraction         │
│    • Unique Vocabulary Boundary Registration        │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│         Continuous Multinomial Naive Bayes          │
│    • Continuous Prior Class Distribution Logging     │
│    • Laplacian Probabilistic Feature Smoothing      │
│    • Multi-Class Log-Likelihood Evaluation          │
└─────────────────────────────────────────────────────┘

```

---

## 🧠 Algorithmic Architecture

The system modifies standard count-based Multinomial Naive Bayes pipelines to handle fractional continuous feature importances directly:

```python
# Multi-class Log-Likelihood Evaluation Function
for label in class_docs:
    log_prior = math.log(class_docs[label] / total_docs)
    log_likelihood = 0
    
    for word, query_weight in query_tfidf.items():
        if word in vocabulary:
            # Modified Laplacian Smoothing with Continuous Weights
            numerator = word_weights[label][word] + 0.1
            denominator = class_totals[label] + (0.1 * vocab_size)
            
            # Weighted probability accumulation
            log_likelihood += query_weight * math.log(numerator / denominator)
            
    total_score = log_prior + log_likelihood

```

### Mathematical Pipeline Configuration

| Parameter | Configuration Strategy |
| --- | --- |
| **Tokenization Base** | Alphanumeric case-insensitive regex filtering |
| **Feature Depth** | Integrated Unigrams + Word-Level Bigrams |
| **Term Frequency (TF)** | Sublinear Scaling: <br>$$1 + \log(\text{count})$$

 |
| **Prior Probabilities** | Document Class Density Distribution Logs |
| **Smoothing Scale ($\alpha$)** | Balanced Factorization Weight ($0.1$) |

---

## 📊 Results & Metrics

* **Overall Classification Accuracy:** `82.96%` across all 77 target buckets.
* **Evaluation Speed:** Instant processing iteration runtime (< 20ms per validation query).

### Visualizations Rendered

* Complete Multi-Class Classification Reports (Precision, Recall, F1-Scores per label).
* Dynamic Subdivided $10 \times 10$ Confusion Matrix Segment Heatmaps utilizing Seaborn styles.

---

## 🚀 Getting Started

### Prerequisites

* Python 3.8+
* Active Terminal Access (CMD, PowerShell, or Bash)

### Installation

```bash
# 1. Clone the repository into your preferred drive (e.g., F:)
F:
git clone https://github.com/Adithya-J05/Optimized-Hierarchical-Intent-Classification-for-Financial-Service-Queries.git
cd Optimized-Hierarchical-Intent-Classification-for-Financial-Service-Queries

# 2. Install all core data science and analytical frameworks
pip install pandas pyarrow scikit-learn matplotlib seaborn datasets

```

### Quick Inference Pipeline execution

```python
# To test prediction pipelines manually in your environment:
from Model_Inference import predict_optimized

query_tokens = ["i", "am", "still", "waiting", "for", "my", "new", "card"]
predicted_intent = predict_optimized(query_tokens, class_docs, word_weights, class_totals, vocab_size, total_docs)

print(f"Target Intent Class Identified: {predicted_intent}")

```

---

## 📁 Project Structure

```
Optimized-Hierarchical-Intent-Classification-for-Financial-Service-Queries/
│
├── data/
│   ├── train-00000-of-00001.parquet    # Local mirror of tracking features
│   └── test-00000-of-00001.parquet     # Test validation split frames
│
├── core_pipeline/
│   ├── preprocess.py                   # Custom lemmatization & N-gram generation engine
│   ├── tfidf_weights.py                # IDF mapping logic implementation from scratch
│   ├── train_model.py                  # Core classifier matrix accumulation loop
│   └── diagnostic_plots.py             # Heatmap chunk division generation file
│
├── intent_classifier.ipynb             # End-to-end interactive development notebook
├── requirements.txt                    # System installation dependency list
└── README.md                           # Comprehensive project documentation

```

---

## 🛠️ Technologies Used

| Category | Technology |
| --- | --- |
| **Languages** | Python 3.x |
| **Data Engine Processing** | Pandas, PyArrow |
| **Dataset Host Platforms** | Hugging Face Datasets API |
| **Visual Diagnostics** | Seaborn, Matplotlib |
| **Statistical Calculations** | Python Standard Math Core |
| **Metrics Verification** | Scikit-Learn (Classification Metrics Evaluation) |

---

## 🤝 Contributing

1. Fork the repository
2. Open a structural feature branch (`git checkout -b feature/optimization-upgrade`)
3. Commit optimizations (`git commit -m 'Refactored log-likelihood loop for speed'`)
4. Push code changes back (`git push origin feature/optimization-upgrade`)
5. Launch a detailed **Pull Request**

---

## 📄 License

This project is open-source and registered under the **MIT License**.

---

*If you find this manual optimization approach helpful, feel free to drop a ⭐ on this repository!*
