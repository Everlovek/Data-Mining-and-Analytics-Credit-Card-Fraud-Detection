# Data-Mining-and-Analytics-Credit-Card-Fraud-Detection
Implementation of core data mining techniques applied to credit card fraud detection as part of Introduction to Data Mining. Demonstrates the complete data mining lifecycle including data quality assessment, SMOTE-based class balancing, supervised classification, unsupervised clustering, and association rule mining using Python and scikit-learn.

**Course:** IHC-Practical Approach to Data Mining and Analytics  
**Module:** Module 1: Introduction to Data Mining  
**Dataset:** Credit Card Fraud Detection (ULB Machine Learning Group, Kaggle)  
**Algorithms:** Decision Tree · Naive Bayes · K-Means · Apriori  

---

## Project Overview
Four data mining techniques are implemented:
- **Decision Tree Classifier** - primary supervised classification model (95.13% accuracy, 94% fraud recall)
- **Naive Bayes Classifier** - baseline supervised classifier for comparison (91.38% accuracy)
- **K-Means Clustering** - unsupervised clustering to assess natural separability of fraud vs legitimate transactions
- **Apriori Algorithm** -association rule mining on discretised transaction features
---

## Repository Structure

```
Credit-Card-Fraud-Detection-DMA/
│
├── DMA_Assignment1.ipynb          # Main Jupyter/Colab notebook (complete implementation)
├── README.md                      # This file
│
└── data/
    └── creditcard.csv             # Dataset (see Data Setup below)
```

---

## Dataset

**Source:** [Kaggle - ULB Machine Learning Group Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

| Attribute | Detail |
|---|---|
| Total Records | 284,807 |
| Features | 31 (V1–V28 via PCA, Time, Amount, Class) |
| Legitimate Transactions | 284,315 (99.83%) |
| Fraudulent Transactions | 492 (0.17%) |
| Time Period | September 2013 (2 days) |

## Requirements

### Environment
- Python 3.8+
- Google Colab **or** local Jupyter Notebook

### Python Libraries

```
pandas
numpy
scikit-learn
imbalanced-learn
mlxtend
matplotlib
seaborn
sqlite3       # built-in
```
## Setup Instructions

### Option A - Google Colab (Recommended)

1. **Download the dataset** from Kaggle:
   - Go to: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
   - Click **Download** and save `creditcard.csv`

2. **Upload to Google Drive:**
   - Place the file at the following path in your Drive:
     ```
     My Drive/Colab Notebooks/DMA Assignment1/creditcard.csv
     ```
3. **Open the notebook in Colab:**
   - Upload `DMA_Assignment1.ipynb` to Google Colab, or open it directly from Drive

4. **Mount Google Drive**:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
5. **Install missing libraries** if prompted - run this in a Colab cell:
   ```python
   !pip install imbalanced-learn mlxtend
   ```
6. **Run all cells in order**
---

### Option B - Local Jupyter Notebook

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/Credit-Card-Fraud-Detection-DMA.git
   cd Credit-Card-Fraud-Detection-DMA
   ```
2. **Install dependencies:**
   ```bash
   pip install pandas numpy scikit-learn imbalanced-learn mlxtend matplotlib seaborn jupyter
   ```
3. **Download the dataset** from Kaggle and place `creditcard.csv` in the `data/` folder.
4. 
5. **Update the file path** :
   ```python
   FILE_PATH = 'data/creditcard.csv'   # Update this line
   ```
6. **Launch Jupyter and run the notebook:**
   ```bash
   jupyter notebook DMA_Assignment1.ipynb
   ```
7. **Run all cells in order** 
---

## Execution Order

The notebook is structured into sequential sections — run all cells top to bottom:

| Section | Description |
|---|---|
| Section 1 | Library Imports |
| Section 2 | Data Acquisition |
| Section 3 | Data Understanding |
| Section 4 | Data Quality Assessment |
| Section 5 | Database Integration (SQLite) |
| Section 6 | Feature Normalisation |
| Section 7 | Outlier Assessment |
| Section 8 | Exploratory Data Analysis |
| Section 9 | Class Balancing (SMOTE) & Data Splitting |
| Section 10 | Algorithm 1 — Decision Tree Classifier |
| Section 11 | Algorithm 2 — Naive Bayes Classifier |
| Section 12 | Algorithm 3 — K-Means Clustering |
| Section 13 | Algorithm 4 — Apriori Association Rules |
| Section 14 | Final Model Comparison |

> **Important:** Cells must be run in order. Later sections depend on variables defined in earlier ones. If you encounter a `NameError`, restart the kernel and run all cells again.
---

## Reproducibility

A global random seed is set at the start of the notebook to ensure consistent results across runs:
```python
RANDOM_STATE = 42
```
This seed is passed to all stochastic components: `train_test_split`, `SMOTE`, `KMeans`, and `DecisionTreeClassifier`.
---

## Key Results Summary

| Algorithm | Type | Key Metric | Score | Fraud Recall |
|---|---|---|---|---|
| Decision Tree | Classification | Accuracy | 95.13% | 94% |
| Naive Bayes | Classification | Accuracy | 91.38% | 85% |
| K-Means | Clustering | Silhouette Score | 0.1044 | N/A |
| Apriori (V14/V17) | Association Rules | Rules Found | 69 rules | N/A |
| Apriori (V11/V4) | Association Rules | Rules Found | 93 rules | N/A |

**Recommended model:** Decision Tree Classifier - highest accuracy and fraud recall with interpretable decision rules.

---

## Common Issues

**Drive not mounting (Colab)**  
Re-run Cell 2 and follow the authentication prompt in the pop-up window.

**File not found error**  
Verify `FILE_PATH` in Cell 6 matches the exact location of `creditcard.csv` in your Drive or local directory.

**Import errors**  
Run `!pip install imbalanced-learn mlxtend` in a new cell before the imports cell.

**NameError on later cells**  
The kernel state was lost = go to Runtime → Restart and Run All to re-execute from the beginning.

**SMOTE memory error (local)**  
SMOTE on 284,807 records requires ~4 GB RAM. Use Google Colab if your local machine has limited memory.

---

## Note

All code in this project is produced for IHC- Practical Approach to Data Mining and Analytics. The dataset is publicly available and legally obtained from Kaggle under its terms of use.


## References

- Dal Pozzolo, A. et al. (2015). Calibrating probability with undersampling for unbalanced classification. *IEEE SSCI*.
- Chawla, N.V. et al. (2002). SMOTE: Synthetic minority over-sampling technique. *JAIR*, 16, 321–357.
- Bhattacharyya, S. et al. (2011). Data mining for credit card fraud. *Decision Support Systems*, 50(3), 602–613.
## By Everlove Fortitude Kavayi
