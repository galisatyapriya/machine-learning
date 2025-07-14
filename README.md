# 🧠 MathBERT-Based Algebra Response Evaluation

This project demonstrates the use of **MathBERT embeddings** and **Machine Learning models** to automate the evaluation of algebraic responses in educational settings. It showcases efficient use of NLP in combination with classifiers like SVM, KNN, Decision Tree, and Random Forest to grade student answers as correct or incorrect.

---

## 🔬 Features

- ✅ Automated grading of open-ended algebra responses  
- ✅ Uses MathBERT for understanding mathematical semantics  
- ✅ Multiple ML Models: SVM, KNN, Decision Tree, Random Forest  
- ✅ Feature selection and dimensionality reduction  
- ✅ Visual comparison of model metrics (accuracy, precision, recall, F1-score)  
- ✅ Clean modular codebase with Jupyter integration  

---

## 🚀 How It Works

- **MathBERT**: Converts algebra answers into 385-dimensional semantic embeddings  
- **Preprocessing**: Includes null value removal, duplicate removal, and label separation  
- **Feature Selection**: Applies correlation-based reduction to optimize model performance  
- **ML Training**: Uses 4-fold cross-validation on each model  
- **Evaluation**: Generates bar plots for performance metrics  

---

## 📂 File Structure

| File/Folder                     | Description                                          |
|--------------------------------|------------------------------------------------------|
| `ML-Project(3rd_year_Cse).pptx`| Presentation slides summarizing project outcomes     |
| `README.md`                    | Project overview                                     |
| `REPORT_ML.pdf`                | Final project report with analysis and methodology   |
| `ml_final.ipynb`               | Core ML training and evaluation notebook             |
| `ml final plotting.ipynb`      | Notebook for generating visualizations and metrics   |

---

## 💻 Running the Project

### Requirements

- Python 3.x  
- `transformers`  
- `scikit-learn`  
- `pandas`  
- `numpy`  
- `matplotlib`, `seaborn`  

### Steps

```bash
# Step 1: Clone the repository
git clone https://github.com/yourusername/mathbert-algebra-evaluation.git
cd mathbert-algebra-evaluation

# Step 2: Install dependencies
pip install -r requirements.txt

# Step 3: Run the notebooks
jupyter notebook "ml_final.ipynb"
jupyter notebook "ml final plotting.ipynb"
```

---

## 📊 Sample Results

| Model        | Accuracy | Precision | Recall | F1-Score |
|--------------|----------|-----------|--------|----------|
| **SVM**      | 78.26%   | 0.77      | 0.78   | 0.74     |
| KNN          | 72.46%   | 0.72      | 0.72   | 0.72     |
| Decision Tree| 68.11%   | 0.69      | 0.68   | 0.68     |
| Random Forest| 77.77%   | 0.76      | 0.77   | 0.76     |

> 📌 **SVM** achieved the best consistency and accuracy across all metrics.

---

## 📘 Report

See `REPORT_ML.pdf` for full project documentation, methodology, and results.

Refer to `ML-Project(3rd_year_Cse).pptx` for presentation-ready project summary.
