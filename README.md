# machine-learning
An automated system to evaluate algebra answers using MathBERT embeddings and machine learning models like SVM and Random Forest, enabling fast, consistent, and scalable grading of student responses.
📘 Automatic Evaluation of Algebra Responses using MathBERT and ML
🧠 Overview
Grading open-ended algebra responses manually is time-consuming and inconsistent. This project automates that process using MathBERT, a BERT-based language model fine-tuned for mathematical contexts, in combination with machine learning classifiers like SVM, Random Forest, KNN, and Decision Tree.

Our approach converts student answers into dense MathBERT embeddings and classifies them as correct or incorrect, enhancing grading efficiency, fairness, and scalability in educational settings.

🎯 Objectives
✅ Efficiency: Reduce manual grading workload.

✅ Consistency: Standardized and unbiased grading.

✅ Scalability: Evaluate large sets of student responses.

✅ NLP Integration: Leverage MathBERT for mathematical language.

✅ Educational Insights: Identify common errors and help improve teaching.

📊 Dataset Description
📄 Responses from ~50 students to 21 algebra questions.

📐 Transformed into 385-dimensional MathBERT embeddings.

📊 Final dataset: 1024 rows of embedded vectors + labels.

🛠️ Methodology
🔹 1. Data Preprocessing
Removed duplicates and null values.

Dropped output labels where necessary for training set generation.

🔹 2. Feature Engineering
Used MathBERT to encode student answers.

Selected relevant features via correlation techniques.

Reduced from 385 to 384 dimensions.

🔹 3. Model Training
Trained the following models:

🔸 Support Vector Machine (SVM)

🔸 Random Forest (RF)

🔸 k-Nearest Neighbors (KNN)

🔸 Decision Tree (DT)

Applied 4-fold cross-validation to prevent overfitting.

🔹 4. Evaluation Metrics
📌 Accuracy

📌 Precision

📌 Recall

📌 F1-Score

Visualizations using bar plots helped interpret model performance.

📈 Results Summary
Model	Accuracy (Train/Test)	F1-Score	Precision	Recall
SVM	76.54% / 78.26%	0.74	0.77	0.78
KNN	78.58% / 72.46%	0.72	0.72	0.72
Decision Tree	93.95% / 68.11%	0.68	0.69	0.68
Random Forest	99.03% / 77.77%	0.76	0.76	0.77

🏆 SVM was found to be the most consistent and accurate model before and after feature selection.

🧠 About MathBERT
MathBERT is a transformer-based NLP model optimized for mathematical content, capturing the semantics and structure of equations and math language. Each student answer is converted into a dense 385-dimensional vector for downstream ML tasks.
