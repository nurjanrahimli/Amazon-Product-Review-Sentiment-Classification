# Amazon-Product-Review-Sentiment-Classification
Sentiment classification on Amazon product reviews using NLP and machine learning.
# Amazon Product Review Sentiment Classification 📦💬

This project analyzes and classifies customer reviews from Amazon to determine whether the sentiment behind each review is **positive** or **negative**. We used real product review data, performed Natural Language Processing (NLP), and trained machine learning models to make accurate predictions.

---

## 📊 Project Overview

- **Dataset:** Amazon product review dataset (Datafiniti)
- **Goal:** Predict review sentiment (Positive = 1 / Negative = 0)
- **Approach:** NLP preprocessing + TF-IDF + ML classification
- **Models Used:** 
  - Naive Bayes ✅
  - Logistic Regression ✅
  - Confusion Matrix & ROC-AUC for evaluation

---

## 🛠️ Key Steps

1. **Data Cleaning & NLP**
   - Lowercasing
   - Punctuation removal
   - Tokenization
   - Stopwords removal
   - Lemmatization (TextBlob)

2. **Feature Engineering**
   - TF-IDF Vectorization

3. **Modeling**
   - Train/Test split
   - Logistic Regression with class balancing
   - Naive Bayes for baseline comparison

4. **Evaluation**
   - Classification Report (Precision, Recall, F1-score)
   - ROC-AUC Score
   - Confusion Matrix visualization
   - Sample sentence predictions

---

## 📈 Results

- **Logistic Regression Accuracy:** ~92%
- **AUC Score:** ~0.91
- **Naive Bayes:** Lower performance due to class imbalance
- **Real sentence predictions:** Successfully predicts sentiment with high accuracy

---

## 🧠 Lessons Learned

- Logistic Regression outperforms Naive Bayes when text complexity increases
- Class balancing (`class_weight='balanced'`) is crucial for fair predictions
- TF-IDF is a powerful and simple method for vectorizing text
- Cleaning text is as important as modeling itself

---

## 📁 Files

| File | Description |
|------|-------------|
| `amazon_reviews.csv` | Raw dataset |
| `amazon_sentiment.ipynb` | Full notebook with EDA + NLP + Modeling |
| `README.md` | Project overview |

---

## 🤖 Example Prediction

```python
Sentence: "This product is amazing! Highly recommend it."
Prediction: Positive ✅
