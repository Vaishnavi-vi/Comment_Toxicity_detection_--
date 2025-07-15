# 🧠 Comment Toxicity Classification

This project focuses on detecting toxic comments using **Deep Learning** models, including **LSTM**, **CNN**, and **BERT**. It is designed to classify user comments into toxic or non-toxic categories, helping to filter harmful content on online platforms.

## 🚀 Project Overview

Toxicity in online discussions can have serious implications for mental health and user experience. This project builds multiple deep learning models to classify text comments based on toxicity using:

- 📌 LSTM (Long Short-Term Memory)
- 📌 CNN (Convolutional Neural Network)
- 📌 BERT (Bidirectional Encoder Representations from Transformers)

---

## 📁 Dataset

The dataset used is a labeled set of comments from online platforms,contains features like id,comment_text,toxic, severe_toxic,insult,obsence,threat,identity_hate.

**Columns include:**
- `id`: Unique ID for each comment
- `comment_text`: The actual text of the comment
Converted (toxic, severe_toxic,insult,obsence,threat,identity_hate) to single column label as toxic and non_toxic.
- `toxic_label`: Binary label indicating whether the comment is toxic

---

## 🔧 Project Structure
comment-toxicity/
│
├── data/
│ └── train.csv, test.csv
│
├── models/
│ ├── lstm_model
│ ├── cnn_model
│ ├── bert_model
│
├── tokenizer/
│ └── tokenizer.pkl (CNN)
│
├── app/
│ └── app.py (Streamlit)
│
├── utils/
│ └── preprocess.py
│ └── evaluate.py
│├── notebooks/
│ └── EDA.ipynb
│ └── LSTM_Model.ipynb
│ └── CNN_Model.ipynb
│ └── BERT_Model.ipynb
│
├── README.md


---

## 🛠️ Models Used

### 1. LSTM
- Tokenized and padded sequences using Word2Vec Word embedding
- Bidirectional LSTM.
- Dropout,Dense layers and callbacks to prevent overfitting.
- Trained with binary crossentropy and Adam optimizer.

### 2. CNN
- Embedding layer followed by 1D convolutions.
- Global max pooling for feature reduction.
- Dense layers for final classification.

### 3. BERT
- HuggingFace Transformers (`bert-base-uncased`)
- Fine-tuned on toxic comment classification task.
- Tokenization handled with `BertTokenizer`.
- Classification head added on top of BERT's pooled output.

---

## 📊 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- AUC-ROC

---
## Comparison

| Model    | Architecture                               | Accuracy | Precision | Recall   | F1 Score  | AUC-ROC  |   
| -------- | ------------------------------------------ | -------- | --------- | -------- | --------- | -------- | 
| **LSTM** | Embedding → BiLSTM → Dense                 | 0.95     | 0.86      | 0.68     | 0.76      | 0.96     |
| **CNN**  | Embedding → Conv1D → GlobalMaxPool → Dense | 0.95     | 0.84      | 0.72     | 0.77      | 0.97     |       
| **BERT** | `bert-base-uncased` + Classification Head  | 0.93     | 0.86      | 0.86     | 0.50      | 0.92     | 

---



