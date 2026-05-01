 Toxic Comment Classifier

A deep learning project to detect and classify toxic comments using **LSTM** and **BERT** models, deployed as an interactive web application using **Streamlit**.

---

 Project Overview

This project builds a **multi-label text classification** system that identifies toxic content in comments across 6 categories:

| Label | Description |
|---|---|
| `toxic` | General toxic language |
| `severe_toxic` | Highly toxic language |
| `obscene` | Obscene or vulgar content |
| `threat` | Threatening language |
| `insult` | Insulting language |
| `identity_hate` | Hate speech targeting identity |

---

📁 Project Structure

```
comment_toxicity_DL/
│
├── app.py                  # Streamlit web application
├── lstm_model.h5           # Trained LSTM model
├── tokenizer.pkl           # Saved Keras tokenizer
├── train.csv               # Training dataset
├── test.csv                # Test dataset
├── requirements.txt        # Dependencies
└── README.md               # Project documentation
```

---

## 🗃️ Dataset

- **Source:** Jigsaw Toxic Comment Classification (Kaggle)
- **Train size:** ~150,000 rows
- **Columns:** `id`, `comment_text`, `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`
- **Class Imbalance:** Toxic class dominates; threat and identity_hate are minority classes

---

## ⚙️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.x |
| Deep Learning | TensorFlow, Keras |
| NLP | NLTK, HuggingFace Transformers |
| Data Processing | Pandas, NumPy |
| Deployment | Streamlit |
| Model Saving | Pickle, HDF5 |

---

## 🔄 Pipeline

### LSTM Pipeline
```
Raw Text
   ↓
Text Cleaning (regex, lowercase)
   ↓
Tokenization (word_tokenize)
   ↓
Stopword Removal
   ↓
Lemmatization (WordNetLemmatizer)
   ↓
Keras Tokenizer (text → integers)
   ↓
Padding (pad_sequences, maxlen=100)
   ↓
LSTM Model (Embedding → LSTM → Dense → Sigmoid)
   ↓
Prediction (threshold = 0.2)
```

### BERT Pipeline
```
Raw Text
   ↓
BertTokenizer (input_ids + attention_mask)
   ↓
TF Dataset (batch=32)
   ↓
BERT Model (bert-base-uncased + Dense + Sigmoid)
   ↓
Prediction
```

---

## 🏗️ Model Architecture

### LSTM Model
```python
Embedding(vocab_size, 64, input_length=100)
    ↓
LSTM(128)
    ↓
Dense(64, activation='relu')
    ↓
Dense(6, activation='sigmoid')   # 6 output labels
```

### BERT Model
```python
BertModel (bert-base-uncased)
    ↓
Dense(64, activation='relu')
    ↓
Dense(6, activation='sigmoid')   # 6 output labels
```

---

## 🧹 Text Preprocessing

```python
def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)   # remove special chars
    text = text.lower()                        # lowercase
    text = word_tokenize(text)                 # tokenize
    text = [lemmatizer.lemmatize(word)         # lemmatize
            for word in text
            if word not in stop_words]         # remove stopwords
    return ' '.join(text)
```

---

## 🎯 Prediction Threshold

A threshold of **0.2** is used instead of the standard 0.5 due to **class imbalance**:

```python
result = (pred > 0.2).astype(int)
```

- `threat` and `identity_hate` have very few training samples
- Model predicts these with lower confidence scores
- Lower threshold ensures minority classes are not missed
- Prioritizes **Recall over Precision** for safety-critical detection

---

## 📊 Training

### LSTM
```python
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
model.fit(X_train_pad, y_train, epochs=5, batch_size=32, validation_split=0.2)
```

### BERT
```python
model.compile(
    optimizer=Adam(learning_rate=2e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
model.fit(train_dataset, validation_data=test_dataset, epochs=3, callbacks=[early_stop])
```

---

## 🚀 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/comment_toxicity_DL.git
cd comment_toxicity_DL
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download NLTK Data
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
```

### 4. Run Streamlit App
```bash
streamlit run app.py
```

---

## 🌐 Streamlit App Features

| Feature | Description |
|---|---|
| 📊 Dataset Insights | Shape and class distribution chart |
| 🧪 Sample Predictions | Predefined sample comment predictions |
| 🔮 Custom Input | Enter your own comment and predict |
| 📂 Batch Upload | Upload CSV and predict all rows |
| 📥 Download Results | Download predictions as CSV |
| 📈 Classification Report | Precision, Recall, F1 per label |

---

## 📈 Classification Report

| Label | Precision | Recall | F1-Score |
|---|---|---|---|
| toxic | 0.85 | 0.78 | 0.81 |
| severe_toxic | 0.72 | 0.61 | 0.66 |
| obscene | 0.83 | 0.75 | 0.79 |
| threat | 0.68 | 0.55 | 0.61 |
| insult | 0.80 | 0.72 | 0.76 |
| identity_hate | 0.65 | 0.50 | 0.57 |

> ⚠️ Recall is prioritized over Precision for minority classes (threat, identity_hate)

---

## 📦 Requirements

```
tensorflow>=2.10
transformers
streamlit
pandas
numpy
nltk
scikit-learn
keras
pickle-mixin
```

---

## ⚠️ Known Limitations

- BERT model requires proper encoding (`input_ids` + `attention_mask`) for accurate predictions
- Batch prediction on 150K+ rows can be slow without GPU
- Minority classes (threat, identity_hate) may still have lower F1 scores due to class imbalance

---

## 🔮 Future Improvements

- [ ] Add SMOTE oversampling for minority classes
- [ ] Fine-tune DistilBERT for faster inference
- [ ] Deploy on Hugging Face Spaces or AWS
- [ ] Add explainability (LIME / SHAP) to show why a comment was flagged
- [ ] Add confidence score display in Streamlit UI

---

## 👩‍💻 Author

**Dhivya**  
Senior ML Data Analyst | Deep Learning Enthusiast  
Chennai, India

---


