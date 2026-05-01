import pandas as pd
import numpy as np
import re
import tensorflow as tf
import streamlit as st
import pickle

from keras.preprocessing.sequence import pad_sequences

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load model
model = tf.keras.models.load_model("lstm_model.h5")

# Constants
Max_len = 100

# Load dataset
df = pd.read_csv("train.csv")

# UI
st.title("Toxic Comment Classifier Dashboard")

# ---------------- DATA INSIGHTS ----------------
st.header("📊 Dataset Insights")
st.write("Shape of dataset:", df.shape)

st.write("Class distribution:")
st.bar_chart(df[["toxic","severe_toxic","obscene","threat","insult","identity_hate"]].sum())

# ---------------- PREPROCESSING ----------------
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', str(text))
    text = text.lower()
    text = word_tokenize(text)
    text = [lemmatizer.lemmatize(word) for word in text if word not in stop_words]
    return ' '.join(text)

# ---------------- PREDICTION ----------------
labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

def predict_labels(sentence):
    text = clean_text(sentence)
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=Max_len, padding='post')

    pred = model.predict(padded, verbose=0)[0]

    return [labels[i] for i in range(len(labels)) if pred[i] > 0.2]

# ---------------- SAMPLE TEST ----------------
st.header("🧪 Sample Predictions")

samples = [
    "You are stupid and disgusting",
    "i killed few people"
]

for words in samples:
    pred = predict_labels(words)
    st.write(f"**Text:** {words}")
    st.write(f"**Prediction:** {pred}")
    st.write("---")

# ---------------- USER INPUT ----------------
st.header("🔮 Try Your Own Text")

user_input = st.text_area("Enter a comment")

if st.button("Predict"):
    result = predict_labels(user_input)
    st.write("Prediction:", result)



st.header("      Upload CSV for Batch Prediction")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file is not None:
    df_test = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:")
    st.dataframe(df_test.head())

    if "comment_text" not in df_test.columns:
         st.error("CSV must contain 'comment_text' column")
    else:
            df_test["prediction"] = df_test["comment_text"].apply(predict_labels)
            st.write("Predictions:")
            st.dataframe(df_test.tail())
            csv = df_test.to_csv(index=False).encode('utf-8')

            st.download_button(
            label="Download Predictions",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv"
        )
            

from sklearn.metrics import classification_report
import pandas as pd

# ---------------- CLASSIFICATION REPORT ----------------
st.header("📈 Classification Report")

# Step 1 — Get true labels and predictions from train data
# (use a sample if full data is too large)
sample_df = df.sample(500, random_state=42)   # 500 rows for speed

# Step 2 — Predict on sample
def predict_multilabel(text):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=Max_len, padding='post')
    pred = model.predict(padded, verbose=0)[0]
    return (pred > 0.2).astype(int)

y_pred = np.array([predict_multilabel(t) for t in sample_df["comment_text"]])
y_true = sample_df[labels].values

# Step 3 — Generate report
report = classification_report(
    y_true,
    y_pred,
    target_names=labels,
    output_dict=True   # returns dict so we can display as table
)

# Step 4 — Display as dataframe
report_df = pd.DataFrame(report).transpose().round(2)
st.dataframe(report_df)