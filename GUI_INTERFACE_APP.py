import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Load trained model and vectorizer
model = pickle.load(open("nb_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf.pkl", "rb"))

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

st.title("Sentiment Analysis System")

user_input = st.text_area("Enter a review:")

if st.button("Predict Sentiment"):
    cleaned_text = clean_text(user_input)
    vectorized_text = vectorizer.transform([cleaned_text])
    prediction = model.predict(vectorized_text)

    if prediction[0] == 1:
        st.success("Positive Sentiment 😊")
    else:
        st.error("Negative Sentiment 😞")
