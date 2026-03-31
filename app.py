import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))

ps = PorterStemmer()

def preprocess(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t.isalnum()]
    tokens = [t for t in tokens if t not in stopwords.words('english') and t not in string.punctuation]
    tokens = [ps.stem(t) for t in tokens]
    return " ".join(tokens)

st.title("📩 Spam Detector")
input_msg = st.text_area("Enter message here")

if st.button("Check"):
    processed = preprocess(input_msg)
    vector = tfidf.transform([processed])
    result = model.predict(vector)[0]

    if result == 1:
        st.error("🚨 SPAM!")
    else:
        st.success("✅ NOT SPAM")

