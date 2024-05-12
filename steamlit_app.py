
import streamlit as st
import json
from difflib import get_close_matches
import numpy as np
import nltk
import string
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Ensure NLTK resources are downloaded
nltk.download('punkt')

# Set up NLTK data path if necessary
nltk_data_path = os.path.join(os.path.dirname(__file__), 'nltk_data')
nltk.data.path.append(nltk_data_path)

def load_knowledge_base(file_path: str) -> dict:
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        st.error("Knowledge base file not found.")
        return {}

def save_knowledge_base(file_path: str, data: dict):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=2)

def find_best_match(user_question: str, questions: list[str]) -> str | None:
    matches = get_close_matches(user_question, questions, n=1, cutoff=0.8)
    return matches[0] if matches else None

def get_answer_for_question(question: str, knowledge_base: dict) -> str | None:
    for q in knowledge_base.get("questions", []):
        if q["question"] == question:
            return q["answer"]
    return None

def preprocess_document(file_path: str):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            raw_doc = file.read().lower()
        sentence_tokens = nltk.sent_tokenize(raw_doc)
        return raw_doc, sentence_tokens
    except FileNotFoundError:
        st.error("Document file not found: " + file_path)
        return "", []

def response(user_response, sentence_tokens):
    lemmer = nltk.stem.WordNetLemmatizer()
    def LemNormalize(text):
        tokens = nltk.word_tokenize(text.lower().translate(str.maketrans('', '', string.punctuation)))
        return [lemmer.lemmatize(token) for token in tokens]
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    all_sentences = sentence_tokens + [user_response]
    tfidf = TfidfVec.fit_transform(all_sentences)
    vals = cosine_similarity(tfidf[-1], tfidf[:-1])
    idx = vals.argsort()[0][-1]
    return all_sentences[idx]

# Initialize session state
if 'mode' not in st.session_state:
    st.session_state.mode = 'chat'

# GUI setup
st.title("Chatbot Interface")
mode_button = st.button("Switch to Translate Mode" if st.session_state.mode == 'chat' else "Switch to Chat Mode")

if mode_button:
    st.session_state.mode = 'translate' if st.session_state.mode == 'chat' else 'chat'

user_input = st.text_input("Type your question or response here:", key="user_input")

if user_input:
    knowledge_base = load_knowledge_base('knowledge_base.json')
    if st.session_state.mode == 'chat':
        best_match = find_best_match(user_input, [q["question"] for q in knowledge_base.get("questions", [])])
        if best_match:
            answer = get_answer_for_question(best_match, knowledge_base)
            st.write('Bot:', answer)
        else:
            st.write("Bot: I don't know the answer. Can you teach me?")
            new_answer = st.text_input("Please teach me the answer:", key="new_answer")
            if new_answer:
                knowledge_base.setdefault("questions", []).append({"question": user_input, "answer": new_answer})
                save_knowledge_base('knowledge_base.json', knowledge_base)
                st.success('Thank you! I learned a new response!')
    elif st.session_state.mode == 'translate':
        raw_doc, sentence_tokens = preprocess_document('input.txt')  # Adjust the path as necessary
        translated_response = response(user_input, sentence_tokens)
        st.write('Bot: Translated:', translated_response)


