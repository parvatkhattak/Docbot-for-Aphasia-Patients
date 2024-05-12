import streamlit as st
import json
from difflib import get_close_matches
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Check if NLTK resources are available, otherwise download them
nltk.download('punkt')
nltk.download('stopwords')

# Streamlit page configuration
st.set_page_config(page_title="Chatbot Interface", page_icon=":robot:")

# Session state to store chat history and teaching phase
if 'history' not in st.session_state:
    st.session_state.history = []
if 'teaching_phase' not in st.session_state:
    st.session_state.teaching_phase = False
if 'last_question' not in st.session_state:
    st.session_state.last_question = None

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

def LemNormalize(text):
    tokens = nltk.word_tokenize(text.lower())
    lemmatizer = nltk.stem.WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]

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
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    all_sentences = sentence_tokens + [user_response]
    tfidf = TfidfVec.fit_transform(all_sentences)
    vals = cosine_similarity(tfidf[-1], tfidf[:-1])
    idx = vals.argsort()[0][-1]
    return all_sentences[idx]

# GUI setup
st.title("Personalized Chatbot for Aphasia Patients")
mode = st.radio("Choose mode:", ('Chat Mode', 'Translate Mode'))

user_input = st.text_input("Type your question or response here:", key="user_input")

def handle_send():
    if user_input:
        st.session_state.history.append(f"You: {user_input}")
        if mode == 'Chat Mode':
            if st.session_state.teaching_phase:
                knowledge_base = load_knowledge_base('knowledge_base.json')
                knowledge_base.setdefault("questions", []).append({"question": st.session_state.last_question, "answer": user_input})
                save_knowledge_base('knowledge_base.json', knowledge_base)
                st.session_state.history.append(f"Bot: Thank you! I've learned the answer to: {st.session_state.last_question}")
                st.success('Thank you! I learned a new response!')
                st.session_state.teaching_phase = False
            else:
                knowledge_base = load_knowledge_base('knowledge_base.json')
                best_match = find_best_match(user_input, [q["question"] for q in knowledge_base.get("questions", [])])
                if best_match:
                    answer = get_answer_for_question(best_match, knowledge_base)
                    st.session_state.history.append(f"Bot: {answer}")
                    st.success(f'Bot: {answer}')
                else:
                    st.session_state.history.append("Bot: I don't know the answer. Can you teach me?")
                    st.error("Bot: I don't know the answer. Can you teach me?")
                    st.session_state.teaching_phase = True
                    st.session_state.last_question = user_input
        elif mode == 'Translate Mode':
            _, sentence_tokens = preprocess_document('input.txt')  # Adjust the path as necessary
            translated_response = response(user_input, sentence_tokens)
            st.session_state.history.append(f"Bot: Translated: {translated_response}")
            st.success(f'Bot: Translated: {translated_response}')
        st.session_state.user_input = ""

if st.button('Send', on_click=handle_send):
    pass

st.text_area("Chat History", value="\n".join(st.session_state.history), height=300, disabled=True)
