import streamlit as st  # Import Streamlit for creating web applications
import json  # Import JSON library to work with JSON files
from difflib import get_close_matches  # Import get_close_matches to find close matches in sequences
import nltk  # Import NLTK library for natural language processing tasks
from sklearn.feature_extraction.text import TfidfVectorizer  # Import TfidfVectorizer for text vectorization
from sklearn.metrics.pairwise import cosine_similarity  # Import cosine_similarity to calculate similarity between vectors

# Check if NLTK resources are available, otherwise download them
nltk.download('punkt')  # Download the Punkt tokenizer models
nltk.download('stopwords')  # Download the stopwords corpus
nltk.download('wordnet')  # Download the WordNet corpus

# Streamlit page configuration
st.set_page_config(page_title="Chatbot Interface", page_icon=":robot:")  # Set the title and icon for the Streamlit page

# Session state to store chat history and teaching phase
if 'history' not in st.session_state:  # Check if 'history' is not in the session state
    st.session_state.history = []  # Initialize 'history' as an empty list
if 'teaching_phase' not in st.session_state:  # Check if 'teaching_phase' is not in the session state
    st.session_state.teaching_phase = False  # Initialize 'teaching_phase' as False
if 'last_question' not in st.session_state:  # Check if 'last_question' is not in the session state
    st.session_state.last_question = None  # Initialize 'last_question' as None


def load_knowledge_base(file_path: str) -> dict:
    try:
        with open(file_path, 'r') as file:  # Open the specified file in read mode
            data = json.load(file)  # Load the JSON data from the file
        return data  # Return the loaded data
    except FileNotFoundError:
        st.error("Knowledge base file not found.")  # Display an error message if the file is not found
        return {}  # Return an empty dictionary if the file is not found

def save_knowledge_base(file_path: str, data: dict):
    with open(file_path, 'w') as file:  # Open the specified file in write mode
        json.dump(data, file, indent=2)  # Save the data to the file in JSON format with indentation

def find_best_match(user_question: str, questions: list[str]) -> str | None:
    matches = get_close_matches(user_question, questions, n=1, cutoff=0.8)  # Find close matches for the user question
    return matches[0] if matches else None  # Return the best match if found, otherwise return None

def get_answer_for_question(question: str, knowledge_base: dict) -> str | None:
    for q in knowledge_base.get("questions", []):  # Iterate over the questions in the knowledge base
        if q["question"] == question:  # If the question matches
            return q["answer"]  # Return the corresponding answer
    return None  # Return None if the question is not found in the knowledge base

def LemNormalize(text):
    tokens = nltk.word_tokenize(text.lower())  # Tokenize the text into words and convert to lowercase
    lemmatizer = nltk.stem.WordNetLemmatizer()  # Initialize the WordNet lemmatizer
    return [lemmatizer.lemmatize(token) for token in tokens]  # Lemmatize each token

def preprocess_document(file_path: str):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:  # Open the specified file in read mode with UTF-8 encoding
            raw_doc = file.read().lower()  # Read the file content and convert text to lowercase
        sentence_tokens = nltk.sent_tokenize(raw_doc)  # Tokenize the text into sentences
        return raw_doc, sentence_tokens  # Return the raw document and sentence tokens
    except FileNotFoundError:
        st.error("Document file not found: " + file_path)  # Display an error message if the file is not found
        return "", []  # Return empty values if the file is not found


def response(user_response, sentence_tokens):
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')  # Initialize TF-IDF Vectorizer
    all_sentences = sentence_tokens + [user_response]  # Combine existing sentences with user response
    tfidf = TfidfVec.fit_transform(all_sentences)  # Fit and transform TF-IDF on all sentences
    vals = cosine_similarity(tfidf[-1], tfidf[:-1])  # Calculate cosine similarity
    idx = vals.argsort()[0][-1]  # Get the index of the most similar sentence
    return all_sentences[idx]  # Return the most similar sentence

# GUI setup
st.title("Personalized Chatbot for Aphasia Patients")  # Set the title of the Streamlit page
mode = st.radio("Choose mode:", ('Chat Mode', 'Translate Mode'))  # Create a radio button for mode selection

user_input = st.text_input("Type your question or response here:", key="user_input")  # Create a text input field for user interaction

def handle_send():
    if user_input:
        st.session_state.history.append(f"You: {user_input}")  # Add user input to chat history
        if mode == 'Chat Mode':
            if st.session_state.teaching_phase:
                if user_input.lower() == "skip":
                    st.session_state.history.append("Bot: Skipping teaching phase.")  # Add message about skipping teaching phase to chat history
                    st.success('Skipping teaching phase.')  # Show success message for skipping teaching phase
                    st.session_state.teaching_phase = False  # End teaching phase
                else:
                    knowledge_base = load_knowledge_base('knowledge_base.json')  # Load knowledge base
                    knowledge_base.setdefault("questions", []).append({"question": st.session_state.last_question, "answer": user_input})  # Add new question-answer pair to knowledge base
                    save_knowledge_base('knowledge_base.json', knowledge_base)  # Save updated knowledge base
                    st.session_state.history.append(f"Bot: Thank you! I've learned the answer to: {st.session_state.last_question}")  # Add message about learning new response to chat history
                    st.success('Thank you! I learned a new response!')  # Show success message for learning new response
                    st.session_state.teaching_phase = False  # End teaching phase
            else:
                knowledge_base = load_knowledge_base('knowledge_base.json')  # Load knowledge base
                best_match = find_best_match(user_input, [q["question"] for q in knowledge_base.get("questions", [])])  # Find best match for user input in knowledge base
                if best_match:
                    answer = get_answer_for_question(best_match, knowledge_base)  # Get answer from knowledge base
                    st.session_state.history.append(f"Bot: {answer}")  # Add bot's response to chat history
                    st.success(f'Bot: {answer}')  # Show bot's response as success message
                else:
                    st.session_state.history.append("Bot: I don't know the answer. Can you teach me?")  # Add message asking user to teach bot to chat history
                    st.error("Bot: I don't know the answer. Can you teach me?")  # Show error message asking user to teach bot
                    st.session_state.teaching_phase = True  # Start teaching phase
                    st.session_state.last_question = user_input  # Store user's question for teaching
        elif mode == 'Translate Mode':
            _, sentence_tokens = preprocess_document('input.txt')  # Preprocess document for translation mode
            translated_response = response(user_input, sentence_tokens)  # Get translated response
            st.session_state.history.append(f"Bot: Translated: {translated_response}")  # Add translated response to chat history
            st.success(f'Bot: Translated: {translated_response}')  # Show translated response as success message
        st.session_state.user_input = ""  # Reset user input after processing

if st.button('Send', on_click=handle_send):  # Create a button to send user input
    pass  # Placeholder for button action

st.text_area("Chat History", value="\n".join(st.session_state.history), height=300, disabled=True)  # Display chat history in a text area
