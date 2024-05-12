import json
from difflib import get_close_matches
import numpy as np
import nltk
import string
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import os

warnings.filterwarnings("ignore", category=UserWarning)

# Set NLTK data path to prevent download messages
nltk_data_path = os.path.join(os.path.dirname(__file__), 'nltk_data')
nltk.data.path.append(nltk_data_path)

# Load knowledge base for chat mode
def load_knowledge_base(file_path: str) -> dict:
    with open('knowledge_base.json') as file:
        data: dict = json.load(file)
    return data

# Save knowledge base for chat mode
def save_knowledge_base(file_path: str, data: dict):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=2)

# Find best match question for chat mode
def find_best_match(user_question: str, questions: list[str]) -> str | None:
    matches: list = get_close_matches(user_question, questions, n=1, cutoff=0.8)
    return matches[0] if matches else None

# Get answer for chat mode
def get_answer_for_question(question: str, knowledge_base: dict) -> str | None:
    for q in knowledge_base.get("questions", []):
        if q["question"] == question:
            return q["answer"]
    return None

# Main chat bot function
def chat_bot():
    knowledge_base_file = 'knowledge_base.json'
    knowledge_base = load_knowledge_base(knowledge_base_file)

    while True:
        print("Hi! I am Docbot! Which mode do you want to use, 'chat mode' or 'translate mode'?")
        mode = input().strip().lower()

        if mode == 'chat mode':
            chat_mode(knowledge_base, knowledge_base_file)
        elif mode == 'translate mode':
            translate_mode()
        else:
            print("Invalid mode. Please choose 'chat mode' or 'translate mode'.")

# Chat mode functionality
def chat_mode(knowledge_base, knowledge_base_file):
    while True:
        user_input = input('You: ')

        if user_input.lower() == 'quit':
            save_knowledge_base(knowledge_base_file, knowledge_base)
            break
        elif user_input.lower() == 'switch':
            break

        best_match = find_best_match(user_input, [q["question"] for q in knowledge_base.get("questions", [])])

        if best_match:
            answer = get_answer_for_question(best_match, knowledge_base)
            if answer:
                print(f'Bot: {answer}')
            else:
                print("Bot: I don't have an answer for that.")
        else:
            print("Bot: I don't know the answer. Can you teach me?")
            new_answer = input('Type the answer or "skip" to skip: ')

            if new_answer.lower() != 'skip':
                knowledge_base.setdefault("questions", []).append({"question": user_input, "answer": new_answer})
                save_knowledge_base(knowledge_base_file, knowledge_base)
                print('Bot: Thank you! I learned a new response!')

# Translate mode functionality
def translate_mode():
    raw_doc, sentence_tokens, word_tokens = preprocess_document(r'C:\Users\sweet\Downloads\selfbot\input.txt')
    flag = True
    print("Hello! I am Docbot. Start typing your text after greeting to talk to me. To end the conversation, type 'bye'.")

    while flag:
        user_response = input('You: ').strip().lower()

        if user_response == "bye":
            flag = False
            print("Bot: Goodbye!")
        elif user_response == "switch":
            break
        elif user_response == "thank you" or user_response == "thanks":
            flag = False
            print('Bot: You are welcome..')
        else:
            if greet(user_response) is not None:
                print('Bot:' + greet(user_response))
            else:
                sentence_tokens.append(user_response)
                word_tokens = word_tokens + nltk.word_tokenize(user_response)
                final_words = list(set(word_tokens))
                print('Bot:', end='')
                print(response(user_response, sentence_tokens))
                sentence_tokens.remove(user_response)

# Load document and preprocess for translate mode
def preprocess_document(file_path: str):
    f = open(file_path, errors='ignore')
    raw_doc = f.read().lower()
    sentence_tokens = nltk.sent_tokenize(raw_doc)
    word_tokens = nltk.word_tokenize(raw_doc)
    return raw_doc, sentence_tokens, word_tokens

# Initialize greeting responses for translate mode
greet_input = ("hello", "hi", "whassup", "how are you?")
greet_responses = (" hi", " hey", " hey there!", " there there!")

# Define greeting function for translate mode
def greet(sentence):
    for word in sentence.split():
        if word.lower() in greet_input:
            return random.choice(greet_responses)

# Define response function for translate mode
def response(user_response, sentence_tokens):
    robo1_response = ''
    Tfidfvec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = Tfidfvec.fit_transform(sentence_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if (req_tfidf == 0):
        robo1_response = robo1_response + " I am sorry. Unable to understand you!"
        return robo1_response
    else:
        robo1_response = robo1_response + sentence_tokens[idx]
        return robo1_response

# Define Lemmatization functions for translate mode
lemmer = nltk.stem.WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punc_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punc_dict)))

if __name__ == '__main__':
    chat_bot()
