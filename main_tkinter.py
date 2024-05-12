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
import tkinter as tk
from tkinter import Text, Scrollbar, Frame, Entry, Button, FLAT, simpledialog, END

warnings.filterwarnings("ignore", category=UserWarning)

# Set NLTK data path
nltk.data.path.append(os.path.join(os.path.dirname(__file__), 'nltk_data'))

# Load and save knowledge base functions
def load_knowledge_base(file_path: str) -> dict:
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

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

# Load document and preprocess for translate mode
def preprocess_document(file_path: str):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            raw_doc = file.read().lower()
        sentence_tokens = nltk.sent_tokenize(raw_doc)
        word_tokens = nltk.word_tokenize(raw_doc)
        return raw_doc, sentence_tokens, word_tokens
    except FileNotFoundError:
        print("File not found:", file_path)
        return "", [], []

# Preprocessing the input.txt once when the bot starts
input_text_path = r'C:\Users\sweet\Downloads\selfbot\input.txt'
raw_doc, sentence_tokens, word_tokens = preprocess_document(input_text_path)

def response(user_response, sentence_tokens):
    lemmer = nltk.stem.WordNetLemmatizer()
    def LemTokens(tokens):
        return [lemmer.lemmatize(token) for token in tokens]
    def LemNormalize(text):
        return LemTokens(nltk.word_tokenize(text.lower().translate(dict((ord(punct), None) for punct in string.punctuation))))

    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    all_sentences = sentence_tokens + [user_response]
    tfidf = TfidfVec.fit_transform(all_sentences)
    vals = cosine_similarity(tfidf[-1], tfidf[:-1])
    idx = vals.argsort()[0][-1]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if req_tfidf == 0:
        return "I am sorry. Unable to understand you!"
    else:
        return all_sentences[idx]

# GUI setup
wind = tk.Tk()
wind.title("Chatbot")
wind.geometry('600x600')
wind.configure(bg='#000000')
current_mode = "chat"  # Initial mode is chat

chat_bg = Frame(height=420, width=580, bg='#f5f5f5')
chat_bg.place(x=10, y=80)
entry_bg = Frame(height=60, width=500, bg='white')
entry_bg.place(x=10, y=520)
sendbtn_bg = Frame(height=60, width=65, bg='white')
sendbtn_bg.place(x=525, y=520)

chat_window = Text(chat_bg, height=24, width=80, bg='#f5f5f5', state='disabled', wrap='word')
chat_window.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

scrollbar = Scrollbar(chat_bg, command=chat_window.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
chat_window.config(yscrollcommand=scrollbar.set)

def send_message():
    user_input = user_entry.get()
    if user_input:
        display_message(user_input + ' <You>', 'right')
        user_entry.delete(0, tk.END)
        process_input(user_input)

def switch_mode():
    global current_mode
    if current_mode == "chat":
        current_mode = "translate"
        switch_button.config(text="Chat Mode")
    else:
        current_mode = "chat"
        switch_button.config(text="Translate Mode")

def process_input(user_input):
    if current_mode == "chat":
        chat_mode(user_input)
    elif current_mode == "translate":
        translate_mode(user_input)

def chat_mode(user_input):
    knowledge_base = load_knowledge_base('knowledge_base.json')
    best_match = find_best_match(user_input, [q["question"] for q in knowledge_base.get("questions", [])])
    if best_match:
        answer = get_answer_for_question(best_match, knowledge_base)
        display_message('Bot: ' + (answer if answer else "I don't have an answer for that."), 'left')
    else:
        display_message("Bot: I don't know the answer. Can you teach me?", 'left')
        teach_bot(user_input, knowledge_base)

def teach_bot(user_input, knowledge_base):
    new_answer = simpledialog.askstring("Teach Me", "Type the answer here:", parent=wind)
    if new_answer:
        knowledge_base.setdefault("questions", []).append({"question": user_input, "answer": new_answer})
        save_knowledge_base('knowledge_base.json', knowledge_base)
        display_message('Bot: Thank you! I learned a new response!', 'left')

def translate_mode(user_input):
    translated_response = response(user_input, sentence_tokens)
    display_message('Bot: Translated: ' + translated_response, 'left')

def display_message(message, side):
    chat_window.config(state='normal')
    chat_window.insert(END, message + '\n\n')
    chat_window.config(state='disabled')
    chat_window.see(END)

user_entry = Entry(entry_bg, width=32, bg='white', font=('Helvetica', 15), relief=FLAT, border=0)
user_entry.place(x=10, y=13)
send_button = Button(sendbtn_bg, text='➤', font=('Helvetica', 20), bg='#000000', fg='white', relief=FLAT, command=send_message)
send_button.place(x=5, y=4)
switch_button = Button(wind, text="Translate Mode", command=switch_mode)
switch_button.place(x=10, y=50)

wind.mainloop()
