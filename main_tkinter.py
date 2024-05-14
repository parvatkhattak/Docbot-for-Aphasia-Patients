import json  # Import the JSON module for working with JSON data
from difflib import get_close_matches  # Import a function to find close matches of strings
import numpy as np  # Import the NumPy library for numerical operations
import nltk  # Import the Natural Language Toolkit library for natural language processing tasks
import string  # Import the string module for string manipulation operations
import random  # Import the random module for generating random numbers and choices
from sklearn.feature_extraction.text import TfidfVectorizer  # Import TF-IDF vectorizer from scikit-learn
from sklearn.metrics.pairwise import cosine_similarity  # Import cosine similarity from scikit-learn
import warnings  # Import the warnings module to handle warnings
import os  # Import the os module for interacting with the operating system
import tkinter as tk  # Import the tkinter module for creating GUI applications
from tkinter import Label, Text, Scrollbar, Frame, Entry, Button, FLAT, simpledialog, END  # Import specific components from tkinter for building GUI

warnings.filterwarnings("ignore", category=UserWarning)  # Ignore UserWarning messages to suppress them during program execution

# Set NLTK data path to include a custom directory for NLTK resources
nltk.data.path.append(os.path.join(os.path.dirname(__file__), 'nltk_data'))


# Function to load the knowledge base from a JSON file
def load_knowledge_base(file_path: str) -> dict:
    # Open the JSON file in read mode
    with open(file_path, 'r') as file:
        # Load the JSON data into a dictionary
        data = json.load(file)
    return data

# Function to save the knowledge base to a JSON file
def save_knowledge_base(file_path: str, data: dict):
    # Open the JSON file in write mode
    with open(file_path, 'w') as file:
        # Write the data to the file with indentation for readability
        json.dump(data, file, indent=2)

# Function to find the best match for a user question from a list of questions
def find_best_match(user_question: str, questions: list[str]) -> str | None:
    # Use the get_close_matches function to find the best match
    matches = get_close_matches(user_question, questions, n=1, cutoff=0.8)
    # Return the best match if found, otherwise return None
    return matches[0] if matches else None

# Function to get the answer for a specific question from the knowledge base
def get_answer_for_question(question: str, knowledge_base: dict) -> str | None:
    # Iterate through the questions in the knowledge base
    for q in knowledge_base.get("questions", []):
        # Check if the question matches the given question
        if q["question"] == question:
            # Return the answer if found
            return q["answer"]
    # Return None if the answer is not found
    return None


# Function to preprocess a document for translate mode
def preprocess_document(file_path: str):
    try:
        # Open the document file in read mode with UTF-8 encoding
        with open(file_path, 'r', encoding='utf-8') as file:
            # Read the content of the file and convert it to lowercase
            raw_doc = file.read().lower()
        # Tokenize the raw document into sentences and words using NLTK
        sentence_tokens = nltk.sent_tokenize(raw_doc)
        word_tokens = nltk.word_tokenize(raw_doc)
        # Return the raw document, sentence tokens, and word tokens
        return raw_doc, sentence_tokens, word_tokens
    except FileNotFoundError:
        # Handle the case where the file is not found
        print("File not found:", file_path)
        return "", [], []


# Define the path to the input text file
input_text_path = r'C:\Users\sweet\Downloads\selfbot\input.txt'

# Preprocess the input text file once when the bot starts
raw_doc, sentence_tokens, word_tokens = preprocess_document(input_text_path)

# Function to generate a response based on user input and sentence tokens
def response(user_response, sentence_tokens):
    # Initialize the WordNet lemmatizer
    lemmer = nltk.stem.WordNetLemmatizer()
    
    # Define a function to lemmatize tokens
    def LemTokens(tokens):
        return [lemmer.lemmatize(token) for token in tokens]
    
    # Define a function to normalize text
    def LemNormalize(text):
        return LemTokens(nltk.word_tokenize(text.lower().translate(dict((ord(punct), None) for punct in string.punctuation))))
    
    # Create a TF-IDF vectorizer using lemmatization and stop words removal
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    
    # Combine user response with existing sentences for TF-IDF transformation
    all_sentences = sentence_tokens + [user_response]
    
    # Calculate TF-IDF vectors
    tfidf = TfidfVec.fit_transform(all_sentences)
    
    # Calculate cosine similarity between user response and all other sentences
    vals = cosine_similarity(tfidf[-1], tfidf[:-1])
    
    # Get the index of the most similar sentence
    idx = vals.argsort()[0][-1]
    
    # Flatten the array and sort it to get the most similar scores
    flat = vals.flatten()
    flat.sort()
    
    # Get the second last element as the last one is the user's response itself
    req_tfidf = flat[-2]
    
    # If there is no similarity score above 0, the bot does not understand the response
    if req_tfidf == 0:
        return "I am sorry. Unable to understand you!"
    else:
        # Return the most similar sentence from the document
        return all_sentences[idx]

# GUI setup using Tkinter
wind = tk.Tk()  # Create a Tkinter window
wind.title("Personalized Chatbot for Aphasia Patients")  # Set window title
wind.geometry('600x600')  # Set window dimensions
wind.configure(bg='#000000')  # Set background color

current_mode = "chat"  # Initial mode is chat

# Label for the chatbot title
hcb_text = Label(height=2, width=14, bg='#000000', text='Chatbot', font=('Impact', 20), fg='white')
hcb_text.place(x=200, y=5)

# Background frame for the chat history display
chat_bg = Frame(height=420, width=580, bg='#f5f5f5')
chat_bg.place(x=10, y=80)

# Background frame for user input entry
entry_bg = Frame(height=60, width=500, bg='white')
entry_bg.place(x=10, y=520)

# Background frame for the send button
sendbtn_bg = Frame(height=60, width=65, bg='white')
sendbtn_bg.place(x=525, y=520)

# Text widget for displaying chat history
chat_window = Text(chat_bg, height=24, width=80, bg='#f5f5f5', state='disabled', wrap='word')
chat_window.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Scrollbar for the chat window
scrollbar = Scrollbar(chat_bg, command=chat_window.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
chat_window.config(yscrollcommand=scrollbar.set)

# Function to send a user message
def send_message():
    user_input = user_entry.get()  # Get user input from the entry field
    if user_input:  # Check if the input is not empty
        # Display user message in the chat window
        display_message(user_input + ' <You>', 'right')
        user_entry.delete(0, tk.END)  # Clear the entry field
        process_input(user_input)  # Process the user input

# Function to switch between chat and translate modes
def switch_mode():
    global current_mode  # Access the global variable for mode
    if current_mode == "chat":
        current_mode = "translate"  # Change mode to translate
        switch_button.config(text="Chat Mode")  # Update the button text
    else:
        current_mode = "chat"  # Change mode to chat
        switch_button.config(text="Translate Mode")  # Update the button text


# Function to process user input based on the current mode
def process_input(user_input):
    if current_mode == "chat":  # If current mode is chat
        chat_mode(user_input)  # Process input in chat mode
    elif current_mode == "translate":  # If current mode is translate
        translate_mode(user_input)  # Process input in translate mode

# Function to handle chat mode based on user input
def chat_mode(user_input):
    # Load the knowledge base
    knowledge_base = load_knowledge_base('knowledge_base.json')
    
    # Find the best match for the user input in the knowledge base
    best_match = find_best_match(user_input, [q["question"] for q in knowledge_base.get("questions", [])])
    
    # If a match is found, retrieve the answer from the knowledge base and display it
    if best_match:
        answer = get_answer_for_question(best_match, knowledge_base)
        display_message('Bot: ' + (answer if answer else "I don't have an answer for that."), 'left')
    else:
        # If no match is found, prompt the user to teach the bot and call the teach_bot function
        display_message("Bot: I don't know the answer. Can you teach me?", 'left')
        teach_bot(user_input, knowledge_base)


# Function to teach the bot a new response based on user input
def teach_bot(user_input, knowledge_base):
    # Prompt the user to provide the answer
    new_answer = simpledialog.askstring("Teach Me", "Type the answer here:", parent=wind)
    # If the user provides an answer, update the knowledge base with the new question-answer pair
    if new_answer:
        knowledge_base.setdefault("questions", []).append({"question": user_input, "answer": new_answer})
        save_knowledge_base('knowledge_base.json', knowledge_base)
        # Display a message confirming that the bot has learned the new response
        display_message('Bot: Thank you! I learned a new response!', 'left')


# Function to handle the translate mode of the chatbot
def translate_mode(user_input):
    # Generate the translated response using the response function
    translated_response = response(user_input, sentence_tokens)
    # Display the translated response in the chat window
    display_message('Bot: Translated: ' + translated_response, 'left')

# Function to display messages in the chat window
def display_message(message, side):
    # Enable the chat window for editing
    chat_window.config(state='normal')
    # Insert the message into the chat window
    chat_window.insert(END, message + '\n\n')
    # Disable editing of the chat window
    chat_window.config(state='disabled')
    # Scroll to the end of the chat window
    chat_window.see(END)


# Create an entry widget for user input
user_entry = Entry(entry_bg, width=32, bg='white', font=('Helvetica', 15), relief=FLAT, border=0)
user_entry.place(x=10, y=13)

# Create a button to send user input
send_button = Button(sendbtn_bg, text='➤', font=('Helvetica', 20), bg='#000000', fg='white', relief=FLAT, command=send_message)
send_button.place(x=5, y=4)

# Create a button to switch between chat and translate mode
switch_button = Button(wind, text="Translate Mode", command=switch_mode)
switch_button.place(x=10, y=50)

# Function to quit the application
def quit_application():
    wind.destroy()  # Close the Tkinter window

# Create a quit button in the top right corner
quit_button = Button(wind, text="Quit", command=quit_application)
quit_button.place(x=550, y=48)  # Adjust the coordinates as needed

# Start the main event loop
wind.mainloop()

