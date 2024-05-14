# Import necessary libraries
import json  # For working with JSON files
from difflib import get_close_matches  # For finding close matches of strings
import numpy as np  # For numerical operations
import nltk  # Natural Language Toolkit
import string  # For string operations
import random  # For generating random numbers
from sklearn.feature_extraction.text import TfidfVectorizer  # For TF-IDF vectorization
from sklearn.metrics.pairwise import cosine_similarity  # For computing cosine similarity
import warnings  # For handling warnings
import os  # For interacting with the operating system

# Suppress user warnings
warnings.filterwarnings("ignore", category=UserWarning)


# Set NLTK data path to prevent download messages
nltk_data_path = os.path.join(os.path.dirname(__file__), 'nltk_data')
nltk.data.path.append(nltk_data_path)

# Load knowledge base for chat mode
def load_knowledge_base(file_path: str) -> dict:
    """
    Load knowledge base from a JSON file.

    Args:
        file_path (str): Path to the JSON file containing the knowledge base.

    Returns:
        dict: The loaded knowledge base.
    """
    with open(file_path, 'r') as file:
        data: dict = json.load(file)
    return data


# Save knowledge base for chat mode
def save_knowledge_base(file_path: str, data: dict):
    """
    Save knowledge base to a JSON file.

    Args:
        file_path (str): Path to the JSON file where the knowledge base will be saved.
        data (dict): The knowledge base to be saved.
    """
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=2)


# Find the best matching question for chat mode
def find_best_match(user_question: str, questions: list[str]) -> str | None:
    """
    Find the best matching question for the user's input.

    Args:
        user_question (str): The user's input question.
        questions (list): List of questions in the knowledge base.

    Returns:
        str | None: The best matching question from the knowledge base, or None if no match is found.
    """
    matches: list = get_close_matches(user_question, questions, n=1, cutoff=0.8)
    return matches[0] if matches else None

# Get answer for chat mode
def get_answer_for_question(question: str, knowledge_base: dict) -> str | None:
    """
    Retrieve the answer corresponding to a given question from the knowledge base.

    Args:
        question (str): The question for which the answer is requested.
        knowledge_base (dict): The knowledge base containing questions and their answers.

    Returns:
        str | None: The answer corresponding to the given question, or None if the question is not found in the knowledge base.
    """
    for q in knowledge_base.get("questions", []):
        if q["question"] == question:
            return q["answer"]
    return None


def quit_execution():
    """
    Function to quit the execution of the chatbot.
    """
    print("Exiting the chatbot. Goodbye!")
    quit()


# Main chat bot function
def chat_bot():
    """
    Main function to run the chatbot.

    The chatbot prompts the user to choose between 'chat mode' and 'translate mode',
    then executes the selected mode accordingly.

    """
    knowledge_base_file = 'knowledge_base.json'
    knowledge_base = load_knowledge_base(knowledge_base_file)

    while True:
        print("Hi! I am Docbot! Which mode do you want to use, 'chat mode' or 'translate mode'?")
        mode = input().strip().lower()

        if mode == 'chat mode':
            chat_mode(knowledge_base, knowledge_base_file)
        elif mode == 'translate mode':
            translate_mode()
        elif mode == 'quit':
            quit_execution()
        else:
            print("Invalid mode. Please choose 'chat mode', 'translate mode', or 'quit'.")


# Chat mode functionality
def chat_mode(knowledge_base, knowledge_base_file):
    """
    Function to handle the chat mode of the chatbot.

    Parameters:
        knowledge_base (dict): The knowledge base containing questions and answers.
        knowledge_base_file (str): The file path to save the knowledge base.

    """
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
    """
    Function to handle the translate mode of the chatbot.

    """
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


# Preprocess document for translate mode
def preprocess_document(file_path: str):
    """
    Preprocesses the document for translate mode.

    Args:
    - file_path (str): The path to the document file.

    Returns:
    - raw_doc (str): The raw document text.
    - sentence_tokens (list): List of sentence tokens.
    - word_tokens (list): List of word tokens.

    """
    # Open the document file
    f = open(file_path, errors='ignore')
    # Read the document text and convert to lowercase
    raw_doc = f.read().lower()
    # Tokenize the document text into sentences
    sentence_tokens = nltk.sent_tokenize(raw_doc)
    # Tokenize the document text into words
    word_tokens = nltk.word_tokenize(raw_doc)
    return raw_doc, sentence_tokens, word_tokens


# Initialize greeting responses for translate mode
greet_input = ("hello", "hi", "whassup", "how are you?")
greet_responses = (" hi", " hey", " hey there!", " there there!")

def greet(sentence):
    """
    Greets the user based on the input sentence.

    Args:
    - sentence (str): The input sentence.

    Returns:
    - str or None: A greeting response if a greeting word is found in the input sentence, otherwise None.
    """
    # Split the input sentence into words and check if any word is a greeting
    for word in sentence.split():
        if word.lower() in greet_input:
            return random.choice(greet_responses)  # Return a random greeting response
    return None  # If no greeting word is found, return None


def response(user_response, sentence_tokens):
    """
    Generates a response for the user input in translate mode.

    Args:
    - user_response (str): The user's input.
    - sentence_tokens (list of str): List of sentence tokens from the document.

    Returns:
    - str: The generated response for the user input.
    """
    robo1_response = ''
    # Initialize TF-IDF vectorizer
    Tfidfvec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    # Transform the sentence tokens into TF-IDF vectors
    tfidf = Tfidfvec.fit_transform(sentence_tokens)
    # Calculate cosine similarity between the user response and all other sentences
    vals = cosine_similarity(tfidf[-1], tfidf)
    # Get the index of the sentence with the highest cosine similarity (excluding the user response itself)
    idx = vals.argsort()[0][-2]
    # Flatten the similarity scores and sort them
    flat = vals.flatten()
    flat.sort()
    # Get the second highest similarity score
    req_tfidf = flat[-2]
    
    if req_tfidf == 0:
        # If there is no similarity score above 0, the bot cannot understand the user
        robo1_response = robo1_response + " I am sorry. Unable to understand you!"
    else:
        # Return the most similar sentence from the document
        robo1_response = robo1_response + sentence_tokens[idx]
        
    return robo1_response


# Initialize WordNet lemmatizer
lemmer = nltk.stem.WordNetLemmatizer()

# Define function to lemmatize tokens
def LemTokens(tokens):
    """
    Lemmatizes the given tokens.

    Args:
    - tokens (list of str): List of tokens to lemmatize.

    Returns:
    - list of str: Lemmatized tokens.
    """
    return [lemmer.lemmatize(token) for token in tokens]

# Define dictionary to remove punctuation
remove_punc_dict = dict((ord(punct), None) for punct in string.punctuation)

# Define function to normalize text using lemmatization
def LemNormalize(text):
    """
    Normalizes the given text using lemmatization.

    Args:
    - text (str): The text to normalize.

    Returns:
    - list of str: Normalized tokens after lemmatization.
    """
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punc_dict)))

if __name__ == '__main__':
    chat_bot()
