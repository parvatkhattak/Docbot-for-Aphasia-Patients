import json  # Import JSON library to work with JSON files
import os  # Import OS library to interact with the operating system
import random  # Import random library to generate random numbers
import nltk  # Import NLTK library for natural language processing tasks
import string  # Import string library for string operations
from difflib import get_close_matches  # Import get_close_matches to find close matches in sequences
from sklearn.feature_extraction.text import TfidfVectorizer  # Import TfidfVectorizer for text vectorization
from sklearn.metrics.pairwise import cosine_similarity  # Import cosine_similarity to calculate similarity between vectors
import warnings  # Import warnings library to handle warnings
import speech_recognition as sr  # Import SpeechRecognition library for speech recognition tasks
import pyttsx3  # Import pyttsx3 library for text-to-speech conversion
from fuzzywuzzy import process  # Import process from fuzzywuzzy for string matching
from nltk.stem import WordNetLemmatizer  # Import WordNetLemmatizer for lemmatization
from nltk.corpus import stopwords  # Import stopwords from NLTK to remove common stopwords

# Suppress specific user warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Ensure you've downloaded the necessary NLTK resources
nltk.download('stopwords', quiet=True)  # Download the stopwords resource quietly (without output)
nltk.download('wordnet', quiet=True)  # Download the WordNet lemmatizer resource quietly
nltk.download('popular', quiet=True)  # Download popular NLTK resources quietly (includes punkt, wordnet, etc.)


# Set the path for the document to be used in translate mode
document_path = "input.txt"  # Specify the file path for the document to be processed in translate mode

# Initialize the WordNet lemmatizer
lemmer = WordNetLemmatizer()  # Create an instance of the WordNet lemmatizer for word lemmatization

# Define a function to lemmatize tokens
def LemTokens(tokens):
    # Lemmatize each token and filter out stopwords
    return [lemmer.lemmatize(token) for token in tokens if token not in set(stopwords.words('english'))]

# Define a function to normalize text
def LemNormalize(text):
    # Create a dictionary to remove punctuation by mapping punctuation characters to None
    remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
    # Convert text to lowercase, remove punctuation, tokenize the text, and then lemmatize the tokens
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


# Function to generate a response based on user input and sentence tokens
def response(user_response, sentence_tokens):
    # Create a TF-IDF Vectorizer object with a custom tokenizer and stop words removal
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    
    # Fit and transform the sentence tokens and the user's response into TF-IDF features
    tfidf = TfidfVec.fit_transform(sentence_tokens + [user_response])
    
    # Calculate the cosine similarity between the user's response and all other sentences
    vals = cosine_similarity(tfidf[-1], tfidf[:-1])
    
    # Get the index of the most similar sentence (excluding the last one, which is the user's response)
    idx = vals.argsort()[0][-1]
    
    # Flatten the similarity scores array and sort it
    flat = vals.flatten()
    flat.sort()
    
    # Get the second last element as the last one is the user's response itself
    req_tfidf = flat[-1]
    
    # If there is no similarity score above 0, the bot does not understand the response
    if req_tfidf == 0:
        return "I am sorry, I do not understand you."
    else:
        # Return the most similar sentence from the document
        return sentence_tokens[idx]


# Initialize speech recognition and text-to-speech
recognizer = sr.Recognizer()  # Create a Recognizer instance for speech recognition
tts_engine = pyttsx3.init()  # Initialize the text-to-speech engine

# Set NLTK data path to prevent download messages
nltk_data_path = os.path.join(os.path.dirname(__file__), 'nltk_data')  # Define the path to store NLTK data
nltk.data.path.append(nltk_data_path)  # Append the defined path to NLTK data paths to prevent download messages

# Define a function to speak text
def speak(text):
    print(f"Speaking: {text}")  # Print the text that will be spoken
    tts_engine.say(text)  # Use the text-to-speech engine to say the text
    tts_engine.runAndWait()  # Wait for the speech to finish

# Function to handle the chatbot functionality
def chat_bot():
    knowledge_base_file = 'knowledge_base.json'  # Path to the knowledge base file
    knowledge_base = load_knowledge_base(knowledge_base_file)  # Load the knowledge base from the file
    mode = choose_mode()  # Ask the user to choose the mode

    while mode:  # Continue running while a valid mode is selected
        if mode == 'chat mode':  # If the selected mode is 'chat mode'
            mode = chat_mode(knowledge_base, knowledge_base_file)  # Run chat mode functionality
        elif mode == 'translate mode':  # If the selected mode is 'translate mode'
            mode = translate_mode()  # Run translate mode functionality


# Function to preprocess the document for translate mode
def preprocess_document(file_path: str):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            raw_doc = file.read().lower()  # Read the file content and convert text to lowercase
        sentence_tokens = nltk.sent_tokenize(raw_doc)  # Tokenize the text into sentences
        word_tokens = nltk.word_tokenize(raw_doc)  # Tokenize the text into words
        return raw_doc, sentence_tokens, word_tokens  # Return the raw document, sentence tokens, and word tokens
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")  # Print an error message if the file is not found
        return "", [], []  # Return empty values if the file is not found


# Function to listen to the user's speech and convert it to text
def listen():
    print("Listening...")  # Indicate that the system is listening
    with sr.Microphone() as source:  # Use the microphone as the audio source
        recognizer.adjust_for_ambient_noise(source, duration=0.5)  # Adjust for ambient noise for 0.5 seconds
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)  # Listen for speech with a timeout of 5 seconds
            text = recognizer.recognize_google(audio)  # Use Google's speech recognition to convert audio to text
            print(f"Heard: {text}")  # Print the recognized text
            return text  # Return the recognized text
        except sr.WaitTimeoutError:
            print("Timeout during listening")  # Print a timeout message if listening times out
            return "listening timeout. please try again."  # Return a timeout error message
        except sr.UnknownValueError:
            print("Speech not understood")  # Print a message if speech is not understood
            return "i did not understand that."  # Return an error message for unrecognized speech
        except sr.RequestError:
            print("Service unavailable")  # Print a message if the speech recognition service is unavailable
            return "speech service is unavailable."  # Return a service unavailable error message


# Function to normalize user input
def normalize_input(input_string: str) -> str:
    normalized_string = input_string.lower().strip()  # Convert input to lowercase and strip leading/trailing whitespace
    choices = ['chat mode', 'translate mode']  # Define the valid mode choices
    best_match, confidence = process.extractOne(normalized_string, choices)  # Find the best match for the input among the choices
    if confidence > 85:  # If the confidence of the match is greater than 85%
        return best_match  # Return the best match
    return normalized_string  # Otherwise, return the normalized input as it is


# Function to choose the mode of operation for the chatbot
def choose_mode():
    speak("Hi! I am DocBot. Which mode do you want to use? 'Chat mode' or 'Translate mode'?")  # Ask the user to choose a mode
    while True:  # Loop until a valid mode is selected
        mode = listen()  # Listen for the user's mode selection
        normalized_mode = normalize_input(mode)  # Normalize the user input
        if normalized_mode in ['chat mode', 'translate mode']:  # If the input matches a valid mode
            return normalized_mode  # Return the selected mode
        elif mode in ["listening timeout. please try again.", "i did not understand that.", "speech service is unavailable."]:
            speak(mode)  # Speak the error message if there was an issue with listening or understanding
        else:
            speak("Invalid mode. Please choose 'chat mode' or 'translate mode'.")  # Prompt the user to choose a valid mode

# Function to load the knowledge base from a file
def load_knowledge_base(file_path: str) -> dict:
    with open(file_path, 'r') as file:  # Open the specified file in read mode
        data = json.load(file)  # Load the JSON data from the file
    return data  # Return the loaded data

# Function to save the knowledge base to a file
def save_knowledge_base(file_path: str, data: dict):
    with open(file_path, 'w') as file:  # Open the specified file in write mode
        json.dump(data, file, indent=2)  # Save the data to the file in JSON format with indentation

# Function to find the best match for a user question from the list of questions
def find_best_match(user_question: str, questions: list[str]) -> str | None:
    matches = get_close_matches(user_question, questions, n=1, cutoff=0.8)  # Find close matches for the user question
    return matches[0] if matches else None  # Return the best match if found, otherwise return None

# Function to get the answer for a specific question from the knowledge base
def get_answer_for_question(question: str, knowledge_base: dict) -> str | None:
    for q in knowledge_base.get("questions", []):  # Iterate over the questions in the knowledge base
        if q["question"] == question:  # If the question matches
            return q["answer"]  # Return the corresponding answer
    return None  # Return None if the question is not found in the knowledge base


# Function to handle chat mode
def chat_mode(knowledge_base, knowledge_base_file):
    speak("You can start chatting with me now. Say 'quit' to stop, or 'switch' to change mode.")  # Prompt the user to start chatting
    while True:  # Start an infinite loop to continuously handle user input
        user_input = listen()  # Listen for user input
        if user_input in ["quit", "listening timeout. please try again.", "i did not understand that.", "speech service is unavailable."]:
            if user_input == "quit":  # If the user wants to quit
                save_knowledge_base(knowledge_base_file, knowledge_base)  # Save the knowledge base
                break  # Exit the loop
            else:
                speak(user_input)  # Speak the error message received from listening
        elif user_input == 'switch':  # If the user wants to switch mode
            return choose_mode()  # Return to mode selection
        else:
            # Find the best match for the user input in the knowledge base
            best_match = find_best_match(user_input, [q["question"] for q in knowledge_base.get("questions", [])])
            if best_match:  # If a match is found
                answer = get_answer_for_question(best_match, knowledge_base)  # Get the answer for the matched question
                speak(answer if answer else "I don't have an answer for that.")  # Speak the answer or a fallback response
            else:  # If no match is found
                speak("I don't know the answer. Can you teach me?")  # Ask the user to teach a new response
                new_answer = listen()  # Listen for the new answer
                if new_answer.lower() == "skip":  # If the user wants to skip teaching
                    speak("Okay, let's move on.")  # Acknowledge skipping
                elif new_answer not in ["listening timeout. please try again.", "i did not understand that.", "speech service is unavailable."]:
                    # If a valid new answer is provided, update the knowledge base
                    knowledge_base.setdefault("questions", []).append({"question": user_input, "answer": new_answer})
                    save_knowledge_base(knowledge_base_file, knowledge_base)  # Save the updated knowledge base
                    speak('Thank you! I learned a new response!')  # Acknowledge learning a new response



# Function to generate a greeting response if the user input is a greeting
def greet(sentence):
    # Define common greetings as inputs
    greet_inputs = ("hello", "hi", "greetings", "what's up", "howdy", "hey")  # List of greeting words/phrases
    
    # Define the bot's responses to greetings
    greet_responses = ["Hi there!", "Hello!", "Hey!", "Hi, how can I help you today?"]  # Possible responses to greetings
    
    # Split the input sentence into words and check if any word is a greeting
    words = sentence.split()  # Split the sentence into individual words
    for word in words:
        if word.lower() in greet_inputs:  # Check if the word is in the list of greeting inputs
            return random.choice(greet_responses)  # Return a random greeting response if a match is found
    return None  # If no greeting word is found, return None


# Function to handle translate mode
def translate_mode():
    raw_doc, sentence_tokens, word_tokens = preprocess_document(document_path)  # Preprocess the document to get tokens
    speak("Translate mode activated. Please speak your sentence to start the conversation.")  # Inform the user that translate mode is active
    while True:  # Start an infinite loop to continuously handle user input
        user_response = listen()  # Listen for user input
        if user_response == "quit":  # If the user wants to quit
            break  # Exit the loop
        elif user_response == "switch":  # If the user wants to switch mode
            return choose_mode()  # Return to mode selection
        elif user_response in ["listening timeout. please try again.", "i did not understand that.", "speech service is unavailable."]:
            speak(user_response)  # Speak the error message if there was an issue with listening or understanding
        else:
            greeting = greet(user_response)  # Check if the user input is a greeting
            if greeting:  # If a greeting is detected
                speak(greeting)  # Respond with a greeting
            else:
                response_text = response(user_response, sentence_tokens)  # Generate a response based on the user input
                speak(response_text)  # Speak the response
                sentence_tokens.append(user_response)  # Append the user input to the sentence tokens after generating the response


if __name__ == '__main__':  # Check if the script is being run directly (not imported as a module)
    chat_bot()  # Call the chat_bot function to start the chatbot
