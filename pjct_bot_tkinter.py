#CREATING A TKINTER INTERFACE

from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer, ListTrainer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import string
import tkinter as tk
from tkinter import Scrollbar, Text, Button, Entry
import collections.abc
collections.Hashable=collections.abc.Hashable

# Initialize NLTK downloads
nltk.download('punkt')
nltk.download('wordnet')

# Load the data and preprocess it
with open(r'D:\DL_Projects\chatbot\new.txt', 'r', errors='ignore') as file:
    raw_doc = file.read().lower()

# Tokenizing document
sent_tokens = nltk.sent_tokenize(raw_doc)
word_tokens = nltk.word_tokenize(raw_doc)

lemmer = nltk.stem.WordNetLemmatizer()


def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]


remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)


def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# Create a ChatBot instance
bot = ChatBot('mybot',
    storage_adapter={
        'import_path': 'chatterbot.storage.SQLStorageAdapter',
        'database_uri': 'sqlite:///mybot_database.db'
    })

# Train the bot using the ChatterBotCorpusTrainer
trainer = ChatterBotCorpusTrainer(bot)

# Train the bot using ChatterBot's built-in English corpus
trainer.train('chatterbot.corpus.english','chatterbot.corpus.english.conversations')

# Additional training using ListTrainer
trainer2 = ListTrainer(bot)
trainer2.train(['what is data science','Data Science is a combination of mathematics, statistics, machine learning, and computer science. Data Science is collecting, analyzing and interpreting data to gather insights into the data that can help decision-makers make informed decisions.',
                'what is your name','I am just a chatbot','who are you','I am just a chatbot','hey','Hello! How can I help you?' ,'how do I become a data scientist?',
                'To become a data scientist, you typically need to learn programming, statistics, and machine learning, and work on real-world projects.',
                'Can you explain machine learning?',
                'Machine learning is a subset of artificial intelligence that focuses on training algorithms to make predictions or decisions based on data.',
                'what is machine learning','Machine learning is a subset of artificial intelligence that focuses on training algorithms to make predictions or decisions based on data.',
                 'what programming languages are commonly used in data science','Common programming languages in data science include Python and R.',
                 'commonly used programming languages in data science','Common programming languages in data science include Python and R.',
                 'what is the difference between supervised and unsupervised learning','Supervised learning involves labeled data, while unsupervised learning deals with unlabeled data.',
                 'difference between supervised and unsupervised learning','Supervised learning involves labeled data, while unsupervised learning deals with unlabeled data.',
                 'data preprocessing techniques','Data preprocessing involves cleaning, transforming, and organizing data to prepare it for analysis.',
                 'what is a neural network','A neural network is a computational model inspired by the human brain, used for tasks like deep learning.',
                 'popular machine learning libraries in Python','Popular Python machine learning libraries include scikit-learn, TensorFlow, and PyTorch.',
                 'what is feature engineering','Feature engineering is the process of creating new, informative features from existing data to improve machine learning models.',
                 'what is natural language processing','NLP is a field that focuses on enabling computers to understand, interpret, and generate human language.',
                 'what is deep learning','Deep learning is a subset of machine learning that uses neural networks with many layers to model complex patterns in data.',
                 'what is clustering','Clustering groups similar data points together based on their features, such as K-means clustering.',
                 'can you recommend books for learning data science','Some popular data science books include "Introduction to Statistical Learning" and "Python for Data Analysis.',
                 'what is the difference between classification and regression','Classification predicts categories or labels, while regression predicts continuous numeric values.',
                 'what are hyperparameters','Hyperparameters are settings that are not learned from data but are set prior to training a model, such as learning rates.',
                 'data visualization libraries in Python','In addition to Matplotlib and Seaborn, Plotly and Bokeh are popular libraries for data visualization.'])

# Define response function using TF-IDF and cosine similarity
def response(user_response):
    bot_response = ''
    Tfidfvec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = Tfidfvec.fit_transform(sent_tokens + [user_response])
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if req_tfidf == 0:
        bot_response = "I am sorry! I don't understand you"
    else:
        bot_response = sent_tokens[idx]
    return bot_response

# Create a tkinter interface
def send_message():
    user_input = user_entry.get()
    if user_input.lower() == 'bye':
        chat_log.config(state=tk.NORMAL)
        chat_log.insert(tk.END, 'You: {}\n'.format(user_input), 'user')
        chat_log.insert(tk.END, 'Bot: Bye\n', 'bot')
        chat_log.config(state=tk.DISABLED)
        user_entry.delete(0, tk.END)
    else:
        chat_log.config(state=tk.NORMAL)
        chat_log.insert(tk.END, 'You: {}\n'.format(user_input), 'user')
        bot_response = bot.get_response(user_input)
        chat_log.insert(tk.END, 'Bot: {}\n'.format(bot_response), 'bot')
        chat_log.config(state=tk.DISABLED)
        user_entry.delete(0, tk.END)

# Create the tkinter window
root = tk.Tk()
root.title('Learn Data Science!!')
root.geometry('500x500')

# Create a chat log
chat_log = Text(root, bd=0, bg='white', height='8', width='60', font='Tahoma')
chat_log.config(state=tk.DISABLED)

# Configure tags for user and bot responses
chat_log.tag_configure('user', foreground='black')  
chat_log.tag_configure('bot', foreground='red')    


# Create a scrollbar for the chat log
scrollbar = Scrollbar(root, command=chat_log.yview, cursor='heart')
chat_log['yscrollcommand'] = scrollbar.set

# Create an entry field for user input
user_entry = Entry(root, bg='White', font=('Tahoma', 12))

# Create a "Send" button to send messages
send_button = Button(root, text='Send', bg='black', activebackground='green', fg='white', font=('Tahoma', 12), command=send_message)

# Place all the widgets on the tkinter window
scrollbar.place(x=376, y=6, height=386)
chat_log.place(x=6, y=6, height=386, width=400)
user_entry.place(x=6, y=401, height=40, width=265)
send_button.place(x=275, y=401, height=40, width=120)

root.mainloop()