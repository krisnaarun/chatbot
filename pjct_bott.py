#CHATBOT USING NLTK AND CHATTERBOT

# importing libraries
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
import nltk
import string
from chatterbot.trainers import ListTrainer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import collections.abc
collections.Hashable=collections.abc.Hashable


import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.feature_extraction.text")

# Initialize NLTK downloads
nltk.download('punkt')
nltk.download('wordnet')

# Load the data and preprocess it
with open(r'D:\DL_Projects\chatbot\new.txt', 'r', errors='ignore') as file:
    raw_doc = file.read().lower()

# Tokenize the document
sent_tokens = nltk.sent_tokenize(raw_doc)
word_tokens = nltk.word_tokenize(raw_doc)

lemmer = nltk.stem.WordNetLemmatizer()


def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]


remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)


def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# Create a ChatBot instance
bot = ChatBot('mybot')

# Train the bot using the ChatterBotCorpusTrainer
trainer = ChatterBotCorpusTrainer(bot)

# Train the bot using ChatterBot's built-in English corpus
trainer.train('chatterbot.corpus.english','chatterbot.corpus.english.conversations')

trainer2=ListTrainer(bot)
trainer2.train(['what is your name','I am just a chatbot','hey','Hello!','how do I become a data scientist?',
                'To become a data scientist, you typically need to learn programming, statistics, and machine learning, and work on real-world projects.',
                'Can you explain machine learning?',
                'Machine learning is a subset of artificial intelligence that focuses on training algorithms to make predictions or decisions based on data.',
                'what is machine learning','Machine learning is a subset of artificial intelligence that focuses on training algorithms to make predictions or decisions based on data.',
                 'what programming languages are commonly used in data science','Common programming languages in data science include Python and R.',
                 'commonly used programming languages in data science','Common programming languages in data science include Python and R.',
                 'what is the difference between supervised and unsupervised learning','Supervised learning involves labeled data, while unsupervised learning deals with unlabeled data.',
                 'difference between supervised and unsupervised learning','Supervised learning involves labeled data, while unsupervised learning deals with unlabeled data.',
                 ' data preprocessing techniques','Data preprocessing involves cleaning, transforming, and organizing data to prepare it for analysis.',
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

# Main conversation loop
while True:
    user_input = input('You: ')
    if user_input.lower() == 'bye':
        print('Bot: Bye')
        break
    else:
        # Check the user's input 
        if user_input.lower() in ['hi', 'hello', 'hey','what is your name','how do I become a data scientist?','Can you explain machine learning?',
                                  'what is machine learning','What programming languages are commonly used in data science','commonly used programming languages in data science',
                                  'what is the difference between supervised and unsupervised learning','difference between supervised and unsupervised learning',
                                  ' data preprocessing techniques','what is a neural network','popular machine learning libraries in Python',
                                  'what is feature engineering','what is natural language processing','what is deep learning','what is clustering',
                                  'Can you recommend books for learning data science','what is the difference between classification and regression',
                                  'what are hyperparameters','data visualization libraries in Python']:
            bot_response = bot.get_response(user_input)
        elif user_input.lower() in ['what is data science', 'data science']:
            bot_response='Data Science is a combination of mathematics, statistics, machine learning, and computer science. Data Science is collecting, analyzing and interpreting data to gather insights into the data that can help decision-makers make informed decisions.'
        else:
            bot_response = response(user_input)
        print('Bot:', bot_response)