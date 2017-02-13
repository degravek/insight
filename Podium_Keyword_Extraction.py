from nltk.sentiment.vader import SentimentIntensityAnalyzer
from gensim.summarization import summarize, keywords
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models.word2vec import Word2Vec
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.manifold import TSNE
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as mp
from string import punctuation
import seaborn as sns
import pandas as pd
import numpy as np
import itertools
import random
import string
import nltk
import re

%matplotlib inline

# Read data from csv file
df = pd.read_csv('/Users/degravek/Insight/project/reviews10000.csv', header=0)
df.rename(columns={'Rating': 'rating', 'Review Text': 'text', 'Location Id': 'location',
                    'Publish Date': 'date', 'Industry': 'industry'}, inplace=True)

# For speed purposes, we can cut the dataframe down
df = df[:10000]

# Drop rows with missing values
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# Remove some punctuation when summarizing reviews
def process(text):
    result = text.replace('/', '').replace('\n', '')
    result = re.sub(r'\.+', '  ', result)
    result = re.sub(r'\!+', '  ', result)
    result = re.sub(r'(.)\1{2,}', r'\1', result)
    result = re.sub(r'\W+', ' ', result).strip()
    result = result + '.'
    return result

# Strip punctuation from the data
def strip_punctuation(text):
    result = ''.join(tmp for tmp in text if tmp not in punctuation)
    result = re.sub(' +',' ', result)
    result = result.lower().strip()
    return result

# Define the Porter stemmer in case we want to use it
porter = PorterStemmer()
def tokenizer_porter(text):
    result = [porter.stem(word) for word in text.split()]
    result = ' '.join(result)
    return result

# Define a function to remove stop words
stop = stopwords.words('english')
def rmstopwords(text):
    result = text.split()
    result = ' '.join(word for word in result if word not in stop)
    return result

# Define a function to lemmatize words
lem = WordNetLemmatizer()
def lemmatize(text):
    result = text.split()
    result = ' '.join(lem.lemmatize(word)for word in result if word not in stop)
    return result

# Define a function to break reviews into individual sentences
def tokenizetext(text):
    sentences = nltk.sent_tokenize(text)
    sentences = [[sent] for sent in sentences]
    return sentences

# Define a function to find n-grams quickly
# If looking for unigrams, make sure they're nouns
def ngrams(text, n):
    result = []
    text = text.split()
    if n==1:
        text = nltk.pos_tag(text)
        result = [word for word, pos in text if pos[0] == 'N']
    else:
        for i in range(len(text)-n+1):
            result.append('_'.join(text[i:i+n]))
    return result

# Define a function to extract noun-phrase chunks of text
# This chunking pattern looks for an optional series of
# adjectives followed by one or more nouns
def extract_candidate_chunks(text, grammar = 'CHUNK: {<JJ.*>*<NN.*>+}'):
    import itertools, nltk, string
    parser = nltk.RegexpParser(grammar)
    tagged_sents = [nltk.pos_tag(nltk.word_tokenize(text))]

    for chunk in tagged_sents:
        if not chunk:
            candidates = []
        else:
            candidates = []
            tree = parser.parse(chunk)
            for subtree in tree.subtrees():
                if subtree.label() == 'CHUNK':
                    candidates.append((' '.join([a for (a,b) in subtree.leaves()])))
    candidates = [word for word in candidates if word not in stop]
    return candidates

# Define a function to sort the aspects
# by weighted sum of sentiment
def SortMatrix(in_matrix, rfilter=None):
    if rfilter:
        in_matrix = in_matrix[in_matrix['rating'].isin(rfilter)].copy()

    in_matrix['counts'] = in_matrix.groupby(['aspects'])['sentiment'].transform('count')
    group1 = in_matrix.groupby(['aspects'])['sentiment'].sum()
    group2 = in_matrix.groupby(['aspects'])['counts'].mean()
    group3 = in_matrix.groupby(['aspects'])['sentiment'].mean()
    sorted_mat = pd.DataFrame()
    #sorted_mat['sent_sum']   = np.round(group1, 2)
    sorted_mat['counts']     = group2
    sorted_mat['frac']       = np.round(100*(group2/group2.sum()), 2)
    sorted_mat['sent_mean']  = np.round(group3, 2)
    sorted_mat['importance'] = np.round(group1/(group2**0.1), 2)
    sorted_mat = sorted_mat.sort_values('importance', ascending=False)
    sorted_mat.reset_index(level=0, inplace=True)
    return sorted_mat

# Define function to summarize
# reviews about certain aspects
def SummarizeReviews(matrix, aspect_list, n_statements):
    # Try to summarize the aspects
    star_rating, summary = [], []
    for i, aspect in enumerate(aspect_list):
        rating = matrix.groupby('aspects')['sentiment'].mean().sort_values(ascending=False)[aspect]
        star_rating.append((rating - (-1))*(5 - 1)/(1 - (-1)) + 1)

        # Try to process the text a little bit
        corpus = pd.DataFrame()
        corpus['text'] = matrix[(matrix['aspects']==aspect)].sort_values('sentiment', ascending=False)['context']
        #corpus = corpus.head(num).append(corpus.tail(num))
        corpus = corpus.sample(n=n_statements)
        corpus = list(corpus['text'].apply(process))
        print('ASPECT: ', aspect)
        print('STAR: ', star_rating[i])
        print('SUMMARY: ', corpus)
        print('\n')
    return corpus

# Add the path to where RAKE was downloaded
rake_path = '/Users/degravek/Downloads/RAKE-tutorial-master/'
import sys
sys.path.insert(0, rake_path)
import rake, operator

# RAKE will look for key phrases with
# at least four characters, composed
# of at most 3 words, appearing in the
# text at least one time
rake_object = rake.Rake(rake_path + 'SmartStoplist.txt', 4, 3, 1)

def ProcessData(df, ptype):
    parse_type = ptype

    # Divide reviews into individual sentences
    sentences = df['text'].apply(tokenizetext)

    # Stick the sentences back into the dataframe
    df['sentlist'] = sentences
    d1, d2, d3 = [], [], []
    d4, d5, d6 = [], [], []

    # Initialize the sentiment vader analyzer
    sid = SentimentIntensityAnalyzer()

    # Loop over sentences and process them
    for i in range(0, df.shape[0]):
        sent_list = df['sentlist'][i]
        for sentence in sent_list:
            sent_raw = ''.join(sentence)
            sent_pro = strip_punctuation(sent_raw)
            sent_pro = rmstopwords(sent_pro)
            sent_pro = lemmatize(sent_pro)
            sentiment = sid.polarity_scores(sent_raw)['compound']
            if parse_type[0] == 'ngram':
                pos = ngrams(sent_pro, ptype[1])
            elif parse_type == 'chunk':
                pos = extract_candidate_chunks(sent_pro)
            elif parse_type == 'rake':
                pos = rake_object.run(sent_raw)
                pos = [word[0] for word in pos]
            for j in pos:
                d1.append(df['date'][i])
                d2.append(df['location'][i])
                d3.append(df['rating'][i])
                d4.append(j),
                d5.append(sentiment)
                d6.append(sent_raw)

    # Put everything in a dataframe
    matrix = pd.DataFrame()
    matrix['date']      = d1
    matrix['location']  = d2
    matrix['rating']    = d3
    matrix['aspects']   = d4
    matrix['sentiment'] = d5
    matrix['context']   = d6

    # Remove any entry where the sentence
    # was determined to be neutral
    matrix = matrix[(matrix['sentiment'] != 0)]
    return matrix

# Process the data using RAKE,
# n-grams, and noun-phrases
matrix_rake = ProcessData(df, 'rake')
matrix_chunk = ProcessData(df, 'chunk')
matrix_ngram1 = ProcessData(df, ('ngram',1))
matrix_ngram2 = ProcessData(df, ('ngram',2))
matrix_ngram3 = ProcessData(df, ('ngram',3))

# Sort by weighted sum of sentiment
sorted_rake = SortMatrix(matrix_rake)
sorted_chunk = SortMatrix(matrix_chunk)
sorted_ngram1 = SortMatrix(matrix_ngram1)
sorted_ngram2 = SortMatrix(matrix_ngram2)
sorted_ngram3 = SortMatrix(matrix_ngram3)

# Create and search for aspects
# Input two static aspects, and grab the top
# three dynamic ones from sorted_ngram1
static_aspects  = ['value', 'service']
dynamic_aspects = sorted_ngram1['aspects'][:3] # Choose three dynamic aspects

# Aspects above are unigrams, so use matrix_ngram1
# Sort them by their frequency in matrix_ngram1
count = Counter(matrix_ngram1['aspects'])
s = [(x, count[x]) for x in static_aspects]
s = sorted(s, key=lambda x: x[1], reverse=True)
d = [(x, count[x]) for x in dynamic_aspects]
d = sorted(d, key=lambda x: x[1], reverse=True)

seen = set()
seen_add = seen.add
merged = s + d
merged = [x for x in merged if not (x in seen or seen_add(x))]

# Get a final list of aspects
aspects = list([ii for ii in zip(*merged)][0])

# Summarize reviews
summary = SummarizeReviews(matrix_ngram1, aspects, 5)