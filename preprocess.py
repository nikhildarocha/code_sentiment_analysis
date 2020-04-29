import numpy as np
import pandas as pd
import re
import nltk
import spacy
import string
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import names
# Reading the dataset
df = pd.read_csv("@Barclaycard_tweets.csv")
df.to_csv('readfile.csv') 


df['text_lower']  = df['text'].str.lower()
df['text_lower'].to_csv('lowertext.csv')

df['text_punct'] = df['text'].str.replace('[^\w\s]','')
df['text_punct'].to_csv('punctionremoval.csv')

from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))

def stopwords(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])
# Applying the stopwords to 'text_punct' and store into 'text_stop'
df["text_stop"] = df["text_punct"].apply(stopwords)
df["text_stop"].to_csv('stopwordremoval.csv')

from collections import Counter
cnt = Counter()
for text in df["text_stop"].values:
    for word in text.split():
        cnt[word] += 1
        
print(cnt.most_common(10))

freq = set([w for (w, wc) in cnt.most_common(10)])
# function to remove the frequent words
def freqwords(text):
    return " ".join([word for word in str(text).split() if word not 
in freq])
# Passing the function freqwords
df["text_common"] = df["text_stop"].apply(freqwords)
df["text_common"].to_csv('commonwordremoval.csv')


freq = pd.Series(' '.join(df['text_common']).split()).value_counts()[-10:] # 10 rare words
freq = list(freq.index)
df['text_rare'] = df['text_common'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
df['text_rare'].to_csv('raretext.csv')

#from textblob import TextBlob
#df['text_rare'][:3239].apply(lambda x: str(TextBlob(x).correct())).to_csv('spellingcorrect.csv')

# Function for url's
def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

#Passing the function to 'text_rare'
df['text_rare_sansurls'] = df['text_rare'].apply(remove_urls)
df['text_rare_sansurls'].to_csv('raretextsansurls.csv')



# lemmatisation
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV} # Pos tag, used Noun, Verb, Adjective and Adverb
# Function for lemmatization using POS tag
def lemmatize_words(text):
    pos_tagged_text = nltk.pos_tag(text.split())
    return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])
# Passing the function to 'text_rare' and store in 'text_lemma'
df["text_lemma"] = df["text_rare_sansurls"].apply(lemmatize_words)
df["text_lemma"].to_csv('lemmatized_text.csv')

#Creating function for tokenization
def tokenization(text):
    text = re.split('\W+', text)
    return text
# Passing the function to 'text_rare' and store into'text_token'
df['text_token'] = df['text_lemma'].apply(lambda x: tokenization(x.lower()))
df['text_token'].to_csv('tokenised_text.csv')

def word_feats(words):
    return dict([(word, True) for word in words])
 
positive_vocab = [ 'awesome', 'outstanding', 'fantastic', 'terrific', 'good', 'nice', 'great', ':)' ]
negative_vocab = [ 'bad', 'terrible','useless', 'hate', ':(' ]
neutral_vocab = [ 'movie','the','sound','was','is','actors','did','know','words','not' ]

positive_features = [(word_feats(pos), 'pos') for pos in positive_vocab]
negative_features = [(word_feats(neg), 'neg') for neg in negative_vocab]
neutral_features = [(word_feats(neu), 'neu') for neu in neutral_vocab]

train_set = negative_features + positive_features + neutral_features

classifier = NaiveBayesClassifier.train(train_set) 

# Predict
neg = 0
pos = 0
neu = 0
words = df['text_token']
for word in words:
    classResult = classifier.classify( word_feats(word))
    if classResult == 'neg':
        neg = neg + 1
    if classResult == 'pos':
        pos = pos + 1
    if classResult == 'neu':
        neu = neu + 1

print('Positive: ' + str(float(pos)/len(words)))
print('Negative: ' + str(float(neg)/len(words)))
print('Neutral: '   + str(float(neu)/len(words)))
