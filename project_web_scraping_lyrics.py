import requests
from bs4 import BeautifulSoup
import re
import os, os.path
import pandas as pd
import numpy as np

import nltk
import nltk.data
from nltk.tokenize import TreebankWordTokenizer 
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

def get_url(artist):
    """
    Get the url link for the artist in www.lyrics.com 
    """
    return f"https://www.lyrics.com/artist/{artist}"
    
kl = get_url('Kendrick-Lamar')
sia = get_url('Sia')

def get_html(url):
    """
    """
    response = requests.get(url)
    return response.text

kl_html = get_html(kl)
    
def get_beautiful_soup(url, artist_name):
    """
    """
    file_name = f"{artist_name}_html"
    with open (file_name, "w") as f:
        f.write(get_html(url))

    with open(file_name) as f:
        return BeautifulSoup(f, 'html.parser')

soup_kl = get_beautiful_soup(kl, 'Kendrick_Lamar')
soup_sia = get_beautiful_soup(sia, 'Sia')

def song_names(soup):
    """
    Get the names of the songs from the 
    beautiful soup object of the artist
    """
    return [t.text for t in soup.find_all(class_ = 'tal qx')]

def get_links(soup, artist):
    """Get the complete link for each song of the artist"""
    links = f'links_{artist}' 
    links = []
    for td in soup.find_all('td'):
        if "tal" in td.get('class',[]):
              links.append('https://www.lyrics.com'+td.find('a')['href'])
    return links

def scrape_song_lyrics(url):
    page = requests.get(url)
    html = BeautifulSoup(page.text, 'html.parser')
    lyrics = html.find(class_='lyric-body').text
    #remove identifiers like chorus, verse, etc
    lyrics = re.sub(r'[\(\[].*?[\)\]]', '', lyrics)
    #remove empty lines
    lyrics = os.linesep.join([s for s in lyrics.splitlines() if s])
    
    return lyrics

def get_lyrics(soup, artist):  
    """ write a file with the lyrics of the artist"""
    c = 0 
    for link in get_links(soup, artist)[:5]:
        try:
            with open (f'{artist}_{song_names(soup_kl)[c].replace("/","")}.txt', 'w') as f:
                s = scrape_song_lyrics(link)
                f.write(s)
                c += 1
                print(f"Songs grabbed:{len(s)}")
        except AttributeError:  
               print(f"some exception at {link}: {c}")

get_lyrics(soup_kl,'Kendrick_Lamar')
get_lyrics(soup_sia, 'Sia')

path = os.path.expanduser('/home/xrusa/Documents/euclidean-eukalyptus/work_in_progress/week_4/nltk_data/corpora')
path in nltk.data.path

row_list = []

for subdir in ['Kendrick_Lamar', 'Sia']:
    for folder, subfolders, filenames in os.walk('/home/xrusa/Documents/euclidean-eukalyptus/work_in_progress/week_4/nltk_data/corpus/'+subdir):
        for file in filenames:
            d = {'artist':subdir}  # assign the name of the subdirectory to the label field
            with open('/home/xrusa/Documents/euclidean-eukalyptus/work_in_progress/week_4/nltk_data/corpus/'+subdir+'/'+file) as f:
                if f.read():      # handles the case of empty files, which become NaN on import
                    f.seek(0)
                    d['lyrics'] = f.read()  # assign the contents of the file to the review field
            row_list.append(d)
        break

df = pd.DataFrame(row_list)
df.dropna(inplace=True)
df.isna().sum()
df['lyrics'] = df['lyrics'].str.replace('\n', ' ')
df['lyrics'] = df['lyrics'].str.replace('\r', ' ')
df['lyrics'] = df['lyrics'].str.replace(r'[0-9]', ' ')

X = df['lyrics']
y = df['artist']

nltk.download("wordnet") 
nltk.download('stopwords')
STOPWORDS = stopwords.words('english')

def tokenize_corpus(CORPUS):
    """tokenize and lemmatize the corpus"""
    CORPUS = [s.lower() for s in CORPUS]
    CLEAN_CORPUS = []
    tokenizer = TreebankWordTokenizer()
    lemmatizer = WordNetLemmatizer()
    for doc in CORPUS:
        tokens = tokenizer.tokenize(text=doc)
        clean_doc = " ".join(lemmatizer.lemmatize(token) for token in tokens)
        CLEAN_CORPUS.append(clean_doc)
    return CLEAN_CORPUS

NEW_CORPUS = tokenize_corpus(X)

def dummy_fun(doc):
    return doc

steps = [('tf-idf', TfidfVectorizer(stop_words=STOPWORDS, max_df=0.8, analyzer='word', tokenizer=dummy_fun)),
         ('MNB', MultinomialNB())
        ]
          

pipeline = Pipeline(steps)

X_train, X_test, y_train,y_test = train_test_split(NEW_CORPUS, y, random_state=0)

pipeline.fit(X_train, y_train)

# print training set accuracy
train_acc = round(pipeline.score(X_train, y_train) * 100, 2)
print("Prediction accuracy (Training Set):", train_acc, "%\n")

# Check and print prediction accuracy and model parameters
test_acc=round(pipeline.score(X_test, y_test) * 100,2)
print("Prediction accuracy (Test Set):", test_acc, "%\n")
