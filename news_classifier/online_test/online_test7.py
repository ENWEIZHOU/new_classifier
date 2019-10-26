import re
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import nltk
from gensim.models import Word2Vec
from scipy.linalg import norm
from scipy.spatial.distance import cosine
from gensim import models
import pandas as pd
def compute_doc_vec_single(clean_text, model):
    vec = np.zeros((model.vector_size,), dtype=np.float32)
    n = 0
    tokenized_clean_text=nltk.word_tokenize(clean_text)
    print(tokenized_clean_text)
    for word in tokenized_clean_text:
        if word in model:
            vec += model[word]
            n += 1
    if(n==0):
        print("*****************")
        return (model["Hello"]*0)
    else:
        return (vec/n)

if __name__=="__main__":
    print("Loading the GoogleNews-vectors-negative300.bin...")
    model = models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    data = {'text':["title 'National tragedy': Trump begins border wall construction in Unesco reserve.", "title Trump administration enters new phase for border wall, sets ambitious timetable after securing land"]}
    stemmer = SnowballStemmer('english')
    words = stopwords.words('english')
    dataFrame = pd.DataFrame(data)
    dataFrame['cleaned_text']= dataFrame['text'].apply(lambda x:" ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]"," ",x).split() if i not in words]))
    # print(dataFrame)
    s1=dataFrame['cleaned_text'][0]
    s2=dataFrame['cleaned_text'][1]
    print(s1)
    print(s2)
    s1=compute_doc_vec_single(s1, model)
    # # print(s1)
    s2 = compute_doc_vec_single(s2, model)
    # print(nltk.word_tokenize("Trump administration enters new phase for border wall, sets ambitious timetable after securing land"))
    # print(1-(np.dot(s1, s2) / (norm(s1) * norm(s2))))
    print("Calculating the cosine distance of vectors...")
    print(cosine(s1,s2))