from flask import Flask, request, render_template

import pandas as pd
import re
import string
import time
import random
from prometheus_client import start_http_server
from prometheus_client import Counter, Summary, Gauge, Histogram

import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import numpy as np

import os
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from joblib import load


REQUESTS =Counter('twitter_search_app_calls_total', 'How many times the app was called')
EXCEPTIONS =Counter('twitter_search_app_exceptions_total', 'How many exceptions the app triggers')
INPROGRESS = Gauge('twitter_search_app_inprogress', 'number of requests in progress')
LAST = Gauge('twitter_search_app_times_seconds', 'the last time our app was called')
LATENCY_SUM = Summary('twitter_search_app_latency_sum_seconds', 'the time needed for a request')
LATENCY_HIS = Histogram('twitter_search_app_latency_his_seconds', 'the time needed for a request')


MODEL_DIR = os.environ["MODEL_DIR"]
MODEL_FILE = os.environ["MODEL_FILE"]
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)

app = Flask(__name__)
model = load(MODEL_PATH)

     
@app.route('/')
def home():   
    return render_template('index.html')
    
@app.route('/', methods=['POST'])
def predict():
    LAST.set(time.time())
    REQUESTS.inc()
    start = time.time()
    rand = random.random()
    #with EXCEPTIONS.count_exceptions():
    #     if rand < 0.2:
    #         raise Exception
    
    INPROGRESS.inc()
    #time.sleep(5)
    if rand < 0.5:
        time.sleep(rand * 0.1)
        
    tweet = pd.read_csv("tweets.csv")
    tweets = tweet['text']
    
    output = []
    score = []
    
    if request.method == 'POST':
        enter = request.form['search']
        test_doc = word_tokenize(enter.lower())
        test_doc_vector = model.infer_vector(test_doc)
        similar_doc=model.docvecs.most_similar(positive = [test_doc_vector], topn=20)
        
        
        for i in range(0,len(similar_doc)):
            index = int(similar_doc[i][0])
            score = similar_doc[i][1]
            sen = tweets[index]
            tup = (score, sen)
            output.append(tup)
            
        INPROGRESS.dec()
        last = time.time()
        LATENCY_SUM.observe(last - start)
        LATENCY_HIS.observe(last - start)
    		
        return render_template('result.html', tweets=output)
    
if __name__ == '__main__':
    start_http_server(8000)
    app.run(host='0.0.0.0')
