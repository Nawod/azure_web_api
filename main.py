import numpy as np
from fastapi import FastAPI, Form, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import pandas as pd
from starlette.responses import HTMLResponse
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


app = FastAPI(title="REST API using FastAPI malicious URL classification EndPoints")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
app.add_middleware(GZipMiddleware)

@app.on_event("startup")
async def startup():
    #load the saved model
    loaded_model = await tf.keras.models.load_model('mal_url')

    #import and train word tokenizer
    data = pd.read_csv('archive/url_train.csv')
    tokenizer = Tokenizer(num_words=10000, split=' ')
    tokenizer.fit_on_texts(data['url'].values)




@app.get('/') #basic get view
def basic_view():
    return {"WELCOME": "GO TO /docs route, or /post or send post request to /predict "}

@app.get('/predict', response_class=HTMLResponse)
def take_inp():
    return '''
        <form method="post">
        <input maxlength="28" name="text" type="text" value="https github com Nawod NDS" />
        <input type="submit" />'''

#tokenize inputs
def token(text):
    X = tokenizer.texts_to_sequences(pd.Series(text).values)
    Y = pad_sequences(X, maxlen=200)
    return Y

@app.post('/predict')
async def predict(text:str = Form(...)):
    # clean_text = preProcess_data(text) #clean the text
    embeded_text = await token(text)
    predictions = loaded_model.predict(embeded_text) #predict the text
    
    sentiment = (predictions > 0.5).astype(np.int) #calculate the index of max sentiment
    # sentiment = 0
   
    if sentiment==0:
         t_sentiment = 'bad' #set appropriate sentiment
    elif sentiment==1:
         t_sentiment = 'good'
    return { #return the dictionary for endpoint
         "OUTPUT" : t_sentiment
         
    }