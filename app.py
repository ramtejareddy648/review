import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,Dense,LSTM,Dropout
from tensorflow.keras.models import load_model
import pickle

with open('/content/drive/MyDrive/projectreview/toke.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

model=load_model('/content/drive/MyDrive/projectreview/review.h5')


def predict_sentiment(text):
  sequences_text=tokenizer.texts_to_sequences([text])
  padd_sequences_text=pad_sequences(sequences_text,maxlen=200,padding='post')
  prediction=model.predict(padd_sequences_text)
  print(prediction)

  if prediction[0][0]>0.5:
    result='review is positive'
  else :
    result='review is negative'
  
  return prediction[0][0],result


st.title('Movie Review System')
user_input=st.text_area('enter review')

if st.button('predict'):
  pred_score,predictions=predict_sentiment(user_input)
  st.write('review text is :',user_input)
  st.write('predction is :',predictions)
  st.write('pred_score is :',pred_score)

