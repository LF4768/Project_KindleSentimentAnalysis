import streamlit as st
import nltk
# nltk.download("stopwords")
from nltk.corpus import stopwords
import re
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
import pickle
import pandas as pd
import numpy as np


st.write("### Enter Kindle Review To Check")

x = st.text_area("")

is_clicked = st.button("Click for Review")

@st.cache_resource
def load_models():
    with open('models/Word2Vec_model.pkl', 'rb') as file:
        w2v_model = pickle.load(file)
    with open('models/ML_model.pkl', 'rb') as file:
        rf_model = pickle.load(file)
    return w2v_model, rf_model

w2v_model, rf_model = load_models()

def convertToVec(text,model) -> pd.DataFrame:
    vectors = np.mean([model.wv[y] for y in x.split() if y in model.wv], axis=0)
    df = pd.DataFrame()
    df = pd.concat([df, pd.DataFrame(vectors.reshape(1,-1))], ignore_index=True)
    return df

def findResult(df,forest) -> bool:
    y = forest.predict(df)
    return y[0]


if(is_clicked and x != ""):
    lemma = WordNetLemmatizer()
    x = re.sub('[^a-z A-z 0-9-]+', '', x)
    x = ' '.join([y for y in x.split() if y not in stopwords.words('english')])
    x = re.sub(r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?','',str(x))
    x = BeautifulSoup(x,'lxml').get_text()
    x = ' '.join(x.split())
    x = ' '.join([lemma.lemmatize(y,pos='v') for y in x.split()])
    df = convertToVec(x, w2v_model)
    if findResult(df, rf_model) == 1:
        result = 'Positive'
    else:
        result = 'Negative'
    
    st.write(f"## The review is {result}")

