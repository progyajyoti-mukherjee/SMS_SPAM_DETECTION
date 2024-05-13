import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
tfidf=pickle.load(open('vectorizer.pkl','rb'))
from sklearn.exceptions import InconsistentVersionWarning
import warnings

warnings.simplefilter("error", InconsistentVersionWarning)

try:
    model = pickle.load(open('model.pkl', 'rb'))
except InconsistentVersionWarning as w:
    print(w.original_sklearn_version)


ps=PorterStemmer()

def transform_text(text): 
    text=text.lower()
    text=nltk.word_tokenize(text)
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
        
    return " ".join(y)



st.title("SMS spam classifier")
sms=st.text_input("Enter the message")
if st.button("Predict"):
#preprocess
    transformed_sms=transform_text(sms)

#vectorize
    vector_input=tfidf.transform([transformed_sms])

#predict
    result=model.predict(vector_input)[0]

#display
    if result==1:
        st.header("Spam")
    else:
        st.header("Not Spam")