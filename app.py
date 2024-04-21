from dataclasses import asdict
from stat import FILE_ATTRIBUTE_NO_SCRUB_DATA
import streamlit as st
import pickle 
import torch
from googletrans import Translator
from langdetect import detect

from transformers import BertTokenizer, BertModel, BertForMaskedLM, AutoTokenizer, AutoModelForMaskedLM
from scipy.spatial.distance import cosine 
import tokenizers 
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from nltk.corpus import stopwords

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from nepali_unicode_converter.convert import Converter
from textblob import TextBlob


# model = AutoModelForMaskedLM.from_pretrained("Shushant/nepaliBERT", output_hidden_states = True, return_dict = True, output_attentions = True)

# tokenizers = AutoTokenizer.from_pretrained("Shushant/nepaliBERT")
# pickle.dump(model, open('nepaliBert.pkl','wb'))
# pickle.dump(tokenizers, open('tokenizers.pkl','wb'))
model = pickle.load(open('bert_model/model','rb'))
tokenizers = pickle.load(open('bert_model/tokenizer','rb'))
# if torch.cuda.is_available():  

#     dev = "cuda:0" 
# else:  

#     dev = "cpu"  

# print(dev)
device = torch.device("cpu")  

st.header("Nepali sentiment analysis")
st.subheader("This app gives the sentiment analysis of Nepali text.")




def get_bert_embedding_sentence(input_sentence):
    md = model
    tokenizer = tokenizers
    marked_text = " [CLS] " + input_sentence + " [SEP] "
    tokenized_text = tokenizer.tokenize(marked_text)

    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(indexed_tokens) 
    

    tokens_tensors = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    
    with torch.no_grad():
        outputs = md(tokens_tensors, segments_tensors)
        hidden_states = outputs.hidden_states

    token_vecs = hidden_states[-2][0]

    sentence_embedding = torch.mean(token_vecs, dim=0)

    return sentence_embedding.numpy()
lang_list = ["hi","ne","mr"]
svc_sentiment = pickle.load(open('scv_sentiment','rb'))
text = st.text_input("Please input your nepali sentence here:")
translator = Translator()
converter = Converter()
if text:
    st.write("Your input text is:          ", text)
    if detect(text) not in lang_list:
        if detect(text) != "en":
            text = text.lower()
            result = converter.convert(text)
            st.write(result)
            embedding = get_bert_embedding_sentence(result)
            svc_pred = svc_sentiment.predict(embedding.reshape(1,-1))[0]
            if svc_pred == 0:
                st.write("Sentiment is: NEGATIVE ")
            else:
                st.write("Sentiment is: POSITIVE ")
        elif detect(text)=='en':
            st.write("Sorry our app can't understand english text")
     
    else:
        embedding = get_bert_embedding_sentence(text)
        svc_pred = svc_sentiment.predict(embedding.reshape(1,-1))[0]
        if svc_pred == 0:
            st.write("Sentiment is: NEGATIVE ")
        else:
            st.write("Sentiment is: POSITIVE ")

