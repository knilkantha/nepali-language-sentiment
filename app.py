import os 
import streamlit as st
import pickle 
import torch
from googletrans import Translator
from langdetect import detect

from transformers import AutoTokenizer, AutoModelForMaskedLM
import tokenizers 
from nepali_unicode_converter.convert import Converter

from preprocess import clean_text, get_bert_embedding_sentence

if not os.path.exists("bert_model"):
    os.makedirs("bert_model")

else:
    pass

# downloading and saving Nepali BERT model from huggingface    
model = AutoModelForMaskedLM.from_pretrained("Shushant/nepaliBERT", output_hidden_states = True, return_dict = True, output_attentions = True)
pickle.dump(model, open('bert_model/nepaliBert.pkl','wb'))

# downloading and saving Nepali tokenizers 
tokenizers = AutoTokenizer.from_pretrained("Shushant/nepaliBERT")
pickle.dump(tokenizers, open('bert_model/tokenizers.pkl','wb'))


# loading bert and tokenizers 
model = pickle.load(open('bert_model/nepaliBert.pkl','rb'))
tokenizers = pickle.load(open('bert_model/tokenizers.pkl','rb'))
# if torch.cuda.is_available():  

device = torch.device("cpu")  

st.header("Nepali sentiment analysis")
st.subheader("This app gives the sentiment analysis of Nepali text.")

lang_list = ["hi","ne","mr"]
svc_sentiment = pickle.load(open('trained_model/scv_sentiment','rb'))
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
            embedding = get_bert_embedding_sentence(clean_text(result))
            svc_pred = svc_sentiment.predict(embedding.reshape(1,-1))[0]
            if svc_pred == 0:
                st.write("Sentiment is: NEGATIVE ")
            else:
                st.write("Sentiment is: POSITIVE ")
        elif detect(text)=='en':
            st.write("Sorry our app can't understand english text")
     
    else:
        embedding = get_bert_embedding_sentence(clean_text(text))
        svc_pred = svc_sentiment.predict(embedding.reshape(1,-1))[0]
        print(svc_pred)
        if svc_pred == 0:
            st.write("Sentiment is: NEGATIVE ")
        else:
            st.write("Sentiment is: POSITIVE ")

