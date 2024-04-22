import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import tokenizers 
from nltk.corpus import stopwords
import snowballstemmer 
import numpy
import re

# loading model from huggingface
model = AutoModelForMaskedLM.from_pretrained("Shushant/nepaliBERT", output_hidden_states = True, return_dict = True, output_attentions = True)
tokenizers = AutoTokenizer.from_pretrained("Shushant/nepaliBERT")

stopwords = stopwords.words("nepali")
nepali_stemmer = snowballstemmer.NepaliStemmer()

def remove_emojis(text):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f" # dingbats
        u"\u3030"
    "]+", re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    return text

def clean_text(text):
    text = remove_emojis(text)
    text = text.split(' ')
    clean_text_list = []
    for word in text:
        if word not in stopwords:
            clean_text_list.append(word)
    clean_text = ' '.join(clean_text_list)
    stem_words = nepali_stemmer.stemWords(clean_text.split())
#     stem_text = ' '.join(stem_words)
#     txt = re.sub(r"[|a-zA-z.'#0-9@,:?'\u200b\u200c\u200d!/&~-]",'',stem_text)
    return ' '.join([i for i in stem_words])

def get_bert_embedding_sentence(input_sentence):
    '''function to generate nepali bert embedding'''
    md = model
    tokenizer = tokenizers
    marked_text = " [CLS] " + input_sentence + " [SEP] "
    tokenized_text = tokenizer.tokenize(marked_text)

    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(indexed_tokens) 
    
    # Convert inputs to Pytorch tensors
    tokens_tensors = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    
    with torch.no_grad():
        outputs = md(tokens_tensors, segments_tensors)
        # removing the first hidden state
        # the first state is the input state 

        hidden_states = outputs.hidden_states

    token_vecs = hidden_states[-2][0]

    # Calculate the average of all 22 token vectors.
    sentence_embedding = torch.mean(token_vecs, dim=0)
    return sentence_embedding.numpy()