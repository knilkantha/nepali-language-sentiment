import os
import re
import time
import requests
import ast
import pickle
import json
import torch
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from langdetect import detect
from nepali_unicode_converter.convert import Converter
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains

# dataset = pd.read_csv("/media/gpu/157/Nepali_sentiment_Analysis/collected_labeled_data.csv")
review_url = "https://my.daraz.com.np/pdp/review/getReviewList?itemId=_id_&pageSize=5&filter=0&sort=0&pageNo=1"

model = pickle.load(open('bert_model/model','rb'))
tokenizers = pickle.load(open('tokenizers.pkl','rb'))
svc_sentiment = pickle.load(open('scv_sentiment','rb'))
chrome_options = Options()
chrome_options.add_argument("--headless")



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

def scrap_data():
    positive_sentimet = dataset.loc[dataset['label'] == 1]
    negative_sentiment = dataset.loc[dataset['label'] == 0]

    return positive_sentimet, negative_sentiment

def comment_sentiment(text):
    lang_list = ["hi","ne","mr"]
    converter = Converter()
    if detect(text) == "ne":
        embedding = get_bert_embedding_sentence(text)
        svc_pred = svc_sentiment.predict(embedding.reshape(1,-1))[0]
    """
    if detect(text) not in lang_list:
        result = converter.convert(text)
        embedding = get_bert_embedding_sentence(result)
        svc_pred = svc_sentiment.predict(embedding.reshape(1,-1))[0]
        # predicted_label.append(svc_pred)
        # comment_text.append(review["reviewContent"])
    else:
        embedding = get_bert_embedding_sentence(text)
        svc_pred = svc_sentiment.predict(embedding.reshape(1,-1))[0]
        # predicted_label.append(svc_pred)
        # comment_text.append(review["reviewContent"])
    """
    return svc_pred

def scrape_comment(url):
    lang_list = ["hi","ne","mr"]
    converter = Converter()
    id = url.split("-")[-2].replace("i","")
    api_url = review_url.replace("_id_",id)
    print("---------------------------------")
    response = requests.get(api_url).text
    print(response)
    response = json.loads(response)
    df = pd.DataFrame(columns=["text",'label'])
    reviews = response["model"]["items"]
    predicted_label =[]
    comment_text =[]
    
    for review in reviews:
        text = review["reviewContent"]
        try:
            
            if detect(text) not in lang_list:
                result = converter.convert(text)
                embedding = get_bert_embedding_sentence(result)
                svc_pred = svc_sentiment.predict(embedding.reshape(1,-1))[0]
                predicted_label.append(svc_pred)
                comment_text.append(review["reviewContent"])
            else:
                embedding = get_bert_embedding_sentence(text)
                svc_pred = svc_sentiment.predict(embedding.reshape(1,-1))[0]
                predicted_label.append(svc_pred)
                comment_text.append(review["reviewContent"])
        except Exception as e:
            print(e)
            pass
    df['text'] = comment_text
    df['label'] = predicted_label
    positive_sentimet = df.loc[df['label'] == 1]
    negative_sentiment = df.loc[df['label'] == 0]
    return positive_sentimet, negative_sentiment

# def scrap_twitter(url):
#     tweets = driver.find_elements(By.XPATH,'//*[@id="id__nspdargek9"]/span/text()')
#     print(tweets)

def scrape_twitter(url):
    '''
        to scrape tweet from given username provide username and tweet id
    '''
    driver = webdriver.Chrome("driver/chromedriver",options=chrome_options)

    # driver.get(f"https://twitter.com/{username}/status/{tweet_id}")
    driver.get(url)
    time.sleep(5) #change according to your pc and internet connection

    tweets = []
    result = False
    old_height = driver.execute_script("return document.body.scrollHeight")

    #set initial all_tweets to start loop
    all_tweets = driver.find_elements(By.XPATH, '//div[@data-testid]//article[@data-testid="tweet"]')

    while result == False:

        for item in all_tweets[1:]: # skip tweet already scrapped

            try:
                text = item.find_element(By.XPATH, './/div[@data-testid="tweetText"]').text
            except:
                text = '[empty]'

            #Append new tweets replies to tweet array
            tweets.append(text)

        #scroll down the page
        driver.execute_script("window.scrollTo(0,document.body.scrollHeight)")

        time.sleep(2)

        try:
            try:
                button = driver.find_element_by_css_selector("div.css-901oao.r-1cvl2hr.r-37j5jr.r-a023e6.r-16dba41.r-rjixqe.r-bcqeeo.r-q4m81j.r-qvutc0")
            except:
                button = driver.find_element_by_css_selector("div.css-1dbjc4n.r-1ndi9ce") #there are two kinds of buttons

            ActionChains(driver).move_to_element(button).click(button).perform()
            time.sleep(2)
            driver.execute_script("window.scrollTo(0,document.body.scrollHeight)")
            time.sleep(2)
        except:
            pass

        new_height = driver.execute_script("return document.body.scrollHeight")

        if new_height == old_height:
            result = True
        old_height = new_height

        #update all_tweets to keep loop
        all_tweets = driver.find_elements(By.XPATH, '//div[@data-testid]//article[@data-testid="tweet"]')
    driver.close()
    text = []
    predicted_label = []
    for comments in tweets:
        try:
            result = comment_sentiment(comments)
            comments = remove_emojis(comments)
            text.append(comments)
            predicted_label.append(result)
        except Exception as e:
            pass 
    df = pd.DataFrame(columns=["text","label"])
    df['text'] = text
    df['label'] = predicted_label
    positive_sentimet = df.loc[df['label'] == 1]
    negative_sentiment = df.loc[df['label'] == 0]
    return positive_sentimet, negative_sentiment


def scrape_youtube(url):
    driver = webdriver.Chrome("driver/chromedriver",options=chrome_options)
    data =[]
    
    wait = WebDriverWait(driver,15)
    driver.get(url)
    predicted_label = []

    for item in range(5):
        wait.until(EC.visibility_of_element_located((By.TAG_NAME, "body"))).send_keys(Keys.END)
        time.sleep(5)
    for comment in wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "#content"))):
        data.append(comment.text)

    text =[] 
    for comments in data:
        try:
            result =comment_sentiment(comments)
            comments = remove_emojis(comments)
            text.append(comments)
            predicted_label.append(result)
        except Exception as e:
            # raise
            pass
    driver.close()
    df = pd.DataFrame(columns=["text","label"])
    df['text'] = text
    df['label'] = predicted_label
    positive_sentimet = df.loc[df['label'] == 1]
    negative_sentiment = df.loc[df['label'] == 0]
    return positive_sentimet, negative_sentiment

if __name__ == "__main__":
    url = "https://www.youtube.com/watch?v=uD58-EHwaeI"
    positive_sentimet, negative_sentiment= scrap_youtube(url=url)
    print(positive_sentimet, negative_sentiment)

