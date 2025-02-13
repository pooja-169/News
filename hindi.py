# -*- coding: utf-8 -*-
"""Hindi_sentimental_dashboard.py"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import MarianTokenizer, MarianMTModel, AutoTokenizer, AutoModelForSequenceClassification
import torch
import requests
import config
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from openpyxl import load_workbook


response = requests.get('https://www.bhaskar.com/')

soup=BeautifulSoup(response.content,'html.parser')


existing_file = 'webscrap.xlsx'

wb = load_workbook(existing_file)
print(wb)
 

ws = wb.active
 
i=0

headlines= soup.find_all('h3')
for headline in headlines:
    new_data = [[i,headline.text]]
    i=i+1
    for row in new_data:
        ws.append(row)
    wb.save(existing_file)

# Streamlit Dashboard Setup
st.set_page_config(layout="wide", page_title="Hindi Sentiment Analysis Dashboard", page_icon="📰")

# Main Title
st.markdown("<h1 style='text-align: center; color: #2C3E50;'>Hindi Sentiment Analysis Dashboard</h1>", unsafe_allow_html=True)

# Load translation and sentiment models
@st.cache_resource(show_spinner=False)
def load_models():
    # Load translation model
    translation_model_name = 'Helsinki-NLP/opus-mt-hi-en'
    translation_tokenizer = MarianTokenizer.from_pretrained(translation_model_name)
    translation_model = MarianMTModel.from_pretrained(translation_model_name)
    
    # Load sentiment model
    sentiment_model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
    sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)
    
    return translation_tokenizer, translation_model, sentiment_tokenizer, sentiment_model

translation_tokenizer, translation_model, sentiment_tokenizer, sentiment_model = load_models()

# Function to translate text from Hindi to English
def translate_text(text):
    inputs = translation_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        translated = translation_model.generate(**inputs)
    return translation_tokenizer.decode(translated[0], skip_special_tokens=True)

# Function to analyze sentiment
def analyze_sentiment(text):
    inputs = sentiment_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = sentiment_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

    pos_score = probs[0][2].item()  # Positive
    neu_score = probs[0][1].item()  # Neutral
    neg_score = probs[0][0].item()  # Negative

    return pos_score, neu_score, neg_score

# Function to calculate intensity
def calculate_intensity(pos_score, neu_score, neg_score):
    intensity = pos_score - neg_score  # Intensity between -1 and 1
    return round(max(min(intensity, 1), -1), 2)

# Load and process data
def load_and_process_data(input_file):
    df = pd.read_csv(input_file)
    df['Title_English'] = df['Title'].apply(translate_text)
    
    results = df['Title_English'].apply(lambda x: pd.Series(analyze_sentiment(x)))
    df[['positive', 'neutral', 'negative']] = results
    df['intensity'] = df.apply(lambda row: calculate_intensity(row['positive'], row['neutral'], row['negative']), axis=1)
    
    return df

input_file = st.sidebar.file_uploader("Upload Hindi News CSV", type=["csv"])

if input_file is not None:
    df = load_and_process_data(input_file)

    st.markdown("<h2 style='color: #2980B9;'>Top 5 News Articles </h2>", unsafe_allow_html=True)
    df_sorted = df.sort_values(by='positive', ascending=False).head(5)

    st.markdown(
        """
        <style>
        .top-news-box {
            background-color: #ECECEC;
            padding: 15px;
            margin: 10px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        .top-news-title {
            font-size: 1.2em;
            font-weight: bold;
            color: #2C3E50;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    for index, row in df_sorted.iterrows():
        st.markdown(
            f"""
            <div class="top-news-box">
                <div class="top-news-title">{row['Title']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Visual Analysis Section
    st.markdown("<h2 style='color: #2980B9;'>Visual Analysis</h2>", unsafe_allow_html=True)

    # Aligning and resizing the charts to be uniform in size
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("<h3 style='color: #2C3E50;'>Sentiment Distribution</h3>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 5))
        sns.countplot(data=df, x='intensity', palette='coolwarm', ax=ax)
        ax.set_title("Sentiment Intensity Distribution", fontsize=14)
        st.pyplot(fig)

    with col2:
        st.markdown("<h3 style='color: #2C3E50;'>Intensity Box Plot</h3>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 5))
        sns.boxplot(data=df, y='intensity', palette='coolwarm', ax=ax)
        ax.set_title("Sentiment Intensity Box Plot", fontsize=14)
        st.pyplot(fig)

    with col3:
        st.markdown("<h3 style='color: #2C3E50;'>Correlation Heatmap</h3>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 5))
        corr = df[['positive', 'neutral', 'negative']].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        ax.set_title('Correlation Heatmap', fontsize=14)
        st.pyplot(fig)

    # Final styling
    st.markdown("""
    <style>
        .stApp {
            background-color: #F7F9F9;
            color: #2C3E50;
        }
        h1, h2, h3 {
            font-family: 'Arial', sans-serif;
        }
        .css-18e3th9 {
            padding: 5px 2px;
        }
        .css-1d391kg {
            background-color: #ECF0F1;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
        }
        .css-qbe2hs {
            background-color: #BDC3C7;
            padding: 8px;
            border-radius: 8px;
        }
    </style>
    """, unsafe_allow_html=True)
else:
    st.warning("Please upload a CSV file to continue.")
