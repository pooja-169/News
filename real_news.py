# -*- coding: utf-8 -*-
"""streamlit_fake_news_dashboard.py"""

import streamlit as st
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
from openpyxl import load_workbook
from wordcloud import WordCloud
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import os

# Streamlit Dashboard Setup
st.set_page_config(layout="wide", page_title="Fake News Detection & Analysis Dashboard", page_icon="📰")

# Main Title
st.markdown("<h1 style='text-align: center; color: #2C3E50;'>Fake News Detection & Analysis Dashboard</h1>", unsafe_allow_html=True)

# Load models and tokenizers
@st.cache_resource(show_spinner=False)
def load_models_and_tokenizers(bert_model_name, roberta_model_name, sentiment_model_name):
    bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
    bert_model = AutoModelForSequenceClassification.from_pretrained(bert_model_name)
    roberta_tokenizer = AutoTokenizer.from_pretrained(roberta_model_name)
    roberta_model = AutoModelForSequenceClassification.from_pretrained(roberta_model_name)
    sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
    sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)
    return bert_tokenizer, bert_model, roberta_tokenizer, roberta_model, sentiment_tokenizer, sentiment_model

bert_model_name = "bert-base-uncased"
roberta_model_name = "roberta-base"
sentiment_model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
bert_tokenizer, bert_model, roberta_tokenizer, roberta_model, sentiment_tokenizer, sentiment_model = load_models_and_tokenizers(bert_model_name, roberta_model_name, sentiment_model_name)

# Function to predict fake news using both models
def predict_fake_news(text):
    # Tokenize and predict with BERT
    bert_inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        bert_outputs = bert_model(**bert_inputs)
    bert_logits = bert_outputs.logits
    bert_probs = torch.softmax(bert_logits, dim=1).squeeze().tolist()

    # Tokenize and predict with RoBERTa
    roberta_inputs = roberta_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        roberta_outputs = roberta_model(**roberta_inputs)
    roberta_logits = roberta_outputs.logits
    roberta_probs = torch.softmax(roberta_logits, dim=1).squeeze().tolist()

    # Average the probabilities from both models
    avg_probs = [(b + r) / 2 for b, r in zip(bert_probs, roberta_probs)]

    # Extracting the class with the highest average probability
    predicted_class = torch.argmax(torch.tensor(avg_probs)).item()

    labels = ["Real", "Fake"]
    return labels[predicted_class], avg_probs

# Function to predict sentiment (positive/negative) for real news
def predict_sentiment(text):
    sentiment_inputs = sentiment_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        sentiment_outputs = sentiment_model(**sentiment_inputs)
    sentiment_logits = sentiment_outputs.logits
    sentiment_probs = torch.softmax(sentiment_logits, dim=1).squeeze().tolist()

    sentiment_labels = ["Negative", "Neutral", "Positive"]
    predicted_sentiment = sentiment_labels[torch.argmax(torch.tensor(sentiment_probs)).item()]
    return predicted_sentiment, sentiment_probs

# Scrape data from Times of India
def scrape_toi_news():
    response = requests.get('https://timesofindia.indiatimes.com/briefs')
    soup = BeautifulSoup(response.content, 'html.parser')
    cards = soup.find_all('div', class_='brief_box')

    data = []
    for i, card in enumerate(cards):
        headline = card.find('h2')
        headline_text = headline.get_text(strip=True) if headline else "No headline found"

        description = card.find('p')
        description_text = description.get_text(strip=True) if description else "No description found"

        span = card.find('span', class_='subsection_card')
        span_text = span.get_text(strip=True) if span else "No span found"

        data.append([i, headline_text, description_text, span_text])
    
    return pd.DataFrame(data, columns=['ID', 'Headline', 'Description', 'Category'])

# Load scraped data or user-uploaded data
@st.cache_data
def load_data(existing_file=None):
    if existing_file:
        return pd.read_csv(existing_file)
    else:
        df = scrape_toi_news()
        df.to_csv("webscrap_fake_real_final.csv", index=False)
        return df

st.sidebar.markdown("### Data Source")
data_source = st.sidebar.radio("Choose data source:", ("Scrape Times of India", "Upload CSV"))

if data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Choose a file", type=["csv"])
    if uploaded_file:
        df = load_data(uploaded_file)
    else:
        st.warning("Please upload a CSV file.")
        st.stop()
else:
    df = load_data()

# Predict fake/real news and sentiment analysis
@st.cache_data(show_spinner=False)
def process_data(df):
    predictions = []
    real_probs = []
    fake_probs = []
    sentiments = []
    sentiment_probs = []

    for description in df['Description']:
        prediction, probs = predict_fake_news(description)
        predictions.append(prediction)
        real_probs.append(probs[0])
        fake_probs.append(probs[1])

        if prediction == "Real":
            sentiment, s_probs = predict_sentiment(description)
            sentiments.append(sentiment)
            sentiment_probs.append(s_probs)
        else:
            sentiments.append("N/A")
            sentiment_probs.append([0, 0, 0])

    df['Prediction'] = predictions
    df['Real_Prob'] = real_probs
    df['Fake_Prob'] = fake_probs
    df['Sentiment'] = sentiments

    # Clustering using DBSCAN
    features = np.array(list(zip(df['Real_Prob'], df['Fake_Prob'])))
    dbscan = DBSCAN(eps=0.1, min_samples=5)
    df['Cluster'] = dbscan.fit_predict(features)

    return df

df = process_data(df)

# Visual Analysis Section
st.markdown("<h2 style='color: #2980B9;'>Visual Analysis</h2>", unsafe_allow_html=True)

# Aligning and resizing the first three charts
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.markdown("<h3 style='color: #2C3E50;'>Prediction Distribution</h3>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.countplot(data=df, x='Prediction', palette='coolwarm', ax=ax)
    ax.set_title("Real vs Fake News", fontsize=14)
    st.pyplot(fig)

with col2:
    st.markdown("<h3 style='color: #2C3E50;'>Category Distribution</h3>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.countplot(data=df, x='Category', palette='coolwarm', ax=ax)
    ax.set_title("Category Distribution", fontsize=14)
    st.pyplot(fig)

with col3:
    st.markdown("<h3 style='color: #2C3E50;'>Word Cloud of Headlines</h3>", unsafe_allow_html=True)
    text = " ".join(headline for headline in df['Headline'])
    if text.strip():  # Ensure there are words to generate a word cloud
        wordcloud = WordCloud(width=400, height=400, background_color='white').generate(text)
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)
    else:
        st.write("No headlines available to generate a word cloud.")

# Cluster Visualization Section
st.markdown("<h2 style='color: #2980B9;'>Cluster Visualization</h2>", unsafe_allow_html=True)

st.markdown("<h3 style='color: #2C3E50;'>DBSCAN Clusters of News Articles</h3>", unsafe_allow_html=True)
fig, ax = plt.subplots(figsize=(8, 5))
sns.scatterplot(
    x='Real_Prob',
    y='Fake_Prob',
    hue='Cluster',
    palette='Set1',
    data=df,
    legend='full',
    s=100,
    ax=ax
)
ax.set_title('DBSCAN Clustering of News Articles', fontsize=14)
ax.set_xlabel('Real Probability', fontsize=12)
ax.set_ylabel('Fake Probability', fontsize=12)
st.pyplot(fig)

# Sentiment Analysis Visualization Section
st.markdown("<h2 style='color: #2980B9;'>Sentiment Analysis of Real News</h2>", unsafe_allow_html=True)

col4, col5 = st.columns([1, 1])

with col4:
    st.markdown("<h3 style='color: #2C3E50;'>Sentiment Distribution of Real News</h3>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(data=df[df['Prediction'] == 'Real'], x='Sentiment', palette='coolwarm', ax=ax)
    ax.set_title('Positive vs Negative News', fontsize=14)
    st.pyplot(fig)

with col5:
    st.markdown("<h3 style='color: #2C3E50;'>Word Cloud of Positive Real News</h3>", unsafe_allow_html=True)
    positive_text = " ".join(df[(df['Prediction'] == 'Real') & (df['Sentiment'] == 'Positive')]['Headline'])
    if positive_text.strip():  # Ensure there are words to generate a word cloud
        positive_wordcloud = WordCloud(width=400, height=400, background_color='white').generate(positive_text)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(positive_wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)
    else:
        st.write("No positive news headlines available to generate a word cloud.")

# Advanced Analysis Section
st.markdown("<h2 style='color: #2980B9;'>Advanced Analysis</h2>", unsafe_allow_html=True)

col6, col7 = st.columns([1, 1])

with col6:
    st.markdown("<h3 style='color: #2C3E50;'>Real vs Fake Box Plot</h3>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.boxplot(data=df, x='Prediction', y='Real_Prob', palette='coolwarm', ax=ax)
    ax.set_title('Real Probability Distribution', fontsize=14)
    st.pyplot(fig)

with col7:
    st.markdown("<h3 style='color: #2C3E50;'>Correlation Heatmap</h3>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    corr = df[['Real_Prob', 'Fake_Prob', 'Cluster']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title('Correlation Heatmap', fontsize=14)
    st.pyplot(fig)

# Word Clouds for Real and Fake News
col8, col9 = st.columns([1, 1])

with col8:
    st.markdown("<h3 style='color: #2C3E50;'>Word Cloud of Real News</h3>", unsafe_allow_html=True)
    real_text = " ".join(df[df['Prediction'] == 'Real']['Headline'])
    if real_text.strip():  # Ensure there are words to generate a word cloud
        real_wordcloud = WordCloud(width=400, height=400, background_color='white').generate(real_text)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(real_wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)
    else:
        st.write("No real news headlines available to generate a word cloud.")

with col9:
    st.markdown("<h3 style='color: #2C3E50;'>Word Cloud of Fake News</h3>", unsafe_allow_html=True)
    fake_text = " ".join(df[df['Prediction'] == 'Fake']['Headline'])
    if fake_text.strip():  # Ensure there are words to generate a word cloud
        fake_wordcloud = WordCloud(width=400, height=400, background_color='white').generate(fake_text)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(fake_wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)
    else:
        st.write("No fake news headlines available to generate a word cloud.")

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
