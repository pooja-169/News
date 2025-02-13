# -*- coding: utf-8 -*-
"""streamlit_sentiment_dashboard.py"""

import pandas as pd
import torch
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn.functional as F
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np
import os

# Load sentiment analysis model and tokenizer
@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer, sentiment_model = load_model_and_tokenizer(model_name)

# Function for sentiment analysis
def analyze_sentiment(text):
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = sentiment_model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)

        pos_score = probs[0][2].item()  # Assuming index 2 is POSITIVE
        neu_score = probs[0][1].item()  # Assuming index 1 is NEUTRAL
        neg_score = probs[0][0].item()  # Assuming index 0 is NEGATIVE

        return pos_score, neu_score, neg_score
    except Exception as e:
        st.error(f"Error in sentiment analysis: {str(e)}")
        return 0.0, 0.0, 0.0

# Function for calculating intensity
def calculate_intensity(pos_score, neu_score, neg_score):
    intensity = pos_score - neg_score  # Intensity between -1 and 1

    # Normalize intensity to be between -1 and 1
    if intensity > 0:
        intensity = min(intensity, 1)
    else:
        intensity = max(intensity, -1)

    return round(intensity, 2)

# Load the dataset
@st.cache_data
def load_data(file_path):
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        st.error("Dataset not found!")
        return pd.DataFrame()

df = load_data('webscrap_final.csv')
texts = df['Description'].astype(str).fillna('')

# Perform sentiment analysis and calculate intensity
def analyze_text(text):
    pos_score, neu_score, neg_score = analyze_sentiment(text)
    intensity = calculate_intensity(pos_score, neu_score, neg_score)
    return pos_score, neu_score, neg_score, intensity

# Update results dynamically
@st.cache_data(show_spinner=False)
def process_data(texts):
    results = texts.apply(lambda x: pd.Series(analyze_text(x)))
    df[['positive', 'neutral', 'negative', 'intensity']] = results

    # Clustering using DBSCAN
    X = df[['positive', 'neutral', 'negative', 'intensity']].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    db = DBSCAN(eps=0.3, min_samples=5).fit(X_scaled)
    labels = db.labels_
    df['cluster'] = labels

    return df, X_scaled, labels

df, X_scaled, labels = process_data(texts)

# Streamlit Dashboard
st.set_page_config(layout="wide", page_title="Sentiment Analysis & Clustering Dashboard", page_icon="📊")

# Main Title
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Sentiment Analysis & Clustering Dashboard</h1>", unsafe_allow_html=True)

# File uploader
st.sidebar.markdown("### Upload a CSV file")
uploaded_file = st.sidebar.file_uploader("Choose a file", type=["csv"])

if uploaded_file:
    df = load_data(uploaded_file)
    texts = df['Description'].astype(str).fillna('')
    df, X_scaled, labels = process_data(texts)

# Visual Analysis Section
st.markdown("<h2 style='color: #FF5733;'>Visual Analysis</h2>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("<h3 style='color: #4CAF50;'>Sentiment Distribution</h3>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(4, 3))
    df[['positive', 'neutral', 'negative']].plot(kind='bar', stacked=True, ax=ax)
    ax.set_title("Stacked Sentiment Distribution", fontsize=10)
    st.pyplot(fig)

with col2:
    st.markdown("<h3 style='color: #4CAF50;'>Intensity Distribution</h3>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.histplot(df['intensity'], bins=20, kde=True, ax=ax)
    ax.set_title("Intensity Distribution", fontsize=10)
    st.pyplot(fig)

with col3:
    st.markdown("<h3 style='color: #4CAF50;'>Cluster Distribution</h3>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.countplot(data=df, x='cluster', palette='coolwarm', ax=ax)
    ax.set_title("Cluster Distribution", fontsize=10)
    st.pyplot(fig)

# Cluster Visualization Section
st.markdown("<h2 style='color: #FF5733;'>Cluster Visualization</h2>", unsafe_allow_html=True)
st.markdown("<h3 style='color: #4CAF50;'>DBSCAN Clusters of Sentiment and Intensity</h3>", unsafe_allow_html=True)

unique_labels = np.unique(df['cluster'].values)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

fig, ax = plt.subplots(figsize=(10, 6))
for label, color in zip(unique_labels, colors):
    class_member_mask = (df['cluster'].values == label)
    xy = X_scaled[class_member_mask]
    ax.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=color, markeredgecolor='k', markersize=6)

ax.set_title('DBSCAN Clusters of Sentiment and Intensity')
ax.set_xlabel('Sentiment Score')
ax.set_ylabel('Intensity')
st.pyplot(fig)

# Conclusion section
st.markdown("<h2 style='text-align: center; color: #FF5733;'>Conclusion</h2>", unsafe_allow_html=True)
st.markdown("""
<p style='text-align: center;'>This dashboard provides an in-depth analysis of sentiment and intensity clustering. 
Explore the visualizations to understand how sentiment is distributed and how patterns emerge in the clustering analysis.</p>
""", unsafe_allow_html=True)

# Add some final styling
st.markdown("""
<style>
    .stApp {
        background-color: #20232a;
        color: white;
    }
    h1, h2, h3 {
        font-family: 'Arial', sans-serif;
    }
    .css-18e3th9 {
        padding: 5px 2px;
    }
    .css-1d391kg {
        background-color: #1e2125;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
    }
    .css-qbe2hs {
        background-color: #0e1117;
        padding: 8px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)
