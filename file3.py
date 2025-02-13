# -*- coding: utf-8 -*-
"""streamlit_fake_real_dashboard.py"""

import pandas as pd
import torch
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.cluster import DBSCAN
import numpy as np

# Load the dataset
df = pd.read_csv("/content/drive/MyDrive/sentimental_analysis/webscrap.csv")

# Load the first model and tokenizer (e.g., BERT)
bert_model_name = "bert-base-uncased"
bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
bert_model = AutoModelForSequenceClassification.from_pretrained(bert_model_name)

# Load the second model and tokenizer (e.g., RoBERTa)
roberta_model_name = "roberta-base"
roberta_tokenizer = AutoTokenizer.from_pretrained(roberta_model_name)
roberta_model = AutoModelForSequenceClassification.from_pretrained(roberta_model_name)

# Function to predict fake news using ensemble of BERT and RoBERTa
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

# Create empty lists to hold the results
predictions = []
real_probs = []
fake_probs = []

# Loop through each description and predict using the ensemble model
for description in df['Description']:
    prediction, probs = predict_fake_news(description)
    predictions.append(prediction)
    real_probs.append(probs[0])
    fake_probs.append(probs[1])

# Add the results to the DataFrame
df['Prediction'] = predictions
df['Real_Prob'] = real_probs
df['Fake_Prob'] = fake_probs

# Perform DBSCAN clustering
features = np.array(list(zip(df['Real_Prob'], df['Fake_Prob'])))
dbscan = DBSCAN(eps=0.1, min_samples=5)
df['Cluster'] = dbscan.fit_predict(features)

# Save the final results with clusters
df.to_csv("/content/drive/MyDrive/sentimental_analysis/fake_or_real_merge_two_model_with_clusters.csv", index=False)

# Streamlit Dashboard
st.set_page_config(layout="wide", page_title="Fake News Detection & Clustering Dashboard", page_icon="📰")

# Main Title
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Fake News Detection & Clustering Dashboard</h1>", unsafe_allow_html=True)

# Display the DataFrame
st.markdown("<h2 style='color: #FF5733;'>Dataset Overview</h2>", unsafe_allow_html=True)
st.dataframe(df.head())

# Visual Analysis Section
st.markdown("<h2 style='color: #FF5733;'>Visual Analysis</h2>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("<h3 style='color: #4CAF50;'>Prediction Distribution</h3>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.countplot(data=df, x='Prediction', ax=ax, palette='coolwarm')
    ax.set_title("Real vs. Fake News Distribution", fontsize=10)
    st.pyplot(fig)

with col2:
    st.markdown("<h3 style='color: #4CAF50;'>Probability Distributions</h3>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.histplot(df['Real_Prob'], bins=20, kde=True, color='green', label='Real Prob', ax=ax)
    sns.histplot(df['Fake_Prob'], bins=20, kde=True, color='red', label='Fake Prob', ax=ax)
    ax.set_title("Real vs. Fake Probability Distribution", fontsize=10)
    ax.legend()
    st.pyplot(fig)

with col3:
    st.markdown("<h3 style='color: #4CAF50;'>Cluster Distribution</h3>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.countplot(data=df, x='Cluster', palette='Set1', ax=ax)
    ax.set_title("Cluster Distribution", fontsize=10)
    st.pyplot(fig)

# Cluster Visualization Section
st.markdown("<h2 style='color: #FF5733;'>Cluster Visualization</h2>", unsafe_allow_html=True)
st.markdown("<h3 style='color: #4CAF50;'>DBSCAN Clustering of News Articles</h3>", unsafe_allow_html=True)

fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(
    x='Real_Prob',
    y='Fake_Prob',
    hue='Cluster',
    palette='Set1',
    data=df,
    legend='full',
    s=100,  # Size of points
    ax=ax
)
ax.set_title('DBSCAN Clustering of News Articles', fontsize=16)
ax.set_xlabel('Real Probability', fontsize=14)
ax.set_ylabel('Fake Probability', fontsize=14)
st.pyplot(fig)

# Conclusion section
st.markdown("<h2 style='text-align: center; color: #FF5733;'>Conclusion</h2>", unsafe_allow_html=True)
st.markdown("""
<p style='text-align: center;'>This dashboard provides an in-depth analysis of fake news detection using an ensemble model 
and clustering analysis using DBSCAN. Explore the visualizations to understand how predictions and patterns emerge in the news articles.</p>
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
