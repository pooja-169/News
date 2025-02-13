# -*- coding: utf-8 -*-
"""fake_or_real_dashboard.py"""

import pandas as pd
import torch
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from wordcloud import WordCloud

# Load the dataset
df = pd.read_csv("webscrap.csv")

# Load the tokenizer and model
model_name = "mrm8488/bert-tiny-finetuned-fake-news-detection"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def predict_fake_news(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    # Perform forward pass with the model
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the logits and convert them to probabilities
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1).squeeze().tolist()

    # Extracting the class with the highest probability
    predicted_class = torch.argmax(logits, dim=1).item()

    labels = ["Real", "Fake"]
    return labels[predicted_class], probabilities

# Create empty lists to hold the results
predictions = []
real_probs = []
fake_probs = []

# Loop through each description and predict
for description in df['Description']:
    prediction, probs = predict_fake_news(description)
    predictions.append(prediction)
    real_probs.append(probs[0])
    fake_probs.append(probs[1])

# Add the results to the DataFrame
df['Prediction'] = predictions
df['Real_Prob'] = real_probs
df['Fake_Prob'] = fake_probs

# Save the DataFrame with predictions to a new CSV file
df.to_csv("fake_or_real.csv", index=False)

# Streamlit dashboard
st.set_page_config(layout="wide", page_title="Fake News Detection Dashboard", page_icon="📰")

# Main Title
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Fake News Detection Dashboard</h1>", unsafe_allow_html=True)

# Filters and Thresholds
st.sidebar.header("Filters")
selected_prediction = st.sidebar.selectbox("Select News Type:", options=["All", "Real", "Fake"], index=0)
threshold = st.sidebar.slider("Probability Threshold:", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

# Filter data based on the user selection
if selected_prediction != "All":
    df_filtered = df[df['Prediction'] == selected_prediction]
else:
    df_filtered = df

df_filtered = df_filtered[(df_filtered['Real_Prob'] >= threshold) | (df_filtered['Fake_Prob'] >= threshold)]

# Check if the filtered DataFrame is empty
if df_filtered.empty:
    st.warning("No data available for the selected filters. Please adjust the filters and try again.")
else:
    # Proceed with visualizations as before

    # Create columns for visualizations
    st.markdown("<h2 style='color: #FF5733;'>Visual Analysis</h2>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.markdown("<h3 style='color: #4CAF50;'>Real vs. Fake News Count</h3>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.countplot(data=df_filtered, x='Prediction', ax=ax, palette='coolwarm')
        ax.set_title("Count of Real vs. Fake News", fontsize=10)
        st.pyplot(fig)

    with col2:
        st.markdown("<h3 style='color: #4CAF50;'>Real vs. Fake News Pie Chart</h3>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(4, 3))
        df_filtered['Prediction'].value_counts().plot.pie(autopct='%1.1f%%', colors=['#1f77b4', '#ff7f0e'], ax=ax, shadow=True)
        ax.set_ylabel('')
        ax.set_title("Distribution of News Types", fontsize=10)
        st.pyplot(fig)

    with col3:
        st.markdown("<h3 style='color: #4CAF50;'>Probability Scatter Plot</h3>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.scatterplot(data=df_filtered, x='Real_Prob', y='Fake_Prob', hue='Prediction', palette='coolwarm', ax=ax)
        ax.set_title("Real vs. Fake Probabilities", fontsize=10)
        st.pyplot(fig)

    # Second row of visualizations
    col4, col5, col6 = st.columns([1, 1, 1])

    with col4:
        st.markdown("<h3 style='color: #4CAF50;'>Real News Word Cloud</h3>", unsafe_allow_html=True)
        real_text = " ".join(df_filtered[df_filtered['Prediction'] == 'Real']['Description'].tolist())
        if real_text.strip():  # Check if there is any text for the word cloud
            wordcloud_real = WordCloud(width=400, height=300, background_color='white', colormap='Blues').generate(real_text)
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.imshow(wordcloud_real, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.warning("No words to display in the Real News Word Cloud.")

    with col5:
        st.markdown("<h3 style='color: #4CAF50;'>Fake News Word Cloud</h3>", unsafe_allow_html=True)
        fake_text = " ".join(df_filtered[df_filtered['Prediction'] == 'Fake']['Description'].tolist())
        if fake_text.strip():  # Check if there is any text for the word cloud
            wordcloud_fake = WordCloud(width=400, height=300, background_color='white', colormap='Reds').generate(fake_text)
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.imshow(wordcloud_fake, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.warning("No words to display in the Fake News Word Cloud.")

    with col6:
        st.markdown("<h3 style='color: #4CAF50;'>Average Probabilities</h3>", unsafe_allow_html=True)
        avg_probs = df_filtered.groupby('Prediction')[['Real_Prob', 'Fake_Prob']].mean()
        if not avg_probs.empty:
            fig, ax = plt.subplots(figsize=(4, 3))
            avg_probs.plot.bar(ax=ax, color=['#2ca02c', '#d62728'])
            ax.set_title("Avg. Real & Fake Probabilities", fontsize=10)
            st.pyplot(fig)
        else:
            st.warning("No data available to calculate average probabilities.")

    # Correlation Heatmap
    st.markdown("<h3 style='color: #4CAF50;'>Correlation Heatmap</h3>", unsafe_allow_html=True)
    
    if len(df_filtered['Real_Prob'].unique()) > 1 and len(df_filtered['Fake_Prob'].unique()) > 1:
        fig, ax = plt.subplots(figsize=(8, 3))
        sns.heatmap(df_filtered[['Real_Prob', 'Fake_Prob']].corr(), annot=True, cmap='coolwarm', ax=ax)
        ax.set_title("Correlation between Real & Fake Probabilities", fontsize=10)
        st.pyplot(fig)
    else:
        st.warning("Not enough variability in data to display a meaningful correlation heatmap.")

# Conclusion section
st.markdown("<h2 style='text-align: center; color: #FF5733;'>Conclusion</h2>", unsafe_allow_html=True)
st.markdown("""
<p style='text-align: center;'>This dashboard provides an in-depth and visually appealing analysis of fake news detection results. 
Explore various visualizations to understand the distribution and characteristics of real and fake news in the dataset.</p>
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
