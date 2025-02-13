import requests
from bs4 import BeautifulSoup
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import IsolationForest
import streamlit as st
import tweepy

# Twitter API setup (replace with your own API keys)
TWITTER_API_KEY = 'your_new_api_key'
TWITTER_API_SECRET = 'your_new_api_secret'
TWITTER_ACCESS_TOKEN = 'your_new_access_token'
TWITTER_ACCESS_SECRET = 'your_new_access_secret'

auth = tweepy.OAuthHandler(TWITTER_API_KEY, TWITTER_API_SECRET)
auth.set_access_token(TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET)
api = tweepy.API(auth)

# Step 1: Data Collection

# Reddit Scraping
def scrape_reddit(subreddit):
    url = f"https://www.reddit.com/r/{subreddit}/"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    posts = []
    for post in soup.find_all('h3'):
        posts.append(post.text)
    
    return posts

reddit_posts = scrape_reddit('technology')

# Twitter Scraping
def scrape_twitter(query, count=100):
    tweets = api.search_tweets(q=query, count=count, lang='en', tweet_mode='extended')
    posts = []
    for tweet in tweets:
        posts.append(tweet.full_text)
    
    return posts

twitter_posts = scrape_twitter('technology')

# Combine Data from Multiple Sources
combined_posts = reddit_posts + twitter_posts

# Step 2: Sentiment and Intensity Analysis

def analyze_sentiment(posts):
    analyzer = SentimentIntensityAnalyzer()
    results = []
    for post in posts:
        sentiment = analyzer.polarity_scores(post)
        results.append(sentiment['compound'])
    return results

sentiments = analyze_sentiment(combined_posts)

# Combine the data into a DataFrame

data = pd.DataFrame({
    'post': combined_posts,
    'sentiment_score': sentiments
})

# Step 3: Pattern Recognition

def detect_anomalies(data):
    model = IsolationForest(contamination=0.1)
    data['anomaly'] = model.fit_predict(data[['sentiment_score']])
    return data

data = detect_anomalies(data)

# Step 4: Dashboard Creation using Streamlit

st.title("Social Media and News Analytics Dashboard")

# Display raw data
st.subheader("Raw Data")
st.write(data)

# Sentiment Analysis Over Time (assuming posts have timestamps)
# For simplicity, here we'll assume an index as date
data['date'] = pd.date_range(start='2024-08-01', periods=len(data), freq='H')
sentiment_over_time = data.groupby('date').mean()['sentiment_score']

st.subheader("Sentiment Analysis Over Time")
st.line_chart(sentiment_over_time)

# Pattern Recognition - Anomalies
st.subheader("Pattern Recognition - Anomalies")
anomalies = data[data['anomaly'] == -1]
st.write(anomalies)

# Additional Feature: Filter posts by sentiment
st.subheader("Filter Posts by Sentiment")
sentiment_filter = st.slider("Select Sentiment Range", min_value=-1.0, max_value=1.0, value=(-0.5, 0.5))
filtered_posts = data[(data['sentiment_score'] >= sentiment_filter[0]) & (data['sentiment_score'] <= sentiment_filter[1])]
st.write(filtered_posts)

