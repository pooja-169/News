News Analysis System
Overview
The News Analysis System is designed to analyze sentiment and classify news articles based on their content. This project uses a combination of state-of-the-art NLP models for sentiment analysis and a transformer-based machine translation model for language translation. The system provides insights such as sentiment polarity (positive/negative) and can translate articles from Hindi to English.

Features
Sentiment Analysis: Analyzes the sentiment of news articles, classifying them as positive or negative using pre-trained sentiment classification models.

Multi-Class Sentiment Classification: Classifies sentiment into three categories (positive, negative, neutral) based on the tone of the news article using RoBERTa.

Translation: Translates Hindi news articles into English using a MarianMT-based model.

Web Scraping: Data is scraped from open-source news platforms like Times of India and Google News, ensuring the system is up-to-date with the latest news.

CSV Data: The system uses CSV files for training models, allowing for seamless integration and easy updates to the data.

Technical Details
1. Sentiment Analysis
Model: distilbert-base-uncased-finetuned-sst-2-english

Purpose: Fine-tuned for binary sentiment classification (POSITIVE/NEGATIVE) on the SST-2 dataset.

Flow:

Input a sentence, e.g., "This movie was fantastic!"

Tokenization (lowercase, split into tokens, add special tokens).

Sentiment classification model (DistilBERT with 6 transformer layers).

Output: Sentiment label and confidence score.

2. Multi-Class Sentiment Classification
Model: cardiffnlp/twitter-roberta-base-sentiment-latest

Purpose: Classifies sentiment into three classes (positive, negative, neutral) based on tweets.

Flow:

Tokenization.

RoBERTa transformer-based encoding.

Classification head outputs 3 logits (positive, negative, neutral).

Softmax function to get probabilities for each class.

3. Translation
Model: Helsinki-NLP/opus-mt-hi-en

Purpose: Translates Hindi news articles into English.

Architecture: MarianMT, a transformer-based encoder-decoder model.

Flow:

Tokenization of Hindi sentence.

Encoding through a transformer encoder stack.

Decoding the sentence into English using the decoder stack.

Output the translated sentence.

4. Web Scraping
Source: Open-source news websites like Times of India and Google News.

Purpose: Gather the latest news data to keep the system updated.

Method: Web scraping using Python libraries like BeautifulSoup and Requests to scrape headlines and articles.

5. Training with CSV Data
The models are trained on datasets stored in CSV format, making it easy to handle and update data as needed for further fine-tuning or retraining.

How It Works
Web Scraping:

The system scrapes articles from open-source news sites like Times of India and Google News to gather a dataset of news headlines and articles.

Data Processing:

The scraped data is stored in a CSV file, which is used for training the sentiment analysis models.

Sentiment Analysis:

Once the model is trained, it is used to classify the sentiment of new news articles, returning either a positive or negative classification.

Language Translation:

The system translates news articles written in Hindi to English using the MarianMT model.
