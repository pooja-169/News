**News Analysis System**

**Overview**
The News Analysis System is designed to analyze sentiment and classify news articles based on their content. This project integrates multiple state-of-the-art NLP models for sentiment analysis and a transformer-based machine translation model for language translation. The system provides real-time insights such as sentiment polarity (positive/negative) and can translate articles from Hindi to English.


**Features**
1. Sentiment Analysis
   Analyzes the sentiment of news articles, classifying them as positive or negative using pre-trained sentiment classification models.
   
2. Multi-Class Sentiment Classification
   Classifies sentiment into three categories (positive, negative, neutral) based on the tone of the news article using the RoBERTa model.
   
3. Translation
   Translates Hindi news articles into English using a MarianMT-based model.

4. Web Scraping
   Scrapes data from open-source news platforms like Times of India and Google News to ensure the system is always updated with the latest news articles.
   
5. CSV Data for Training
   The system uses CSV files for model training, enabling easy updates and seamless integration of new data for retraining or fine-tuning.
Technical Details
**1. Sentiment Analysis**
- Model: distilbert-base-uncased-finetuned-sst-2-english
   - Purpose: Fine-tuned for binary sentiment classification (POSITIVE/NEGATIVE) on the SST-2 dataset.
   
   Flow:
   1. Input: A sentence, e.g., "This movie was fantastic!"
   2. Tokenization: The text is lowercased, split into tokens, and special tokens like [CLS] and [SEP] are added.
   3. Model: The DistilBERT model processes the sentence with 6 transformer layers.
   4. Output: The sentiment classification is returned with a confidence score.
      
**2. Multi-Class Sentiment Classification**
- Model: cardiffnlp/twitter-roberta-base-sentiment-latest
   - Purpose: Classifies sentiment into three categories (positive, negative, neutral) based on tweets.
   
   Flow:
   1. Tokenization: Input text is tokenized.
   2. Encoding: The RoBERTa transformer model processes the tokens.
   3. Classification: A classification head outputs 3 logits (positive, negative, neutral).
   4. Softmax: Probabilities for each sentiment class are calculated.
      
**3. Translation**
- Model: Helsinki-NLP/opus-mt-hi-en
   - Purpose: Translates Hindi news articles into English.
   - Architecture: MarianMT (transformer-based encoder-decoder model).
   
   Flow:
   1. Input: Tokenization of a Hindi sentence.
   2. Encoding: The sentence is passed through 6 stacked transformer encoder layers.
   3. Decoding: The sentence is decoded into English using the decoder stack.
   4. Output: Translated English sentence.
      
**4. Web Scraping**
- Source: Open-source news websites like Times of India and Google News.
   - Purpose: Scrape the latest news data to keep the system updated.
   - Method: Web scraping is implemented using BeautifulSoup and Requests libraries to collect news headlines and articles.
     
**5. Training with CSV Data**
- Training: The models are trained on datasets stored in CSV format. This ensures easy integration and updates to data for further model fine-tuning or retraining.
  
**How It Works**
1. Web Scraping:
   - The system scrapes articles from open-source news sites like Times of India and Google News to gather a dataset of headlines and articles.
   
2. Data Processing:
   - The scraped data is stored in CSV format, which is then used for training the sentiment analysis models.
   
3. Sentiment Analysis:
   - The trained model classifies the sentiment of newly scraped news articles, determining if they are positive or negative.
   
4. Language Translation:
   - For Hindi news articles, the system translates them into English using the MarianMT model, providing a more accessible version of the content.
     
**Conclusion**
The News Analysis System is a powerful tool that combines sentiment analysis, language translation, and real-time news scraping. By leveraging advanced transformer models, it provides an insightful, multi-functional system for analyzing news articles in multiple languages.
