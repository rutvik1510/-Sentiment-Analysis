import matplotlib
matplotlib.use('Agg') # Set the Matplotlib backend to 'Agg' for headless operation

from flask import Flask, request, jsonify
from flask_cors import CORS
from bs4 import BeautifulSoup
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt # Import matplotlib.pyplot after setting the backend
import base64
from io import BytesIO
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import time
import warnings
import pickle
import spacy # Import spacy for aspect extraction, though the provided code uses a keyword-based approach
from datetime import datetime # Import datetime for potential trend analysis, though not fully implemented in the provided snippet
from collections import Counter # Import Counter for aspect extraction

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Load the saved model and vectorizer
# IMPORTANT: These paths now assume .pkl files are directly inside the flask_app directory
with open("/Users/rutvik/Desktop/nlp/flask_app/logistic_regression_model.pkl", 'rb') as model_file:
    loaded_model = pickle.load(model_file)

with open("/Users/rutvik/Desktop/nlp/flask_app/tfidf_vectorizer.pkl", 'rb') as vectorizer_file:
    loaded_vectorizer = pickle.load(vectorizer_file)

# Set up Selenium WebDriver (Ensure to set the path to your chromedriver)
# IMPORTANT: This path now assumes chromedriver is directly inside the flask_app directory
chrome_driver_path = "/Users/rutvik/Desktop/nlp/flask_app/chromedriver" 

service = Service(chrome_driver_path)
driver = webdriver.Chrome(service=service)

# Headers to mimic browser request
headers = {
    'authority': 'www.amazon.com',
    'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,/;q=0.8,application/signed-exchange;v=b3;q=0.9',
    'accept-language': 'en-US,en;q=0.9,bn;q=0.8',
    'sec-ch-ua': '" Not A;Brand";v="99", "Chromium";v="102", "Google Chrome";v="102"',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36'
}

# Function to extract product name
def get_product_name(url):
    try:
        driver.get(url)
        time.sleep(3) # Give time for the page to load

        # Check for CAPTCHA
        # The original code's CAPTCHA handling is manual. For automated scraping,
        # you'd need a more robust CAPTCHA solution (e.g., using a CAPTCHA solving service).
        # For simplicity in this example, we assume manual intervention if CAPTCHA appears.
        try:
            captcha_element = driver.find_element(By.ID, 'captchacharacters')
            print("CAPTCHA detected. Please solve it manually in the Chrome window that opens.")
            # You might want to add a longer sleep or a user prompt here
            # for the user to solve the CAPTCHA in the opened browser window.
            time.sleep(10) # Give user time to solve CAPTCHA
            # After manual CAPTCHA solving, the page should reload or navigate to the product page.
            # We assume the driver is now on the correct page.
        except:
            pass # No CAPTCHA element found, proceed

        soup = BeautifulSoup(driver.page_source, 'html.parser')
        product_title_element = soup.select_one('span#productTitle')
        if product_title_element:
            product_name = product_title_element.get_text(strip=True)
        else:
            product_name = "Product Name Not Found"
        return product_name
    except Exception as e:
        return f"Error getting product name: {str(e)}"

# Function to analyze sentiment with VADER
def analyze_sentiment_vader(reviews):
    if not reviews:
        return 0, 0, 0 # Return percentages as 0 if no reviews

    analyzer = SentimentIntensityAnalyzer()
    positive_count = 0
    negative_count = 0
    neutral_count = 0

    for review in reviews:
        score = analyzer.polarity_scores(review)['compound']
        if score > 0.05:
            positive_count += 1
        elif score < -0.05:
            negative_count += 1
        else:
            neutral_count += 1

    total_reviews = len(reviews)
    positive_percentage = (positive_count / total_reviews) * 100
    negative_percentage = (negative_count / total_reviews) * 100
    neutral_percentage = (neutral_count / total_reviews) * 100

    return positive_percentage, negative_percentage, neutral_percentage

# Preprocessing text for ML model
# Download stopwords if not already done (though done in initial setup, good to ensure)
try:
    stp_words = stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
    stp_words = stopwords.words('english')

def preprocess_text(text_data):
    preprocessed_text = []
    for sentence in text_data:
        # Remove non-alphanumeric characters but keep spaces
        sentence = re.sub(r'[^\w\s]', '', sentence)
        # Tokenize, convert to lower, and remove stopwords
        preprocessed_text.append(' '.join(token.lower() for token in nltk.word_tokenize(sentence) if token.lower() not in stp_words))
    return preprocessed_text

# Function to analyze sentiment using the loaded ML model
def analyze_sentiment_ml(reviews):
    if not reviews:
        return 0, 0, 0

    reviews_preprocessed = preprocess_text(reviews)
    # Ensure loaded_vectorizer is fitted with similar data as during training
    X_reviews = loaded_vectorizer.transform(reviews_preprocessed) # .toarray() might not be strictly necessary, depends on model
    predictions = loaded_model.predict(X_reviews)

    positive_count = sum(predictions == 1) # Assuming 1 is positive, 0 is negative
    negative_count = sum(predictions == 0)

    total_predictions = len(predictions)
    positive_percentage = (positive_count / total_predictions) * 100
    negative_percentage = (negative_count / total_predictions) * 100
    neutral_percentage = 0  # ML model is binary (positive/negative), so neutral is not directly applicable from its output

    return positive_percentage, negative_percentage, neutral_percentage

# Function to generate a word cloud
def generate_word_cloud(text):
    if not text:
        return "" # Return empty string if no text

    wordcloud = WordCloud(width=800, height=400, background_color='black', random_state=21, max_font_size=110).generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')

    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close() # Close the plot to prevent it from displaying in a GUI or consuming memory
    return img_str

# Aspect-Based Sentiment Analysis (Simplified Keyword-based as per original report structure)
# Note: A more sophisticated approach would involve NLP parsing (e.g., spaCy dependency parsing)
# to link aspects to sentiment words, and perhaps training a custom aspect extractor.
def extract_aspects(reviews, top_n=5):
    aspect_keywords = {
        'performance': ['performance', 'speed', 'efficiency', 'effectiveness', 'functionality', 'capability', 'power', 'responsiveness', 'reliability', 'stability'],
        'quality': ['quality', 'durability', 'build', 'construction', 'materials', 'craftsmanship', 'robustness', 'sturdiness', 'resilience', 'longevity'],
        'design': ['design', 'aesthetics', 'style', 'appearance', 'look', 'feel', 'ergonomics', 'comfort', 'form factor', 'layout'],
        'usability': ['usability', 'ease of use', 'user-friendly', 'intuitive', 'simplicity', 'convenience', 'accessibility', 'learnability', 'straightforward', 'effortless'],
        'features': ['features', 'functions', 'capabilities', 'options', 'specifications', 'attributes', 'properties', 'characteristics', 'elements', 'aspects'],
        'value': ['value', 'price', 'cost', 'affordability', 'worth', 'expensive', 'cheap', 'budget-friendly', 'investment', 'economical'],
        'service': ['service', 'support', 'customer service', 'warranty', 'returns', 'assistance', 'help', 'responsiveness', 'communication', 'after-sales'],
        'shipping': ['shipping', 'delivery', 'packaging', 'transport', 'arrival', 'postage', 'freight', 'courier', 'dispatch', 'handling'],
        'size & fit': ['size', 'dimensions', 'weight', 'fit', 'proportions', 'measurements', 'compact', 'large', 'portable', 'bulky'],
        'compatibility': ['compatibility', 'compatible', 'integration', 'connectivity', 'interface', 'adaptable', 'synchronization', 'link', 'coordination', 'matching']
    }

    aspect_mentions = Counter()
    for review in reviews:
        lower_review = review.lower()
        for aspect, keywords in aspect_keywords.items():
            if any(keyword in lower_review for keyword in keywords):
                aspect_mentions[aspect] += 1
    # Sort aspects by frequency and return top_n
    return [aspect for aspect, count in aspect_mentions.most_common(top_n)]

def analyze_aspect_sentiment(reviews, aspects):
    analyzer = SentimentIntensityAnalyzer()
    aspect_sentiments = {}

    for aspect in aspects:
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        relevant_reviews = []

        # Find reviews mentioning the aspect (case-insensitive)
        for review in reviews:
            if aspect.lower() in review.lower(): # Simple check, can be improved
                relevant_reviews.append(review)

        if relevant_reviews:
            for review_text in relevant_reviews:
                score = analyzer.polarity_scores(review_text)['compound']
                if score > 0.05:
                    positive_count += 1
                elif score < -0.05:
                    negative_count += 1
                else:
                    neutral_count += 1

            # Store counts (you might want percentages here too)
            aspect_sentiments[aspect] = {
                'positive': positive_count,
                'negative': negative_count,
                'neutral': neutral_count
            }
    return aspect_sentiments

# Function to get word frequency for positive/negative reviews
def get_word_frequency(reviews, sentiment_scores, sentiment_type):
    # This function would need review_dates to match scores to dates
    # For now, it will apply to all reviews filtered by sentiment_type
    words = []
    analyzer = SentimentIntensityAnalyzer()
    for i, review in enumerate(reviews):
        score = analyzer.polarity_scores(review)['compound']
        if sentiment_type == 'positive' and score > 0.05:
            words.extend(nltk.word_tokenize(re.sub(r'[^\w\s]', '', review.lower())))
        elif sentiment_type == 'negative' and score < -0.05:
            words.extend(nltk.word_tokenize(re.sub(r'[^\w\s]', '', review.lower())))

    # Filter out stopwords and common non-informative words
    filtered_words = [word for word in words if word not in stp_words and len(word) > 2]
    return Counter(filtered_words).most_common(10) # Top 10 most common words

# Function to group reviews by month for trend analysis (placeholder logic)
def group_reviews_by_month(reviews, review_dates):
    # This is a placeholder. You'd need actual review dates from scraping.
    # For demonstration, we'll create some dummy data.
    trend_data = []
    # Assuming review_dates are available and are datetime objects
    # Example: { "2024-04": 0.3, "2024-05": 0.5, ... } (average sentiment per month)
    # The original report's graph shows dates from April 2024 to May 2025.
    # To generate this, you'd need the actual review dates and their sentiments.
    # For a simple mock, let's create a few data points:
    months = ["April 2024", "September 2024", "November 2024", "December 2024",
              "January 2025", "February 2025", "March 2025", "April 2025", "May 2025"]
    dummy_sentiments = [0.35, 0.7, -0.15, -0.85, -0.35, -0.25, 0.0, 0.55, 0.1] # Sample average sentiments

    for i, month in enumerate(months):
        trend_data.append({'month': month, 'average_sentiment': dummy_sentiments[i]})
    return trend_data


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        url = data.get('url')

        if not url:
            return jsonify({'error': 'URL is required'}), 400

        # Attempt to get product name and reviews
        product_name = get_product_name(url)

        # Check if CAPTCHA was truly solved (though manual for now)
        # If the driver is still on a CAPTCHA page or product name extraction failed due to it
        if "Error getting product name" in product_name: # Simple error check
             return jsonify({'error': 'Failed to get product name. CAPTCHA might need manual solving or page structure changed.'}), 400

        # After getting the product name, re-soup the page to get reviews
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        amazon_reviews = [review.get_text(strip=True) for review in soup.select('span[data-hook="review-body"]')]

        # Placeholder for review dates (you would scrape these too)
        # For now, create dummy dates for trend analysis to work with mock data
        review_dates = [datetime.now()] * len(amazon_reviews)


        if not amazon_reviews:
            return jsonify({
                'product_name': product_name,
                'positive_percentage': 0,
                'negative_percentage': 0,
                'neutral_percentage': 0,
                'word_cloud': '',
                'aspect_sentiments': {},
                'trend_data': [],
                'word_frequency': {'positive': [], 'negative': []}
            })

        # Perform sentiment analysis using VADER
        vader_positive_percentage, vader_negative_percentage, vader_neutral_percentage = analyze_sentiment_vader(amazon_reviews)

        # Determine if ML model is needed (if VADER's neutral percentage is high)
        if vader_neutral_percentage > 40: # Threshold for using ML model
            ml_positive_percentage, ml_negative_percentage, _ = analyze_sentiment_ml(amazon_reviews)
            # Combine or prefer ML results
            positive_percentage = ml_positive_percentage
            negative_percentage = ml_negative_percentage
            neutral_percentage = 0 # ML is binary, so explicit neutral is not directly from it
        else:
            positive_percentage = vader_positive_percentage
            negative_percentage = vader_negative_percentage
            neutral_percentage = vader_neutral_percentage

        # Generate word cloud
        word_cloud_text = ' '.join(amazon_reviews)
        word_cloud_image = generate_word_cloud(word_cloud_text)

        # Perform aspect-based sentiment analysis
        # First extract aspects
        aspects = extract_aspects(amazon_reviews, top_n=5)
        # Then analyze sentiment for those aspects
        aspect_sentiments_results = analyze_aspect_sentiment(amazon_reviews, aspects)

        # Generate trend data (using dummy data for now, actual dates needed)
        trend_data = group_reviews_by_month(amazon_reviews, review_dates) # Pass review_dates if scraped

        # Get word frequency for positive and negative reviews
        # This requires sentiment scores for individual reviews, which VADER provides.
        # We need the individual review sentiment scores to filter for positive/negative words.
        # Re-calculate or pass scores from analyze_sentiment_vader if needed.
        analyzer = SentimentIntensityAnalyzer()
        individual_sentiment_scores = [analyzer.polarity_scores(review)['compound'] for review in amazon_reviews]

        positive_word_freq = get_word_frequency(amazon_reviews, individual_sentiment_scores, 'positive')
        negative_word_freq = get_word_frequency(amazon_reviews, individual_sentiment_scores, 'negative')

        # Prepare response data
        response_data = {
            'product_name': product_name,
            'positive_percentage': round(positive_percentage, 2),
            'negative_percentage': round(negative_percentage, 2),
            'neutral_percentage': round(neutral_percentage, 2),
            'word_cloud': word_cloud_image,
            'aspect_sentiments': aspect_sentiments_results,
            'trend_data': trend_data, # Include trend data
            'word_frequency': { # Include word frequency
                'positive': positive_word_freq,
                'negative': negative_word_freq
            }
        }

        return jsonify(response_data)
    except Exception as e:
        # Log the error for debugging
        print(f"An error occurred: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # It's recommended to run in production without debug=True for security reasons
    app.run(debug=True)
