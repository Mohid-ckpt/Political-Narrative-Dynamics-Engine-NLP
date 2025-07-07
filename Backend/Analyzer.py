import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
# New import for sentiment analysis
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from preprocessing import preprocess_text

def analyze_sentiment(text):
    """
    Analyzes the sentiment of a text using VADER.
    Returns the compound score, which is a single normalized score.
    -1 (most negative) to +1 (most positive).
    """
    analyzer = SentimentIntensityAnalyzer()
    sentiment_score = analyzer.polarity_scores(text)
    return sentiment_score['compound']


if __name__ == '__main__':
    print("Loading and preprocessing data...")
    try:
        df = pd.read_csv('sotu_speeches.csv')
    except FileNotFoundError:
        print("Error: sotu_speeches.csv not found.")
        print("Please run scraper.py first to generate the data.")
        exit()

    print("Preprocessing text... (this may take a moment)")
    df['cleaned_speech'] = df['speech_text'].apply(preprocess_text)

    print("Analyzing sentiment...")
    # Apply sentiment analysis to the ORIGINAL speech text
    df['sentiment_score'] = df['speech_text'].apply(analyze_sentiment)

    # Save the processed data with sentiment scores
    df.to_csv('sotu_speeches_processed.csv', index=False)

    print("Data processing and sentiment analysis complete. Saved to 'sotu_speeches_processed.csv'.")
    print("\nHere's a sample with the new 'sentiment_score' column:")
    print(df[['president', 'date', 'sentiment_score']].head())

    # Quick check: Find the most positive and most negative speeches
    most_positive = df.loc[df['sentiment_score'].idxmax()]
    most_negative = df.loc[df['sentiment_score'].idxmin()]

    print("\nMost Positive Speech:")
    print(
        f"  President: {most_positive['president']}, Date: {most_positive['date']}, Score: {most_positive['sentiment_score']:.2f}")

    print("\nMost Negative Speech:")
    print(
        f"  President: {most_negative['president']}, Date: {most_negative['date']}, Score: {most_negative['sentiment_score']:.2f}")

