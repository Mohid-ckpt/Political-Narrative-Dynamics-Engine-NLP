import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


def preprocess_text(text):
    """
    Cleans and preprocesses a single piece of text.
    - Lowercase
    - Tokenize
    - Remove stopwords, punctuation, and numbers
    - Lemmatize
    """
    # 1. Lowercase the text
    text = text.lower()

    # 2. Tokenize (split text into words)
    tokens = word_tokenize(text)

    # 3. Setup for cleaning
    stop_words = set(stopwords.words('english'))
    punct = string.punctuation
    lemmatizer = WordNetLemmatizer()

    # 4. Clean tokens
    cleaned_tokens = []
    for token in tokens:
        # Remove stopwords and punctuation
        if token not in stop_words and token not in punct and token.isalpha():
            # Lemmatize the token
            cleaned_tokens.append(lemmatizer.lemmatize(token))

    # 5. Join tokens back into a single string
    return " ".join(cleaned_tokens)

if __name__ == '__main__':

    print("Loading and preprocessing data...")
    # Load the scraped data
    try:
        df = pd.read_csv('sotu_speeches.csv')
    except FileNotFoundError:
        print("Error: sotu_speeches.csv not found.")
        print("Please run scraper.py first to generate the data.")
        exit()

    # Apply the preprocessing function to the 'speech_text' column
    # This might take a minute or two
    df['cleaned_speech'] = df['speech_text'].apply(preprocess_text)

    # Save the processed data to a new CSV to avoid re-running this step
    df.to_csv('sotu_speeches_processed.csv', index=False)

    print("Data preprocessing complete. Saved to 'sotu_speeches_processed.csv'.")
    print("\nHere's a sample with the new 'cleaned_speech' column:")
    print(df[['president', 'date', 'cleaned_speech']].head())