import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

print("Loading processed data...")
try:
    df = pd.read_csv('sotu_speeches_processed.csv')
    # Handle any potential missing values in the cleaned text
    df.dropna(subset=['cleaned_speech'], inplace=True)
except FileNotFoundError:
    print("Error: sotu_speeches_processed.csv not found.")
    print("Please run app.py first to generate the processed data.")
    exit()

# We need the cleaned speech text for topic modeling
documents = df['cleaned_speech']

# Step 1: Create the TF-IDF Vectorizer
# TF-IDF stands for Term Frequency-Inverse Document Frequency.
# It gives more weight to words that are frequent in one document but rare across all documents.
print("Creating TF-IDF vectors...")
vectorizer = TfidfVectorizer(max_df=0.9, min_df=5, stop_words='english')
tfidf_vectors = vectorizer.fit_transform(documents)

# Step 2: Create and run the LDA Model
# We'll ask it to find 10 topics. This number is arbitrary and can be tuned.
num_topics = 10
print(f"Running LDA to find {num_topics} topics...")
lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
lda.fit(tfidf_vectors)

# Step 3: Display the topics
print("\n--- Top Words for Each Topic ---")
feature_names = vectorizer.get_feature_names_out()
for topic_idx, topic in enumerate(lda.components_):
    # Get the top 10 words for this topic
    top_words = [feature_names[i] for i in topic.argsort()[:-11:-1]]
    print(f"Topic #{topic_idx + 1}: {', '.join(top_words)}")