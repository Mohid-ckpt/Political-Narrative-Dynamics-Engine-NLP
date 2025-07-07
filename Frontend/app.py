# app.py (Final Streamlit App version with Topic Modeling)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# --- Page Configuration ---
st.set_page_config(
    page_title="Political Rhetoric Analyzer",
    page_icon="🏛️",
    layout="wide"
)


# --- Caching Data Loading ---
@st.cache_data
def load_data():
    """Loads the processed SOTU speeches data."""
    try:
        df = pd.read_csv('C:\\Python Projects\\Political Narrative Dynamics Engine\\Backend\\sotu_speeches_processed.csv')
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        return df
    except FileNotFoundError:
        st.error("Error: `sotu_speeches_processed.csv` not found.")
        st.info("Please make sure the file is in the same directory as app.py, or update the path.")
        return None


# --- Highlighting Function---
def highlight_keyword(text, keyword):
    if not keyword.strip():
        return text
    escaped_keyword = re.escape(keyword)
    highlighted_text = re.sub(
        f'({escaped_keyword})', r'<mark>\1</mark>', text, flags=re.IGNORECASE
    )
    return highlighted_text


#Function to perform Topic Modeling
@st.cache_data
def perform_topic_modeling(data, num_topics, words_per_topic):
    """Performs LDA Topic Modeling and returns the topics."""
    documents = data['cleaned_speech'].dropna()

    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer(max_df=0.9, min_df=5, stop_words='english')
    tfidf_vectors = vectorizer.fit_transform(documents)

    # Run LDA
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(tfidf_vectors)

    # Get the topic words
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-words_per_topic - 1:-1]]
        topics.append(f"**Topic {topic_idx + 1}:** {', '.join(top_words)}")
    return topics


# --- Main App ---
st.title("🏛️ The Political Rhetoric Analyzer")
st.markdown(
    "Analyze sentiment, keyword usage, and underlying topics in US Presidential State of the Union addresses."
)

df = load_data()

if df is not None:
    # --- Sidebar for User Inputs (no changes) ---
    st.sidebar.header("Analyzer Controls")
    president_list = sorted(df['president'].unique())
    selected_president = st.sidebar.selectbox("Select a President:", options=president_list)
    keyword = st.sidebar.text_input("Enter a keyword to track:", value="economy").lower()

    # --- Create Tabs for Different Analyses ---
    tab1, tab2 = st.tabs(["Presidential Analysis", "Overall Topic Modeling"])

    # --- TAB 1: PRESIDENTIAL ANALYSIS (Existing Code) ---
    with tab1:
        st.header(f"Analysis for {selected_president}")

        # Filter data and calculate keyword frequency
        president_df = df[df['president'] == selected_president].copy()


        def count_keyword(text, key):
            return text.lower().count(key) if isinstance(text, str) else 0


        president_df['keyword_freq'] = president_df['speech_text'].apply(lambda text: count_keyword(text, keyword))

        if president_df.empty:
            st.warning(f"No data available for {selected_president}.")
        else:
            col1, col2 = st.columns(2)
            # Plot 1: Sentiment Over Time
            with col1:
                st.subheader("Sentiment Over Time")
                fig1, ax1 = plt.subplots(figsize=(10, 6))
                sns.lineplot(data=president_df, x='year', y='sentiment_score', ax=ax1, marker='o', color='royalblue')
                ax1.set_title(f"Sentiment of SOTU Speeches by {selected_president}")
                ax1.set_xlabel("Year")
                ax1.set_ylabel("Sentiment Score (VADER Compound)")
                ax1.grid(True, linestyle='--', alpha=0.6)
                plt.xticks(rotation=45)
                st.pyplot(fig1)

            # Plot 2: Keyword Frequency Over Time
            with col2:
                st.subheader(f"Frequency of '{keyword}' Over Time")
                if president_df['keyword_freq'].sum() == 0:
                    st.warning(f"The keyword '{keyword}' was not found in any speeches by {selected_president}.")
                else:
                    fig2, ax2 = plt.subplots(figsize=(10, 6))
                    sns.lineplot(data=president_df, x='year', y='keyword_freq', ax=ax2, marker='o', color='firebrick')
                    ax2.set_title(f"Usage of '{keyword}' by {selected_president}")
                    ax2.set_xlabel("Year")
                    ax2.set_ylabel("Keyword Count")
                    ax2.grid(True, linestyle='--', alpha=0.6)
                    plt.xticks(rotation=45)
                    st.pyplot(fig2)

            # Display Raw Data and Individual Speeches
            with st.expander("View Speech Data and Read Individual Speeches"):
                st.dataframe(president_df[['year', 'sentiment_score', 'keyword_freq', 'url']])
                st.subheader("Read a Specific Speech")
                speech_options = president_df['date'].dt.strftime('%B %d, %Y').tolist()
                selected_speech_str = st.selectbox("Choose a speech to display its text:", options=speech_options,
                                                   key="speech_select")
                if selected_speech_str:
                    selected_date = pd.to_datetime(selected_speech_str)
                    selected_speech_row = president_df[president_df['date'] == selected_date].iloc[0]
                    speech_text = selected_speech_row['speech_text']
                    highlighted_text = highlight_keyword(speech_text, keyword)
                    st.markdown(f"**Displaying speech from: {selected_speech_row['date'].strftime('%B %d, %Y')}**")
                    st.markdown(
                        f'<div style="border: 1px solid #e6e6e6; border-radius: 5px; padding: 10px; height: 300px; overflow-y: scroll;">{highlighted_text}</div>',
                        unsafe_allow_html=True
                    )

    # --- TAB 2: TOPIC MODELING ---
    with tab2:
        st.header("Discovering Core Themes in SOTU Addresses")
        st.markdown("""
        Topic Modeling is an unsupervised machine learning technique that scans a set of documents,
        detects word and phrase patterns within them, and automatically clusters word groups that
        best characterize a set of documents. Here, we analyze all speeches to find the main recurring themes.
        """)

        # User controls for topic modeling
        st.subheader("Topic Model Controls")
        num_topics = st.slider("Select Number of Topics to Discover:", min_value=3, max_value=20, value=10, step=1)
        words_per_topic = st.slider("Select Number of Words per Topic:", min_value=5, max_value=15, value=10, step=1)

        # Button to run the analysis
        if st.button("Run Topic Analysis", key="run_topics"):
            with st.spinner("Analyzing all speeches... This may take a moment on the first run."):
                # Call our cached function
                topics = perform_topic_modeling(df, num_topics, words_per_topic)

                st.subheader("Discovered Topics")
                for topic in topics:
                    st.markdown(f"* {topic}")