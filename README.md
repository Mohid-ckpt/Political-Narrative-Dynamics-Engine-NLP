# 🏛️ The Political Rhetoric Analyzer

An interactive web application built with Python and Streamlit to analyze the evolution of political rhetoric in U.S. Presidential State of the Union (SOTU) addresses.

This tool scrapes historical speech data, processes it using natural language processing (NLP) techniques, and presents a rich, interactive dashboard for exploring trends in sentiment, keyword usage, and thematic content over time.

---

## ✨ Features

*   **Sentiment Analysis:** Tracks the sentiment (positivity/negativity) of a selected president's speeches over their term using a line chart.
*   **Keyword Frequency Tracking:** Allows a user to input any keyword and see a graph of how often it was used by a selected president over time.
*   **Interactive Speech Viewer:** Select and read the full text of any SOTU address for a given president.
*   **Dynamic Keyword Highlighting:** The chosen keyword is automatically highlighted in the speech text for easy scanning.
*   **Overall Topic Modeling:** Uses Latent Dirichlet Allocation (LDA) to discover the underlying thematic topics across the entire corpus of SOTU addresses. Users can dynamically adjust the number of topics to find.
*   **Automated Data Pipeline:** Includes scripts to perform the entire data workflow:
    1.  **Scraping:** Fetches and cleans speech transcripts from [The American Presidency Project](https://www.presidency.ucsb.edu/).
    2.  **Processing:** Cleans the text (lemmatization, stop-word removal) and performs sentiment analysis, saving the results to a structured CSV file.

---

## 🛠️ Tech Stack & Libraries

*   **Web Framework:** [Streamlit](https://streamlit.io/)
*   **Data Manipulation & Analysis:** [pandas](https://pandas.pydata.org/)
*   **Web Scraping:** [requests](https://requests.readthedocs.io/en/latest/) & [BeautifulSoup4](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
*   **NLP & Text Processing:**
    *   [NLTK (Natural Language Toolkit)](https://www.nltk.org/): For tokenization, stop-words, and lemmatization.
    *   [scikit-learn](https://scikit-learn.org/): For TF-IDF vectorization and LDA Topic Modeling.
    *   [VADER (Valence Aware Dictionary and sEntiment Reasoner)](https://github.com/cjhutto/vaderSentiment): For robust sentiment analysis.
*   **Data Visualization:** [matplotlib](https://matplotlib.org/) & [seaborn](https://seaborn.pydata.org/)
*   **Excel Export:** [openpyxl](https://openpyxl.readthedocs.io/en/stable/)

---

## 📂 Project Structure

```
/
├── app.py              # The main Streamlit web application
├── scraper.py          # Scrapes SOTU speeches and saves raw data
├── preprocessing.py    # Contains the text cleaning and preprocessing function
├── Analyzer.py         # Script to run the full data processing pipeline (cleaning + sentiment)
├── nltk_data.py        # Utility to download required NLTK datasets
├── requirements.txt    # Lists all Python package dependencies
└── README.md           # You are here
```

---

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
