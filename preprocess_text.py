import numpy as np
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_data(file_path):
    """
    Loads the dataset from a specified CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    data = pd.read_csv(file_path, encoding="utf-8", encoding_errors="replace")
    return data

def label_sentiment(rating):
    if rating <= 2:
        return 'negative'
    elif rating == 3:
        return 'neutral'
    else:
        return 'positive'

def preprocess_data(df_data):
    """
    Preprocesses a DataFrame by cleaning and preparing text data in the 'Review' column.

    Steps:
    1. Remove duplicate rows.
    2. Drop rows with missing values.
    3. Strip whitespace, remove non-alphanumeric characters, newline characters, and URLs.
    4. Convert text to lowercase and remove extra spaces.
    5. Remove rows with empty 'Review'.
    6. Reset index.

    Args:
        df_data (pd.DataFrame): DataFrame containing the 'Review' column.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    df_copy = df_data.copy()
    df_copy.drop_duplicates(inplace=True)
    df_copy.dropna(inplace=True)
    df_copy['Review'] = df_copy['Review'].str.strip()
    df_copy['Review'] = df_copy['Review'].str.replace(r'[^\w\s]', '', regex=True)
    df_copy['Review'] = df_copy['Review'].str.replace(r'\n', ' ', regex=True)
    df_copy['Review'] = df_copy['Review'].str.lower()
    df_copy['Review'] = df_copy['Review'].apply(lambda x: re.sub(r'[^a-z\s]', '', x))
    df_copy['Review'] = df_copy['Review'].apply(lambda x: re.sub(r'http\S+|www.\S+', '', x))
    df_copy['Review'] = df_copy['Review'].apply(lambda x: ' '.join(x.split()))
    df_copy = df_copy[df_copy['Review'].str.len() > 0]
    df_copy.reset_index(drop=True, inplace=True)
    df_copy['sentiment'] = df_copy['Rating'].apply(label_sentiment)
    return df_copy

def tokenize_text(df):
    """
    Tokenizes the text in the 'Review' column.

    Args:
        df (pd.DataFrame): DataFrame containing a 'Review' column.

    Returns:
        pd.DataFrame: DataFrame with a new 'tokens' column containing tokenized words.
    """
    df['tokens'] = df['Review'].apply(word_tokenize)
    return df

def remove_stopwords(df):
    """
    Removes stopwords from the tokenized text.

    Args:
        df (pd.DataFrame): DataFrame with a 'tokens' column.

    Returns:
        pd.DataFrame: DataFrame with a new 'tokens_no_stop' column containing tokens without stopwords.
    """
    stop_words = set(stopwords.words('english'))
    df['tokens_no_stop'] = df['tokens'].apply(lambda x: [word for word in x if word.lower() not in stop_words])
    return df

def lemmatize_text(df):
    """
    Lemmatizes the tokens in the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with a 'tokens_no_stop' column.

    Returns:
        pd.DataFrame: DataFrame with a new 'lemmatized' column containing lemmatized tokens.
    """
    lemmatizer = WordNetLemmatizer()
    df['lemmatized'] = df['tokens_no_stop'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])
    return df

def join_tokens(df):
    """
    Joins the lemmatized tokens back into a single string.

    Args:
        df (pd.DataFrame): DataFrame with a 'lemmatized' column.

    Returns:
        pd.DataFrame: DataFrame with a new 'processed_text' column containing the joined text.
    """
    df['processed_text'] = df['lemmatized'].apply(lambda x: ' '.join(x))
    return df

def create_tfidf(df, max_features=5000):
    """
    Converts the processed text data into a TF-IDF matrix.

    Args:
        df (pd.DataFrame): DataFrame containing the 'processed_text' column with preprocessed text data.
        max_features (int, optional): The maximum number of features to consider for the TF-IDF matrix.
                                      Defaults to 5000.

    Returns:
        tuple: A tuple containing:
            - tfidf_matrix (scipy.sparse.csr_matrix): The TF-IDF feature matrix.
            - tfidf (TfidfVectorizer): The fitted TfidfVectorizer object for possible reuse.
    """
    tfidf = TfidfVectorizer(max_features=max_features)
    tfidf_matrix = tfidf.fit_transform(df['processed_text'])
    return tfidf_matrix, tfidf

def create_balanced_dataset(df, n_samples=20491): 
    """ 
    Create a balanced dataset with specified number of samples 
    """ 
    # Convert sentiment to numeric 
    sentiment_map = { 
        'negative': 0, 
        'neutral': 1, 
        'positive': 2 
    } 
    df['label_sentiment'] = df['sentiment'].map(sentiment_map) 
     
    # Calculate samples per class 
    samples_per_class = n_samples // 3 
     
    # Get balanced data for each class 
    balanced_dfs = [] 
    for label in range(3): 
        class_df = df[df['label_sentiment'] == label] 
        if len(class_df) > samples_per_class: 
            balanced_dfs.append(class_df.sample(n=samples_per_class, 
random_state=42)) 
        else: 
            # If we don't have enough samples, oversample 
            balanced_dfs.append(class_df.sample(n=samples_per_class, 
replace=True, random_state=42)) 
     
    # Combine balanced datasets 
    balanced_df = pd.concat(balanced_dfs) 
     
    # Shuffle the final dataset 
    return balanced_df.sample(frac=1, random_state=42)


def data_description(df):
    """
    Visualizes the distribution of ratings in a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing a 'Rating' column.

    Returns:
        None: Displays a bar plot of the ratings distribution.
    """
    ratings_count = df['Rating'].value_counts(normalize=True) * 100
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(x=ratings_count.index, y=ratings_count.values, palette="viridis")
    for p in ax.patches:
        ax.annotate(
            f'{p.get_height():.1f}%', 
            (p.get_x() + p.get_width() / 2., p.get_height()), 
            ha='center', va='center', fontsize=12, 
            color='black', xytext=(0, 5), 
            textcoords='offset points'
        )
    plt.xlabel("Rating", fontsize=12)
    plt.ylabel("Percentage", fontsize=12)
    plt.title("Distribution of Ratings", fontsize=14)
    plt.show()

