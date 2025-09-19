"""
Utility functions for Natural Language Processing tasks.
Contains common preprocessing, evaluation, and visualization functions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
import string
from collections import Counter

# Text preprocessing imports
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

# Download required NLTK data (run once)
nltk_downloads = ['punkt', 'stopwords', 'wordnet', 'omw-1.4', 'averaged_perceptron_tagger']
for item in nltk_downloads:
    try:
        nltk.data.find(f'tokenizers/{item}')
    except LookupError:
        nltk.download(item, quiet=True)

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text, remove_stopwords=True, lemmatize=True, min_length=2):
    """
    Comprehensive text preprocessing function.
    
    Args:
        text (str): Input text to preprocess
        remove_stopwords (bool): Whether to remove stopwords
        lemmatize (bool): Whether to lemmatize words
        min_length (int): Minimum word length to keep
    
    Returns:
        str: Preprocessed text
    """
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove punctuation and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Filter tokens
    processed_tokens = []
    for token in tokens:
        # Skip short tokens
        if len(token) < min_length:
            continue
        
        # Skip stopwords if requested
        if remove_stopwords and token in stop_words:
            continue
        
        # Lemmatize if requested
        if lemmatize:
            token = lemmatizer.lemmatize(token)
        
        processed_tokens.append(token)
    
    return ' '.join(processed_tokens)

def create_feature_matrix(texts, method='tfidf', max_features=10000, ngram_range=(1, 1)):
    """
    Create feature matrix from texts using TF-IDF or Count Vectorizer.
    
    Args:
        texts (list): List of preprocessed texts
        method (str): 'tfidf' or 'count'
        max_features (int): Maximum number of features
        ngram_range (tuple): N-gram range for vectorization
    
    Returns:
        tuple: (feature_matrix, vectorizer)
    """
    if method == 'tfidf':
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english'
        )
    elif method == 'count':
        vectorizer = CountVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english'
        )
    else:
        raise ValueError("Method must be 'tfidf' or 'count'")
    
    feature_matrix = vectorizer.fit_transform(texts)
    return feature_matrix, vectorizer

def evaluate_classifier(y_true, y_pred, labels=None):
    """
    Comprehensive evaluation of classification results.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: List of label names for display
    
    Returns:
        dict: Dictionary containing various evaluation metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score (weighted): {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=labels))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'confusion_matrix': cm
    }

def create_wordcloud(text_data, title="Word Cloud", max_words=100, 
                    background_color='white', colormap='viridis'):
    """
    Create and display a word cloud from text data.
    
    Args:
        text_data (str or list): Text data for word cloud
        title (str): Title for the plot
        max_words (int): Maximum number of words in the cloud
        background_color (str): Background color
        colormap (str): Matplotlib colormap name
    """
    if isinstance(text_data, list):
        text_data = ' '.join(text_data)
    
    wordcloud = WordCloud(
        width=800, 
        height=400,
        max_words=max_words,
        background_color=background_color,
        colormap=colormap,
        relative_scaling=0.5
    ).generate(text_data)
    
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=16, pad=20)
    plt.tight_layout()
    plt.show()

def plot_feature_importance(model, feature_names, top_n=20, title="Feature Importance"):
    """
    Plot feature importance for models that have feature_importances_ or coef_ attribute.
    
    Args:
        model: Trained model
        feature_names: List of feature names
        top_n (int): Number of top features to display
        title (str): Plot title
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        print("Model doesn't have feature importance information")
        return
    
    # Get top features
    indices = np.argsort(importances)[-top_n:]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(indices)), importances[indices])
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.title(title)
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.show()

def get_most_frequent_words(texts, n=20, remove_stopwords=True):
    """
    Get the most frequent words from a list of texts.
    
    Args:
        texts (list): List of texts
        n (int): Number of top words to return
        remove_stopwords (bool): Whether to remove stopwords
    
    Returns:
        list: List of (word, frequency) tuples
    """
    all_words = []
    for text in texts:
        words = word_tokenize(text.lower())
        if remove_stopwords:
            words = [word for word in words if word not in stop_words and word.isalpha()]
        all_words.extend(words)
    
    return Counter(all_words).most_common(n)

def plot_word_frequency(word_freq, title="Most Frequent Words", top_n=20):
    """
    Plot word frequency bar chart.
    
    Args:
        word_freq (list): List of (word, frequency) tuples
        title (str): Plot title
        top_n (int): Number of words to display
    """
    words, freqs = zip(*word_freq[:top_n])
    
    plt.figure(figsize=(12, 6))
    plt.barh(range(len(words)), freqs)
    plt.yticks(range(len(words)), words)
    plt.xlabel('Frequency')
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

def load_and_prepare_data(file_path, text_column, label_column=None, sample_size=None):
    """
    Load and prepare data from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        text_column (str): Name of the text column
        label_column (str): Name of the label column (optional)
        sample_size (int): Number of samples to use (optional)
    
    Returns:
        tuple: (texts, labels) or just texts if no label_column
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} samples from {file_path}")
        
        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42)
            print(f"Sampled {sample_size} examples")
        
        texts = df[text_column].fillna('').astype(str)
        
        if label_column:
            labels = df[label_column]
            print(f"Label distribution:")
            print(labels.value_counts())
            return texts, labels
        else:
            return texts
    
    except FileNotFoundError:
        print(f"File {file_path} not found. Please check the path.")
        return None
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def train_and_evaluate_models(X_train, X_test, y_train, y_test, models=None):
    """
    Train and evaluate multiple models.
    
    Args:
        X_train, X_test: Training and test feature matrices
        y_train, y_test: Training and test labels
        models (dict): Dictionary of models to train
    
    Returns:
        dict: Results for each model
    """
    if models is None:
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Naive Bayes': MultinomialNB(),
            'SVM': SVC(kernel='linear', random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Training {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'f1_score': f1,
            'predictions': y_pred
        }
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-Score: {f1:.4f}")
    
    return results

def save_results(results, filename):
    """
    Save results to a file.
    
    Args:
        results (dict): Results dictionary
        filename (str): Output filename
    """
    with open(filename, 'w') as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    print(f"Results saved to {filename}")

# Constants for common dataset URLs (for reference)
DATASET_URLS = {
    'imdb': 'https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews',
    'ag_news': 'https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset',
    'fake_news': 'https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset',
    'conll2003': 'https://www.kaggle.com/datasets/abhinavwalia95/entity-annotated-corpus',
    'bbc_news': 'https://www.kaggle.com/datasets/alfathterry/bbc-full-text-document-classification'
}

def print_dataset_info():
    """Print information about recommended datasets."""
    print("Recommended Datasets:")
    print("=" * 50)
    for task, url in DATASET_URLS.items():
        print(f"{task.upper()}: {url}")
    print("=" * 50)
    print("Download these datasets to the 'data' folder before running the tasks.")