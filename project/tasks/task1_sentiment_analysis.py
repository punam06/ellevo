"""
Task 1: Sentiment Analysis on Product Reviews
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.nlp_utils import preprocess_text, evaluate_classifier, create_wordcloud

def load_sample_data():
    """
    Create sample sentiment data for demonstration.
    In a real scenario, you would load IMDb or Amazon reviews dataset.
    """
    sample_reviews = [
        ("This movie is absolutely fantastic! Great acting and storyline.", "positive"),
        ("Terrible movie, worst acting I've ever seen. Complete waste of time.", "negative"),
        ("Amazing product, highly recommend it to everyone!", "positive"),
        ("Poor quality, broke after one day. Very disappointed.", "negative"),
        ("Excellent service and fast delivery. Very satisfied with purchase.", "positive"),
        ("The product didn't match the description. False advertising.", "negative"),
        ("Outstanding quality and great value for money. Love it!", "positive"),
        ("Completely useless product. Don't buy this junk.", "negative"),
        ("Great experience overall. Would definitely buy again.", "positive"),
        ("Customer service was rude and unhelpful. Avoid this company.", "negative"),
        ("Best purchase I've made in years. Highly recommend!", "positive"),
        ("Product arrived damaged and seller refused to help.", "negative"),
        ("Excellent build quality and works perfectly as advertised.", "positive"),
        ("Overpriced for what you get. Not worth the money.", "negative"),
        ("Fast shipping and product exactly as described. Happy customer!", "positive"),
        ("Complete waste of money. Product stopped working immediately.", "negative"),
        ("Great customer support and quick resolution of issues.", "positive"),
        ("Poor packaging resulted in damaged product. Very disappointed.", "negative"),
        ("Outstanding value and excellent performance. Couldn't be happier!", "positive"),
        ("Misleading product description. Does not work as advertised.", "negative")
    ]
    
    df = pd.DataFrame(sample_reviews, columns=['review', 'sentiment'])
    return df

def sentiment_analysis():
    """
    Main function to perform sentiment analysis.
    """
    print("=" * 60)
    print("TASK 1: SENTIMENT ANALYSIS ON PRODUCT REVIEWS")
    print("=" * 60)
    
    # Load data
    print("Loading sample data...")
    df = load_sample_data()
    print(f"Loaded {len(df)} reviews")
    print("\nSentiment distribution:")
    print(df['sentiment'].value_counts())
    
    # Preprocess text
    print("\nPreprocessing text...")
    df['processed_review'] = df['review'].apply(preprocess_text)
    
    # Create features using TF-IDF
    print("Creating TF-IDF features...")
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
    X = vectorizer.fit_transform(df['processed_review'])
    y = df['sentiment']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Train Logistic Regression
    print("\n" + "="*50)
    print("TRAINING LOGISTIC REGRESSION")
    print("="*50)
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    
    print("Logistic Regression Results:")
    lr_accuracy = accuracy_score(y_test, lr_pred)
    print(f"Accuracy: {lr_accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, lr_pred))
    
    # Train Naive Bayes
    print("\n" + "="*50)
    print("TRAINING NAIVE BAYES")
    print("="*50)
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    nb_pred = nb_model.predict(X_test)
    
    print("Naive Bayes Results:")
    nb_accuracy = accuracy_score(y_test, nb_pred)
    print(f"Accuracy: {nb_accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, nb_pred))
    
    # Compare models
    print("\n" + "="*50)
    print("MODEL COMPARISON")
    print("="*50)
    print(f"Logistic Regression Accuracy: {lr_accuracy:.4f}")
    print(f"Naive Bayes Accuracy: {nb_accuracy:.4f}")
    
    # Feature analysis
    print("\n" + "="*50)
    print("FEATURE ANALYSIS")
    print("="*50)
    
    # Get feature names
    feature_names = vectorizer.get_feature_names_out()
    
    # Most important features for Logistic Regression
    lr_coef = lr_model.coef_[0]
    
    # Top positive features
    top_positive_idx = np.argsort(lr_coef)[-10:]
    print("Top 10 Positive Features:")
    for idx in reversed(top_positive_idx):
        print(f"  {feature_names[idx]}: {lr_coef[idx]:.4f}")
    
    # Top negative features
    top_negative_idx = np.argsort(lr_coef)[:10]
    print("\nTop 10 Negative Features:")
    for idx in top_negative_idx:
        print(f"  {feature_names[idx]}: {lr_coef[idx]:.4f}")
    
    # Create word clouds
    print("\n" + "="*50)
    print("GENERATING WORD CLOUDS")
    print("="*50)
    
    positive_reviews = ' '.join(df[df['sentiment'] == 'positive']['processed_review'])
    negative_reviews = ' '.join(df[df['sentiment'] == 'negative']['processed_review'])
    
    # Visualization
    plt.style.use('default')
    
    # Word clouds
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    from wordcloud import WordCloud
    
    # Positive word cloud
    wordcloud_pos = WordCloud(width=400, height=300, background_color='white', 
                             colormap='Greens').generate(positive_reviews)
    axes[0].imshow(wordcloud_pos, interpolation='bilinear')
    axes[0].set_title('Positive Reviews Word Cloud')
    axes[0].axis('off')
    
    # Negative word cloud
    wordcloud_neg = WordCloud(width=400, height=300, background_color='white',
                             colormap='Reds').generate(negative_reviews)
    axes[1].imshow(wordcloud_neg, interpolation='bilinear')
    axes[1].set_title('Negative Reviews Word Cloud')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('results/task1_wordclouds.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, lr_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'], 
                yticklabels=['Negative', 'Positive'])
    plt.title('Confusion Matrix - Logistic Regression')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('results/task1_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save results
    results = {
        'logistic_regression_accuracy': lr_accuracy,
        'naive_bayes_accuracy': nb_accuracy,
        'best_model': 'Logistic Regression' if lr_accuracy > nb_accuracy else 'Naive Bayes',
        'total_samples': len(df),
        'positive_samples': len(df[df['sentiment'] == 'positive']),
        'negative_samples': len(df[df['sentiment'] == 'negative'])
    }
    
    with open('results/task1_results.txt', 'w') as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    
    print(f"\nTask 1 completed! Results saved to results/")
    print("Note: This uses sample data. For real implementation, download IMDb or Amazon reviews dataset.")
    
    return results

if __name__ == "__main__":
    sentiment_analysis()