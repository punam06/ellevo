"""
Task 3: Fake News Detection
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.naive_bayes import MultinomialNB
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.nlp_utils import preprocess_text, evaluate_classifier, create_wordcloud

def load_sample_data():
    """
    Create sample fake/real news data for demonstration.
    In a real scenario, you would load the Fake and Real News Dataset.
    """
    sample_articles = [
        # Real news
        ("The Federal Reserve announced a new interest rate policy following economic indicators showing steady growth in the manufacturing sector.", "real"),
        ("Scientists at major university publish research findings on climate change impacts in peer-reviewed journal.", "real"),
        ("Local government approves budget allocation for infrastructure improvements in downtown area.", "real"),
        ("Stock market closes higher following positive earnings reports from major technology companies.", "real"),
        ("International health organization releases guidelines for pandemic prevention measures.", "real"),
        ("Supreme Court schedules hearing for landmark constitutional case next month.", "real"),
        ("Weather service issues severe storm warning for coastal regions this weekend.", "real"),
        ("University researchers develop new treatment showing promise in clinical trials.", "real"),
        ("City council votes to approve new public transportation expansion project.", "real"),
        ("Economic data shows unemployment rate declining for third consecutive quarter.", "real"),
        
        # Fake news
        ("BREAKING: Secret government documents reveal aliens have been living among us for decades!", "fake"),
        ("Miracle cure discovered that doctors don't want you to know about - cures everything instantly!", "fake"),
        ("SHOCKING: Celebrity admits to being robot controlled by mysterious organization!", "fake"),
        ("Scientists confirm that drinking this common beverage will make you live forever!", "fake"),
        ("Government conspiracy exposed: They've been controlling weather with secret machines!", "fake"),
        ("URGENT: New study shows that breathing air is actually dangerous for your health!", "fake"),
        ("Billionaire reveals the one weird trick that made him rich overnight - banks hate him!", "fake"),
        ("Ancient prophecy predicted today's events with 100% accuracy - what happens next will amaze you!", "fake"),
        ("Local man discovers simple method to become millionaire in 24 hours using this one trick!", "fake"),
        ("EXCLUSIVE: Time traveler from 2050 warns about catastrophic event happening tomorrow!", "fake"),
    ]
    
    df = pd.DataFrame(sample_articles, columns=['text', 'label'])
    return df

def fake_news_detection():
    """
    Main function to perform fake news detection.
    """
    print("=" * 60)
    print("TASK 3: FAKE NEWS DETECTION")
    print("=" * 60)
    
    # Load data
    print("Loading sample news data...")
    df = load_sample_data()
    print(f"Loaded {len(df)} news articles")
    print("\nLabel distribution:")
    print(df['label'].value_counts())
    
    # Preprocess text
    print("\nPreprocessing text...")
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    # Create TF-IDF features
    print("Creating TF-IDF features...")
    vectorizer = TfidfVectorizer(
        max_features=1500,
        stop_words='english',
        ngram_range=(1, 3),  # Include trigrams for better fake news detection
        min_df=1,
        max_df=0.95
    )
    X = vectorizer.fit_transform(df['processed_text'])
    y = df['label']
    
    print(f"Feature matrix shape: {X.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'SVM': SVC(kernel='linear', random_state=42, probability=True),
        'Naive Bayes': MultinomialNB()
    }
    
    results = {}
    predictions = {}
    probabilities = {}
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"TRAINING {name.upper()}")
        print("="*50)
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        predictions[name] = y_pred
        
        # Get probabilities for ROC curve
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1]  # Probability of fake news
        else:
            y_prob = model.decision_function(X_test)
        probabilities[name] = y_prob
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, pos_label='fake')
        
        results[name] = {
            'accuracy': accuracy,
            'f1_score': f1
        }
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))
    
    # Compare models
    print("\n" + "="*50)
    print("MODEL COMPARISON")
    print("="*50)
    for model_name, metrics in results.items():
        print(f"{model_name}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")
    
    # Best model based on F1-score (important for fake news detection)
    best_model_name = max(results, key=lambda x: results[x]['f1_score'])
    best_model = models[best_model_name]
    best_predictions = predictions[best_model_name]
    
    print(f"\nBest Model (by F1-Score): {best_model_name}")
    print(f"Best F1-Score: {results[best_model_name]['f1_score']:.4f}")
    
    # Feature analysis
    print("\n" + "="*50)
    print("FEATURE ANALYSIS")
    print("="*50)
    
    feature_names = vectorizer.get_feature_names_out()
    
    if hasattr(best_model, 'coef_'):
        coef = best_model.coef_[0]
        
        # Features most indicative of fake news (positive coefficients)
        top_fake_indices = np.argsort(coef)[-15:]
        print("Top 15 features indicating FAKE news:")
        for idx in reversed(top_fake_indices):
            print(f"  {feature_names[idx]}: {coef[idx]:.4f}")
        
        print("\nTop 15 features indicating REAL news:")
        # Features most indicative of real news (negative coefficients)
        top_real_indices = np.argsort(coef)[:15]
        for idx in top_real_indices:
            print(f"  {feature_names[idx]}: {coef[idx]:.4f}")
    
    # Text analysis
    print("\n" + "="*50)
    print("TEXT PATTERN ANALYSIS")
    print("="*50)
    
    # Analyze text characteristics
    fake_articles = df[df['label'] == 'fake']['text']
    real_articles = df[df['label'] == 'real']['text']
    
    # Average text length
    fake_avg_len = fake_articles.str.len().mean()
    real_avg_len = real_articles.str.len().mean()
    
    print(f"Average length of fake news: {fake_avg_len:.1f} characters")
    print(f"Average length of real news: {real_avg_len:.1f} characters")
    
    # Exclamation marks (often more in fake news)
    fake_exclamations = fake_articles.str.count('!').mean()
    real_exclamations = real_articles.str.count('!').mean()
    
    print(f"Average exclamation marks in fake news: {fake_exclamations:.2f}")
    print(f"Average exclamation marks in real news: {real_exclamations:.2f}")
    
    # Capital letters (often more in fake news)
    fake_caps = fake_articles.str.count('[A-Z]').mean()
    real_caps = real_articles.str.count('[A-Z]').mean()
    
    print(f"Average capital letters in fake news: {fake_caps:.2f}")
    print(f"Average capital letters in real news: {real_caps:.2f}")
    
    # Visualization
    print("\n" + "="*50)
    print("GENERATING VISUALIZATIONS")
    print("="*50)
    
    plt.style.use('default')
    
    # Model performance comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Accuracy comparison
    model_names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in model_names]
    f1_scores = [results[name]['f1_score'] for name in model_names]
    
    axes[0, 0].bar(model_names, accuracies, color='skyblue')
    axes[0, 0].set_title('Model Accuracy Comparison')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_xticklabels(model_names, rotation=45)
    
    for i, acc in enumerate(accuracies):
        axes[0, 0].text(i, acc + 0.01, f'{acc:.3f}', ha='center', va='bottom')
    
    # F1-Score comparison
    axes[0, 1].bar(model_names, f1_scores, color='lightcoral')
    axes[0, 1].set_title('Model F1-Score Comparison')
    axes[0, 1].set_ylabel('F1-Score')
    axes[0, 1].set_xticklabels(model_names, rotation=45)
    
    for i, f1 in enumerate(f1_scores):
        axes[0, 1].text(i, f1 + 0.01, f'{f1:.3f}', ha='center', va='bottom')
    
    # Text length distribution
    axes[1, 0].hist([fake_articles.str.len(), real_articles.str.len()], 
                   bins=10, alpha=0.7, label=['Fake', 'Real'], color=['red', 'blue'])
    axes[1, 0].set_title('Text Length Distribution')
    axes[1, 0].set_xlabel('Text Length (characters)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    
    # Exclamation mark distribution
    fake_excl_counts = fake_articles.str.count('!')
    real_excl_counts = real_articles.str.count('!')
    
    axes[1, 1].hist([fake_excl_counts, real_excl_counts], 
                   bins=range(0, max(max(fake_excl_counts), max(real_excl_counts)) + 2), 
                   alpha=0.7, label=['Fake', 'Real'], color=['red', 'blue'])
    axes[1, 1].set_title('Exclamation Marks Distribution')
    axes[1, 1].set_xlabel('Number of Exclamation Marks')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('results/task3_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, best_predictions, labels=['real', 'fake'])
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Real', 'Fake'],
                yticklabels=['Real', 'Fake'])
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('results/task3_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Word clouds
    fake_text = ' '.join(df[df['label'] == 'fake']['processed_text'])
    real_text = ' '.join(df[df['label'] == 'real']['processed_text'])
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    from wordcloud import WordCloud
    
    # Fake news word cloud
    if len(fake_text.strip()) > 0:
        wordcloud_fake = WordCloud(width=400, height=300, background_color='white',
                                  colormap='Reds').generate(fake_text)
        axes[0].imshow(wordcloud_fake, interpolation='bilinear')
        axes[0].set_title('Fake News Word Cloud')
        axes[0].axis('off')
    
    # Real news word cloud
    if len(real_text.strip()) > 0:
        wordcloud_real = WordCloud(width=400, height=300, background_color='white',
                                  colormap='Blues').generate(real_text)
        axes[1].imshow(wordcloud_real, interpolation='bilinear')
        axes[1].set_title('Real News Word Cloud')
        axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('results/task3_wordclouds.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save results
    final_results = {
        'total_articles': len(df),
        'fake_articles': len(df[df['label'] == 'fake']),
        'real_articles': len(df[df['label'] == 'real']),
        'best_model': best_model_name,
        'best_accuracy': results[best_model_name]['accuracy'],
        'best_f1_score': results[best_model_name]['f1_score'],
        'feature_count': X.shape[1],
        'avg_fake_length': fake_avg_len,
        'avg_real_length': real_avg_len,
        'avg_fake_exclamations': fake_exclamations,
        'avg_real_exclamations': real_exclamations
    }
    
    # Add individual model results
    for model_name, metrics in results.items():
        final_results[f'{model_name.lower().replace(" ", "_")}_accuracy'] = metrics['accuracy']
        final_results[f'{model_name.lower().replace(" ", "_")}_f1_score'] = metrics['f1_score']
    
    with open('results/task3_results.txt', 'w') as f:
        for key, value in final_results.items():
            f.write(f"{key}: {value}\n")
    
    print(f"\nTask 3 completed! Results saved to results/")
    print("Note: This uses sample data. For real implementation, download Fake and Real News Dataset from Kaggle.")
    
    return final_results

if __name__ == "__main__":
    fake_news_detection()