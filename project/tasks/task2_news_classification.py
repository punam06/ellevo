"""
Task 2: News Category Classification
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.nlp_utils import preprocess_text, evaluate_classifier, create_wordcloud

def load_sample_data():
    """
    Create sample news data for demonstration.
    In a real scenario, you would load the AG News dataset.
    """
    sample_news = [
        # Sports
        ("The basketball team won the championship after a thrilling final game.", "sports"),
        ("Tennis player serves an ace to win the match in straight sets.", "sports"),
        ("Football season starts next month with new regulations.", "sports"),
        ("Olympic swimmer breaks world record in 200m freestyle.", "sports"),
        ("Baseball team signs new pitcher for upcoming season.", "sports"),
        
        # Business
        ("Stock market reaches all-time high amid positive earnings reports.", "business"),
        ("Tech company announces new product launch for next quarter.", "business"),
        ("Oil prices surge following supply chain disruptions.", "business"),
        ("Merger between two major corporations creates industry giant.", "business"),
        ("Cryptocurrency market shows volatility in trading session.", "business"),
        
        # Technology
        ("New AI algorithm improves machine learning accuracy significantly.", "technology"),
        ("Smartphone manufacturer releases latest model with advanced features.", "technology"),
        ("Cloud computing platform expands global infrastructure.", "technology"),
        ("Cybersecurity firm develops new threat detection system.", "technology"),
        ("Electric vehicle sales increase dramatically this quarter.", "technology"),
        
        # Politics
        ("President signs new legislation into law after congressional approval.", "politics"),
        ("Senate debates healthcare reform bill in heated session.", "politics"),
        ("Election results show narrow victory for incumbent candidate.", "politics"),
        ("Supreme Court hearing addresses constitutional questions.", "politics"),
        ("International diplomats meet to discuss trade agreements.", "politics"),
        
        # Entertainment
        ("Movie wins multiple awards at prestigious film festival.", "entertainment"),
        ("Pop star announces world tour dates for next year.", "entertainment"),
        ("Television series finale breaks viewership records.", "entertainment"),
        ("Actor signs contract for upcoming blockbuster sequel.", "entertainment"),
        ("Music album debuts at number one on charts.", "entertainment"),
    ]
    
    df = pd.DataFrame(sample_news, columns=['text', 'category'])
    return df

def news_classification():
    """
    Main function to perform news category classification.
    """
    print("=" * 60)
    print("TASK 2: NEWS CATEGORY CLASSIFICATION")
    print("=" * 60)
    
    # Load data
    print("Loading sample news data...")
    df = load_sample_data()
    print(f"Loaded {len(df)} news articles")
    print("\nCategory distribution:")
    print(df['category'].value_counts())
    
    # Preprocess text
    print("\nPreprocessing text...")
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    # Create TF-IDF features
    print("Creating TF-IDF features...")
    vectorizer = TfidfVectorizer(
        max_features=2000, 
        stop_words='english', 
        ngram_range=(1, 2),
        min_df=1,  # Lower threshold for small dataset
        max_df=0.95
    )
    X = vectorizer.fit_transform(df['processed_text'])
    y = df['category']
    
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
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='linear', random_state=42)
    }
    
    results = {}
    predictions = {}
    
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
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
    
    # Compare models
    print("\n" + "="*50)
    print("MODEL COMPARISON")
    print("="*50)
    for model_name, accuracy in results.items():
        print(f"{model_name}: {accuracy:.4f}")
    
    # Best model analysis
    best_model_name = max(results, key=results.get)
    best_model = models[best_model_name]
    best_predictions = predictions[best_model_name]
    
    print(f"\nBest Model: {best_model_name}")
    print(f"Best Accuracy: {results[best_model_name]:.4f}")
    
    # Feature analysis for best model
    print("\n" + "="*50)
    print("FEATURE ANALYSIS")
    print("="*50)
    
    feature_names = vectorizer.get_feature_names_out()
    
    if hasattr(best_model, 'coef_'):
        # For linear models
        n_features = 5
        categories = best_model.classes_
        
        print(f"Top {n_features} features per category:")
        for i, category in enumerate(categories):
            top_indices = np.argsort(best_model.coef_[i])[-n_features:]
            print(f"\n{category.upper()}:")
            for idx in reversed(top_indices):
                print(f"  {feature_names[idx]}: {best_model.coef_[i][idx]:.4f}")
    
    elif hasattr(best_model, 'feature_importances_'):
        # For tree-based models
        importances = best_model.feature_importances_
        top_indices = np.argsort(importances)[-10:]
        
        print("Top 10 Most Important Features:")
        for idx in reversed(top_indices):
            print(f"  {feature_names[idx]}: {importances[idx]:.4f}")
    
    # Visualization
    print("\n" + "="*50)
    print("GENERATING VISUALIZATIONS")
    print("="*50)
    
    plt.style.use('default')
    
    # Category distribution
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    category_counts = df['category'].value_counts()
    plt.pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%')
    plt.title('News Category Distribution')
    
    # Model comparison
    plt.subplot(1, 2, 2)
    model_names = list(results.keys())
    accuracies = list(results.values())
    bars = plt.bar(model_names, accuracies)
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('results/task2_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Confusion Matrix for best model
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, best_predictions)
    
    # Create labels for confusion matrix
    labels = sorted(df['category'].unique())
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.ylabel('True Category')
    plt.xlabel('Predicted Category')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('results/task2_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Word clouds for each category
    categories = df['category'].unique()
    n_categories = len(categories)
    
    fig, axes = plt.subplots(1, n_categories, figsize=(20, 4))
    if n_categories == 1:
        axes = [axes]
    
    from wordcloud import WordCloud
    
    for i, category in enumerate(categories):
        category_text = ' '.join(df[df['category'] == category]['processed_text'])
        
        if len(category_text.strip()) > 0:
            wordcloud = WordCloud(width=300, height=200, 
                                background_color='white',
                                colormap='viridis').generate(category_text)
            axes[i].imshow(wordcloud, interpolation='bilinear')
            axes[i].set_title(f'{category.capitalize()} News')
            axes[i].axis('off')
        else:
            axes[i].text(0.5, 0.5, 'No data', ha='center', va='center', 
                        transform=axes[i].transAxes)
            axes[i].set_title(f'{category.capitalize()} News')
    
    plt.tight_layout()
    plt.savefig('results/task2_wordclouds.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save results
    final_results = {
        'total_articles': len(df),
        'categories': list(df['category'].unique()),
        'best_model': best_model_name,
        'best_accuracy': results[best_model_name],
        'feature_count': X.shape[1],
        'train_size': X_train.shape[0],
        'test_size': X_test.shape[0]
    }
    
    # Add individual model results
    for model_name, accuracy in results.items():
        final_results[f'{model_name.lower().replace(" ", "_")}_accuracy'] = accuracy
    
    with open('results/task2_results.txt', 'w') as f:
        for key, value in final_results.items():
            f.write(f"{key}: {value}\n")
    
    print(f"\nTask 2 completed! Results saved to results/")
    print("Note: This uses sample data. For real implementation, download AG News dataset from Kaggle.")
    
    return final_results

if __name__ == "__main__":
    news_classification()