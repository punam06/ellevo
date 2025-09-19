"""
Task 5: Topic Modeling on News Articles
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.cluster import KMeans
import pickle
from wordcloud import WordCloud
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.nlp_utils import preprocess_text

def load_sample_data():
    """
    Create sample news articles for topic modeling demonstration.
    In a real scenario, you would load the BBC News Dataset.
    """
    sample_articles = [
        # Technology topics
        "Apple Inc. releases new iPhone with advanced AI capabilities and improved camera system. The device features cutting-edge processors and enhanced battery life.",
        "Microsoft announces breakthrough in quantum computing research. Scientists develop new algorithms for faster data processing and encryption methods.",
        "Google's artificial intelligence system achieves human-level performance in language understanding tasks. Machine learning models show significant improvements.",
        "Tesla unveils new electric vehicle model with autonomous driving features. The car includes advanced sensors and neural network technology.",
        "Facebook introduces virtual reality platform for social networking. Users can interact in immersive digital environments with realistic avatars.",
        
        # Sports topics
        "Basketball championship reaches exciting conclusion as teams compete for the title. Players demonstrate exceptional skills in playoff games.",
        "Football season begins with new rules and regulations. Teams prepare for challenging matches with intensive training sessions.",
        "Olympic games showcase world-class athletic performances. Athletes from different countries compete in various sporting events and disciplines.",
        "Tennis tournament attracts top players from around the globe. Matches feature incredible rallies and strategic gameplay.",
        "Baseball league announces expansion plans with new teams joining. Stadiums prepare for increased attendance and fan engagement.",
        
        # Business/Finance topics
        "Stock market experiences significant volatility amid economic uncertainty. Investors monitor company earnings and financial indicators closely.",
        "Major corporation announces merger with competitor company. The deal will create industry leader with expanded market presence.",
        "Cryptocurrency market shows mixed signals as prices fluctuate. Digital currencies face regulatory challenges and adoption barriers.",
        "Banking sector reports strong quarterly profits despite challenges. Financial institutions adapt to changing market conditions.",
        "Retail companies struggle with supply chain disruptions. E-commerce platforms experience increased demand and shipping delays.",
        
        # Politics topics
        "President delivers speech addressing national economic policies. Government officials discuss budget allocations and spending priorities.",
        "Senate debates new healthcare legislation with bipartisan support. Lawmakers negotiate provisions for insurance coverage and medical costs.",
        "International summit brings world leaders together for climate discussions. Countries commit to reducing carbon emissions and environmental protection.",
        "Supreme Court ruling impacts voting rights and election procedures. Legal experts analyze constitutional implications and future consequences.",
        "Congressional committee investigates government transparency issues. Representatives examine disclosure requirements and public accountability measures.",
        
        # Health/Medicine topics
        "Medical researchers discover new treatment for chronic disease. Clinical trials show promising results for patient recovery rates.",
        "Pharmaceutical company develops innovative vaccine technology. Scientists work on prevention methods for infectious diseases.",
        "Hospital systems implement digital health record systems. Medical professionals adopt electronic documentation and patient monitoring tools.",
        "Public health officials launch awareness campaign about nutrition. Experts recommend dietary changes for disease prevention and wellness.",
        "Mental health services expand to address growing demand. Therapists and counselors provide support for anxiety and depression treatment."
    ]
    
    # Create labels for evaluation (in real scenario, these might not be available)
    labels = ['technology'] * 5 + ['sports'] * 5 + ['business'] * 5 + ['politics'] * 5 + ['health'] * 5
    
    df = pd.DataFrame({
        'text': sample_articles,
        'category': labels  # This is just for evaluation purposes
    })
    
    return df

def perform_lda_topic_modeling(texts, n_topics=5, max_features=1000):
    """
    Perform topic modeling using Latent Dirichlet Allocation.
    
    Args:
        texts (list): List of preprocessed texts
        n_topics (int): Number of topics to extract
        max_features (int): Maximum number of features for vectorization
    
    Returns:
        tuple: (lda_model, vectorizer, feature_matrix, topic_words)
    """
    print(f"Performing LDA topic modeling with {n_topics} topics...")
    
    # Use CountVectorizer for LDA (works better with raw term frequencies)
    vectorizer = CountVectorizer(
        max_features=max_features,
        stop_words='english',
        min_df=2,  # Ignore terms that appear in less than 2 documents
        max_df=0.8,  # Ignore terms that appear in more than 80% of documents
        ngram_range=(1, 2)
    )
    
    feature_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    
    # Fit LDA model
    lda_model = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        learning_method='batch',
        max_iter=25,
        learning_decay=0.7
    )
    
    lda_model.fit(feature_matrix)
    
    # Extract top words for each topic
    n_top_words = 10
    topic_words = []
    
    for topic_idx, topic in enumerate(lda_model.components_):
        top_words_idx = topic.argsort()[-n_top_words:][::-1]
        top_words = [feature_names[i] for i in top_words_idx]
        topic_words.append(top_words)
        
        print(f"Topic {topic_idx + 1}: {', '.join(top_words)}")
    
    return lda_model, vectorizer, feature_matrix, topic_words

def perform_nmf_topic_modeling(texts, n_topics=5, max_features=1000):
    """
    Perform topic modeling using Non-negative Matrix Factorization.
    
    Args:
        texts (list): List of preprocessed texts
        n_topics (int): Number of topics to extract
        max_features (int): Maximum number of features for vectorization
    
    Returns:
        tuple: (nmf_model, vectorizer, feature_matrix, topic_words)
    """
    print(f"Performing NMF topic modeling with {n_topics} topics...")
    
    # Use TF-IDF for NMF (adjusted for small datasets)
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words='english',
        min_df=1,  # Reduced for small sample datasets
        max_df=0.9,  # Increased for small datasets
        ngram_range=(1, 2)
    )
    
    feature_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    
    # Check if feature matrix is empty
    if feature_matrix.shape[1] == 0:
        raise ValueError("No features extracted. Try reducing min_df or increasing max_df.")
    
    # Fit NMF model with more robust parameters
    nmf_model = NMF(
        n_components=n_topics,
        random_state=42,
        max_iter=200,  # Increased iterations
        init='nndsvda',  # Better initialization
        beta_loss='frobenius',
        solver='mu'
    )
    
    nmf_model.fit(feature_matrix)
    
    # Extract top words for each topic
    n_top_words = 10
    topic_words = []
    
    for topic_idx, topic in enumerate(nmf_model.components_):
        top_words_idx = topic.argsort()[-n_top_words:][::-1]
        top_words = [feature_names[i] for i in top_words_idx]
        topic_words.append(top_words)
        
        print(f"Topic {topic_idx + 1}: {', '.join(top_words)}")
    
    return nmf_model, vectorizer, feature_matrix, topic_words

def analyze_document_topics(model, feature_matrix, texts, topic_words):
    """
    Analyze which topics are most prevalent in each document.
    
    Args:
        model: Fitted topic model
        feature_matrix: Document-term matrix
        texts: Original texts
        topic_words: List of top words for each topic
    
    Returns:
        list: List of topic assignments for each document
    """
    # Get topic probabilities for each document
    doc_topic_probs = model.transform(feature_matrix)
    
    # Assign each document to its most probable topic
    doc_topics = []
    
    for i, (text, topic_probs) in enumerate(zip(texts, doc_topic_probs)):
        dominant_topic = np.argmax(topic_probs)
        topic_prob = topic_probs[dominant_topic]
        
        doc_topics.append({
            'document_id': i,
            'text_preview': text[:100] + '...' if len(text) > 100 else text,
            'dominant_topic': dominant_topic,
            'topic_probability': topic_prob,
            'topic_words': ', '.join(topic_words[dominant_topic][:5])
        })
    
    return doc_topics

def evaluate_topic_coherence(model, feature_matrix, vectorizer):
    """
    Simple evaluation of topic coherence using perplexity (for LDA).
    
    Args:
        model: Fitted topic model
        feature_matrix: Document-term matrix
        vectorizer: Fitted vectorizer
    
    Returns:
        dict: Evaluation metrics
    """
    metrics = {}
    
    if hasattr(model, 'perplexity'):
        # LDA model
        perplexity = model.perplexity(feature_matrix)
        log_likelihood = model.score(feature_matrix)
        
        metrics['perplexity'] = perplexity
        metrics['log_likelihood'] = log_likelihood
        
        print(f"Model Perplexity: {perplexity:.2f}")
        print(f"Log Likelihood: {log_likelihood:.2f}")
    
    elif hasattr(model, 'reconstruction_err_'):
        # NMF model
        reconstruction_error = model.reconstruction_err_
        metrics['reconstruction_error'] = reconstruction_error
        
        print(f"Reconstruction Error: {reconstruction_error:.2f}")
    
    return metrics

def create_topic_visualizations(topic_words, doc_topics, model_name="Topic Model"):
    """
    Create visualizations for topic modeling results.
    
    Args:
        topic_words: List of top words for each topic
        doc_topics: List of document topic assignments
        model_name: Name of the model for titles
    """
    print(f"Creating visualizations for {model_name}...")
    
    plt.style.use('default')
    
    # Topic distribution
    topic_assignments = [doc['dominant_topic'] for doc in doc_topics]
    topic_counts = np.bincount(topic_assignments)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{model_name} Results', fontsize=16)
    
    # Topic distribution bar chart
    axes[0, 0].bar(range(len(topic_counts)), topic_counts, color='skyblue')
    axes[0, 0].set_title('Document Distribution Across Topics')
    axes[0, 0].set_xlabel('Topic')
    axes[0, 0].set_ylabel('Number of Documents')
    axes[0, 0].set_xticks(range(len(topic_counts)))
    axes[0, 0].set_xticklabels([f'Topic {i+1}' for i in range(len(topic_counts))])
    
    # Topic distribution pie chart
    axes[0, 1].pie(topic_counts, labels=[f'Topic {i+1}' for i in range(len(topic_counts))], 
                   autopct='%1.1f%%')
    axes[0, 1].set_title('Topic Distribution (Pie Chart)')
    
    # Topic probability distribution
    topic_probs = [doc['topic_probability'] for doc in doc_topics]
    axes[1, 0].hist(topic_probs, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[1, 0].set_title('Distribution of Topic Probabilities')
    axes[1, 0].set_xlabel('Topic Probability')
    axes[1, 0].set_ylabel('Frequency')
    
    # Word frequency in topics (simplified)
    all_topic_words = [word for topic in topic_words for word in topic[:5]]
    word_counts = {}
    for word in all_topic_words:
        word_counts[word] = word_counts.get(word, 0) + 1
    
    top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    words, counts = zip(*top_words) if top_words else ([], [])
    
    if words:
        axes[1, 1].barh(range(len(words)), counts, color='coral')
        axes[1, 1].set_yticks(range(len(words)))
        axes[1, 1].set_yticklabels(words)
        axes[1, 1].set_title('Most Frequent Words Across All Topics')
        axes[1, 1].set_xlabel('Frequency')
        axes[1, 1].invert_yaxis()
    
    plt.tight_layout()
    model_safe_name = model_name.lower().replace(' ', '_')
    plt.savefig(f'results/task5_{model_safe_name}_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_topic_wordclouds(topic_words, model_name="Topic Model"):
    """
    Create word clouds for each topic.
    
    Args:
        topic_words: List of top words for each topic
        model_name: Name of the model for titles
    """
    n_topics = len(topic_words)
    
    if n_topics == 0:
        return
    
    # Calculate grid size
    cols = min(3, n_topics)
    rows = (n_topics + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    if n_topics == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(f'{model_name} - Topic Word Clouds', fontsize=16)
    
    for i, words in enumerate(topic_words):
        row = i // cols
        col = i % cols
        
        if rows == 1:
            ax = axes[col] if cols > 1 else axes
        else:
            ax = axes[row, col] if cols > 1 else axes[row]
        
        # Create word cloud
        word_text = ' '.join(words)
        
        if len(word_text.strip()) > 0:
            wordcloud = WordCloud(width=400, height=300, 
                                 background_color='white',
                                 colormap='viridis').generate(word_text)
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.set_title(f'Topic {i+1}')
            ax.axis('off')
        else:
            ax.text(0.5, 0.5, 'No words', ha='center', va='center',
                   transform=ax.transAxes)
            ax.set_title(f'Topic {i+1}')
            ax.axis('off')
    
    # Hide unused subplots
    for i in range(n_topics, rows * cols):
        row = i // cols
        col = i % cols
        if rows == 1:
            ax = axes[col] if cols > 1 else axes
        else:
            ax = axes[row, col] if cols > 1 else axes[row]
        ax.axis('off')
    
    plt.tight_layout()
    model_safe_name = model_name.lower().replace(' ', '_')
    plt.savefig(f'results/task5_{model_safe_name}_wordclouds.png', dpi=300, bbox_inches='tight')
    plt.show()

def topic_modeling():
    """
    Main function to perform topic modeling on news articles.
    """
    print("=" * 60)
    print("TASK 5: TOPIC MODELING ON NEWS ARTICLES")
    print("=" * 60)
    
    # Load sample data
    print("Loading sample news articles...")
    df = load_sample_data()
    print(f"Loaded {len(df)} news articles")
    print("\nActual category distribution (for reference):")
    print(df['category'].value_counts())
    
    # Preprocess text
    print("\nPreprocessing text...")
    df['processed_text'] = df['text'].apply(preprocess_text)
    processed_texts = df['processed_text'].tolist()
    
    # Determine optimal number of topics
    n_topics = 5  # Can be adjusted based on data
    print(f"\nUsing {n_topics} topics for modeling...")
    
    # Perform LDA topic modeling
    print("\n" + "="*50)
    print("LATENT DIRICHLET ALLOCATION (LDA)")
    print("="*50)
    
    lda_model, lda_vectorizer, lda_matrix, lda_topic_words = perform_lda_topic_modeling(
        processed_texts, n_topics=n_topics
    )
    
    # Analyze document topics for LDA
    lda_doc_topics = analyze_document_topics(lda_model, lda_matrix, processed_texts, lda_topic_words)
    
    # Evaluate LDA model
    print("\nLDA Model Evaluation:")
    lda_metrics = evaluate_topic_coherence(lda_model, lda_matrix, lda_vectorizer)
    
    # Perform NMF topic modeling
    print("\n" + "="*50)
    print("NON-NEGATIVE MATRIX FACTORIZATION (NMF)")
    print("="*50)
    
    nmf_model, nmf_vectorizer, nmf_matrix, nmf_topic_words = perform_nmf_topic_modeling(
        processed_texts, n_topics=n_topics
    )
    
    # Analyze document topics for NMF
    nmf_doc_topics = analyze_document_topics(nmf_model, nmf_matrix, processed_texts, nmf_topic_words)
    
    # Evaluate NMF model
    print("\nNMF Model Evaluation:")
    nmf_metrics = evaluate_topic_coherence(nmf_model, nmf_matrix, nmf_vectorizer)
    
    # Compare models
    print("\n" + "="*50)
    print("MODEL COMPARISON")
    print("="*50)
    
    # Simple comparison based on topic assignment consistency
    lda_assignments = [doc['dominant_topic'] for doc in lda_doc_topics]
    nmf_assignments = [doc['dominant_topic'] for doc in nmf_doc_topics]
    
    print("LDA Topic Assignments:", lda_assignments)
    print("NMF Topic Assignments:", nmf_assignments)
    
    # Check agreement between models
    agreement = sum(1 for l, n in zip(lda_assignments, nmf_assignments) if l == n)
    agreement_rate = agreement / len(lda_assignments)
    print(f"Model Agreement Rate: {agreement_rate:.2f}")
    
    # Display sample document assignments
    print("\n" + "="*50)
    print("SAMPLE DOCUMENT TOPIC ASSIGNMENTS")
    print("="*50)
    
    print("LDA Results (first 5 documents):")
    for i, doc in enumerate(lda_doc_topics[:5]):
        print(f"Doc {i+1}: Topic {doc['dominant_topic']+1} "
              f"({doc['topic_probability']:.3f}) - {doc['topic_words']}")
    
    print("\nNMF Results (first 5 documents):")
    for i, doc in enumerate(nmf_doc_topics[:5]):
        print(f"Doc {i+1}: Topic {doc['dominant_topic']+1} "
              f"({doc['topic_probability']:.3f}) - {doc['topic_words']}")
    
    # Create visualizations
    print("\n" + "="*50)
    print("GENERATING VISUALIZATIONS")
    print("="*50)
    
    # LDA visualizations
    create_topic_visualizations(lda_topic_words, lda_doc_topics, "LDA")
    create_topic_wordclouds(lda_topic_words, "LDA")
    
    # NMF visualizations
    create_topic_visualizations(nmf_topic_words, nmf_doc_topics, "NMF")
    create_topic_wordclouds(nmf_topic_words, "NMF")
    
    # Topic comparison visualization
    plt.figure(figsize=(15, 8))
    
    # Side-by-side topic words comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # LDA topics
    y_pos = np.arange(n_topics)
    ax1.barh(y_pos, [1] * n_topics, alpha=0.3, color='blue')
    
    for i, words in enumerate(lda_topic_words):
        ax1.text(0.5, i, f"Topic {i+1}: {', '.join(words[:5])}", 
                ha='center', va='center', fontsize=10, weight='bold')
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([f'Topic {i+1}' for i in range(n_topics)])
    ax1.set_title('LDA Topics')
    ax1.set_xlabel('Topics')
    ax1.set_xlim(0, 1)
    
    # NMF topics
    ax2.barh(y_pos, [1] * n_topics, alpha=0.3, color='red')
    
    for i, words in enumerate(nmf_topic_words):
        ax2.text(0.5, i, f"Topic {i+1}: {', '.join(words[:5])}", 
                ha='center', va='center', fontsize=10, weight='bold')
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([f'Topic {i+1}' for i in range(n_topics)])
    ax2.set_title('NMF Topics')
    ax2.set_xlabel('Topics')
    ax2.set_xlim(0, 1)
    
    plt.suptitle('LDA vs NMF Topic Comparison', fontsize=16)
    plt.tight_layout()
    plt.savefig('results/task5_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save models and results
    print("\n" + "="*50)
    print("SAVING MODELS AND RESULTS")
    print("="*50)
    
    # Save models
    with open('results/task5_lda_model.pkl', 'wb') as f:
        pickle.dump(lda_model, f)
    
    with open('results/task5_nmf_model.pkl', 'wb') as f:
        pickle.dump(nmf_model, f)
    
    # Save vectorizers
    with open('results/task5_lda_vectorizer.pkl', 'wb') as f:
        pickle.dump(lda_vectorizer, f)
    
    with open('results/task5_nmf_vectorizer.pkl', 'wb') as f:
        pickle.dump(nmf_vectorizer, f)
    
    # Save results
    final_results = {
        'total_documents': len(df),
        'n_topics': n_topics,
        'lda_topics': lda_topic_words,
        'nmf_topics': nmf_topic_words,
        'model_agreement_rate': agreement_rate,
        'lda_feature_count': lda_matrix.shape[1],
        'nmf_feature_count': nmf_matrix.shape[1]
    }
    
    # Add metrics
    final_results.update(lda_metrics)
    final_results.update(nmf_metrics)
    
    # Save detailed results
    with open('results/task5_results.txt', 'w') as f:
        f.write("TOPIC MODELING RESULTS\n")
        f.write("=" * 30 + "\n\n")
        
        f.write(f"Total Documents: {final_results['total_documents']}\n")
        f.write(f"Number of Topics: {final_results['n_topics']}\n")
        f.write(f"Model Agreement Rate: {final_results['model_agreement_rate']:.3f}\n\n")
        
        f.write("LDA TOPICS:\n")
        for i, words in enumerate(lda_topic_words):
            f.write(f"Topic {i+1}: {', '.join(words)}\n")
        
        f.write("\nNMF TOPICS:\n")
        for i, words in enumerate(nmf_topic_words):
            f.write(f"Topic {i+1}: {', '.join(words)}\n")
        
        f.write("\nDOCUMENT ASSIGNMENTS (LDA):\n")
        for doc in lda_doc_topics:
            f.write(f"Doc {doc['document_id']+1}: Topic {doc['dominant_topic']+1} "
                   f"({doc['topic_probability']:.3f})\n")
        
        f.write("\nDOCUMENT ASSIGNMENTS (NMF):\n")
        for doc in nmf_doc_topics:
            f.write(f"Doc {doc['document_id']+1}: Topic {doc['dominant_topic']+1} "
                   f"({doc['topic_probability']:.3f})\n")
    
    # Save document-topic assignments as CSV
    lda_df = pd.DataFrame(lda_doc_topics)
    nmf_df = pd.DataFrame(nmf_doc_topics)
    
    lda_df.to_csv('results/task5_lda_document_topics.csv', index=False)
    nmf_df.to_csv('results/task5_nmf_document_topics.csv', index=False)
    
    print("Task 5 completed! Results saved to results/")
    print("Files generated:")
    print("  - task5_lda_analysis.png: LDA analysis charts")
    print("  - task5_lda_wordclouds.png: LDA topic word clouds")
    print("  - task5_nmf_analysis.png: NMF analysis charts")
    print("  - task5_nmf_wordclouds.png: NMF topic word clouds")
    print("  - task5_model_comparison.png: LDA vs NMF comparison")
    print("  - task5_lda_model.pkl: Trained LDA model")
    print("  - task5_nmf_model.pkl: Trained NMF model")
    print("  - task5_results.txt: Detailed results")
    print("  - task5_*_document_topics.csv: Document topic assignments")
    print("\nNote: This uses sample data. For real implementation, download BBC News Dataset from Kaggle.")
    
    return final_results

if __name__ == "__main__":
    topic_modeling()