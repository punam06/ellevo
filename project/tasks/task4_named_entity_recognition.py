"""
Task 4: Named Entity Recognition (NER) from News Articles
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
from spacy import displacy
from collections import Counter, defaultdict
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.nlp_utils import preprocess_text

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
    print("SpaCy model loaded successfully!")
except IOError:
    print("SpaCy model not found. Please run: python -m spacy download en_core_web_sm")
    sys.exit(1)

def load_sample_data():
    """
    Create sample news articles for NER demonstration.
    In a real scenario, you would load the CoNLL2003 dataset.
    """
    sample_articles = [
        "Apple Inc. CEO Tim Cook announced the company's latest iPhone at the Steve Jobs Theater in Cupertino, California. The event was attended by tech journalists from around the world.",
        
        "President Joe Biden met with European leaders in Brussels, Belgium, to discuss NATO policies and the ongoing situation in Ukraine. The meeting included French President Emmanuel Macron and German Chancellor Olaf Scholz.",
        
        "Tesla's stock price surged after Elon Musk tweeted about the company's expansion plans in China. The electric vehicle manufacturer is building a new factory in Shanghai.",
        
        "The New York Times reported that Amazon is planning to open new fulfillment centers in Texas and Florida. Jeff Bezos, the company's founder, stepped down as CEO last year.",
        
        "Microsoft announced a partnership with OpenAI to develop advanced artificial intelligence systems. The collaboration was revealed at a conference in Seattle, Washington.",
        
        "Google's parent company Alphabet reported strong quarterly earnings. CEO Sundar Pichai highlighted the company's progress in cloud computing and YouTube advertising revenue.",
        
        "Facebook, now Meta, is investing heavily in virtual reality technology. Mark Zuckerberg demonstrated the company's latest VR headset at their headquarters in Menlo Park, California.",
        
        "The Federal Reserve Chairman Jerome Powell announced interest rate changes during a press conference in Washington, D.C. The decision affects banks across the United States.",
        
        "Netflix signed a multi-billion dollar deal with Sony Pictures Entertainment to stream their movies. The agreement was negotiated in Los Angeles and approved by both companies' boards.",
        
        "SpaceX successfully launched a Falcon 9 rocket from Cape Canaveral, Florida. The mission delivered satellites for Starlink, Elon Musk's internet service project."
    ]
    
    return sample_articles

def extract_entities_spacy(texts):
    """
    Extract named entities using spaCy.
    
    Args:
        texts (list): List of text strings
    
    Returns:
        list: List of dictionaries containing entity information
    """
    all_entities = []
    
    for i, text in enumerate(texts):
        doc = nlp(text)
        text_entities = []
        
        for ent in doc.ents:
            entity_info = {
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char,
                'description': spacy.explain(ent.label_),
                'article_id': i
            }
            text_entities.append(entity_info)
        
        all_entities.extend(text_entities)
    
    return all_entities

def analyze_entities(entities):
    """
    Analyze extracted entities and generate statistics.
    
    Args:
        entities (list): List of entity dictionaries
    
    Returns:
        dict: Analysis results
    """
    # Count entities by type
    entity_types = Counter([ent['label'] for ent in entities])
    
    # Count entities by text
    entity_texts = Counter([ent['text'] for ent in entities])
    
    # Group entities by type
    entities_by_type = defaultdict(list)
    for ent in entities:
        entities_by_type[ent['label']].append(ent['text'])
    
    # Get unique entities per type
    unique_entities_by_type = {}
    for ent_type, ent_list in entities_by_type.items():
        unique_entities_by_type[ent_type] = list(set(ent_list))
    
    return {
        'entity_types': entity_types,
        'entity_texts': entity_texts,
        'entities_by_type': dict(entities_by_type),
        'unique_entities_by_type': unique_entities_by_type,
        'total_entities': len(entities),
        'unique_entities': len(set([ent['text'] for ent in entities]))
    }

def compare_ner_models(texts):
    """
    Compare different spaCy models for NER.
    
    Args:
        texts (list): List of text strings
    
    Returns:
        dict: Comparison results
    """
    models_to_compare = ["en_core_web_sm"]
    
    # Try to load additional models if available
    try:
        import spacy
        models_available = ["en_core_web_md", "en_core_web_lg"]
        for model_name in models_available:
            try:
                spacy.load(model_name)
                models_to_compare.append(model_name)
            except IOError:
                pass
    except:
        pass
    
    comparison_results = {}
    
    for model_name in models_to_compare:
        try:
            model_nlp = spacy.load(model_name)
            model_entities = []
            
            for text in texts:
                doc = model_nlp(text)
                for ent in doc.ents:
                    model_entities.append({
                        'text': ent.text,
                        'label': ent.label_
                    })
            
            comparison_results[model_name] = {
                'total_entities': len(model_entities),
                'unique_entities': len(set([ent['text'] for ent in model_entities])),
                'entity_types': Counter([ent['label'] for ent in model_entities])
            }
            
        except IOError:
            print(f"Model {model_name} not available")
    
    return comparison_results

def visualize_entities(text, filename=None):
    """
    Create a visualization of named entities in text.
    
    Args:
        text (str): Input text
        filename (str): Optional filename to save visualization
    """
    doc = nlp(text)
    
    # Generate HTML visualization
    html = displacy.render(doc, style="ent", jupyter=False)
    
    if filename:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"Entity visualization saved to {filename}")
    
    return html

def named_entity_recognition():
    """
    Main function to perform Named Entity Recognition.
    """
    print("=" * 60)
    print("TASK 4: NAMED ENTITY RECOGNITION (NER)")
    print("=" * 60)
    
    # Load sample data
    print("Loading sample news articles...")
    articles = load_sample_data()
    print(f"Loaded {len(articles)} articles")
    
    # Extract entities using spaCy
    print("\nExtracting named entities using spaCy...")
    entities = extract_entities_spacy(articles)
    print(f"Extracted {len(entities)} entities")
    
    # Analyze entities
    print("\nAnalyzing extracted entities...")
    analysis = analyze_entities(entities)
    
    print(f"Total entities: {analysis['total_entities']}")
    print(f"Unique entities: {analysis['unique_entities']}")
    print(f"Entity types found: {len(analysis['entity_types'])}")
    
    # Display entity type distribution
    print("\n" + "="*50)
    print("ENTITY TYPE DISTRIBUTION")
    print("="*50)
    for ent_type, count in analysis['entity_types'].most_common():
        description = spacy.explain(ent_type) or "No description available"
        print(f"{ent_type}: {count} ({description})")
    
    # Display most common entities
    print("\n" + "="*50)
    print("MOST COMMON ENTITIES")
    print("="*50)
    for entity, count in analysis['entity_texts'].most_common(10):
        print(f"'{entity}': {count} occurrences")
    
    # Display entities by type
    print("\n" + "="*50)
    print("ENTITIES BY TYPE")
    print("="*50)
    for ent_type, ent_list in analysis['unique_entities_by_type'].items():
        if len(ent_list) > 0:
            description = spacy.explain(ent_type) or "No description"
            print(f"\n{ent_type} ({description}):")
            for entity in sorted(set(ent_list))[:10]:  # Show top 10
                print(f"  - {entity}")
            if len(set(ent_list)) > 10:
                print(f"  ... and {len(set(ent_list)) - 10} more")
    
    # Compare different spaCy models
    print("\n" + "="*50)
    print("MODEL COMPARISON")
    print("="*50)
    model_comparison = compare_ner_models(articles)
    
    for model_name, results in model_comparison.items():
        print(f"\n{model_name}:")
        print(f"  Total entities: {results['total_entities']}")
        print(f"  Unique entities: {results['unique_entities']}")
        print(f"  Entity types: {len(results['entity_types'])}")
    
    # Create visualizations
    print("\n" + "="*50)
    print("GENERATING VISUALIZATIONS")
    print("="*50)
    
    plt.style.use('default')
    
    # Entity type distribution
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Bar chart of entity types
    entity_types = list(analysis['entity_types'].keys())
    entity_counts = list(analysis['entity_types'].values())
    
    axes[0, 0].bar(entity_types, entity_counts, color='skyblue')
    axes[0, 0].set_title('Distribution of Entity Types')
    axes[0, 0].set_xlabel('Entity Type')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Pie chart of entity types
    axes[0, 1].pie(entity_counts, labels=entity_types, autopct='%1.1f%%')
    axes[0, 1].set_title('Entity Types Distribution (Pie Chart)')
    
    # Most common entities
    common_entities = analysis['entity_texts'].most_common(10)
    entities_names, entities_counts = zip(*common_entities)
    
    axes[1, 0].barh(range(len(entities_names)), entities_counts, color='lightgreen')
    axes[1, 0].set_yticks(range(len(entities_names)))
    axes[1, 0].set_yticklabels(entities_names)
    axes[1, 0].set_title('Most Common Entities')
    axes[1, 0].set_xlabel('Frequency')
    axes[1, 0].invert_yaxis()
    
    # Entities per article
    entities_per_article = defaultdict(int)
    for entity in entities:
        entities_per_article[entity['article_id']] += 1
    
    article_ids = list(entities_per_article.keys())
    entities_counts_per_article = [entities_per_article[aid] for aid in article_ids]
    
    axes[1, 1].bar(article_ids, entities_counts_per_article, color='coral')
    axes[1, 1].set_title('Entities per Article')
    axes[1, 1].set_xlabel('Article ID')
    axes[1, 1].set_ylabel('Number of Entities')
    
    plt.tight_layout()
    plt.savefig('results/task4_ner_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Entity type heatmap (if we have enough data)
    if len(analysis['entity_types']) > 1:
        # Create entity co-occurrence matrix (simplified)
        entity_cooccurrence = defaultdict(lambda: defaultdict(int))
        
        # Group entities by article
        entities_by_article = defaultdict(list)
        for entity in entities:
            entities_by_article[entity['article_id']].append(entity['label'])
        
        # Calculate co-occurrence
        for article_id, article_entities in entities_by_article.items():
            unique_types = set(article_entities)
            for type1 in unique_types:
                for type2 in unique_types:
                    if type1 != type2:
                        entity_cooccurrence[type1][type2] += 1
        
        # Create heatmap
        if entity_cooccurrence:
            entity_types_list = sorted(list(analysis['entity_types'].keys()))
            cooc_matrix = np.zeros((len(entity_types_list), len(entity_types_list)))
            
            for i, type1 in enumerate(entity_types_list):
                for j, type2 in enumerate(entity_types_list):
                    cooc_matrix[i][j] = entity_cooccurrence[type1][type2]
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(cooc_matrix, 
                       xticklabels=entity_types_list,
                       yticklabels=entity_types_list,
                       annot=True, fmt='.0f', cmap='Blues')
            plt.title('Entity Type Co-occurrence Matrix')
            plt.xlabel('Entity Type')
            plt.ylabel('Entity Type')
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig('results/task4_cooccurrence_matrix.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    # Generate entity visualizations for sample texts
    print("Generating entity visualizations...")
    
    # Save entity visualization for first article
    if len(articles) > 0:
        visualization_html = visualize_entities(
            articles[0], 
            "results/task4_entity_visualization.html"
        )
    
    # Create detailed entity report
    print("\n" + "="*50)
    print("DETAILED ENTITY ANALYSIS")
    print("="*50)
    
    # Analyze specific entity types
    person_entities = [ent for ent in entities if ent['label'] == 'PERSON']
    org_entities = [ent for ent in entities if ent['label'] == 'ORG']
    location_entities = [ent for ent in entities if ent['label'] in ['GPE', 'LOC']]
    
    print(f"Persons mentioned: {len(set([ent['text'] for ent in person_entities]))}")
    print(f"Organizations mentioned: {len(set([ent['text'] for ent in org_entities]))}")
    print(f"Locations mentioned: {len(set([ent['text'] for ent in location_entities]))}")
    
    # Save results
    final_results = {
        'total_articles': len(articles),
        'total_entities': analysis['total_entities'],
        'unique_entities': analysis['unique_entities'],
        'entity_types_count': len(analysis['entity_types']),
        'most_common_entity': analysis['entity_texts'].most_common(1)[0] if analysis['entity_texts'] else None,
        'persons_count': len(set([ent['text'] for ent in person_entities])),
        'organizations_count': len(set([ent['text'] for ent in org_entities])),
        'locations_count': len(set([ent['text'] for ent in location_entities])),
    }
    
    # Add entity type counts
    for ent_type, count in analysis['entity_types'].items():
        final_results[f'{ent_type.lower()}_entities'] = count
    
    # Save detailed results
    with open('results/task4_results.txt', 'w') as f:
        f.write("NAMED ENTITY RECOGNITION RESULTS\n")
        f.write("=" * 40 + "\n\n")
        
        for key, value in final_results.items():
            f.write(f"{key}: {value}\n")
        
        f.write("\nENTITY TYPE DESCRIPTIONS:\n")
        for ent_type in analysis['entity_types'].keys():
            description = spacy.explain(ent_type) or "No description available"
            f.write(f"{ent_type}: {description}\n")
        
        f.write("\nUNIQUE ENTITIES BY TYPE:\n")
        for ent_type, ent_list in analysis['unique_entities_by_type'].items():
            f.write(f"\n{ent_type}:\n")
            for entity in sorted(set(ent_list)):
                f.write(f"  - {entity}\n")
    
    # Save entities as CSV
    entities_df = pd.DataFrame(entities)
    entities_df.to_csv('results/task4_entities.csv', index=False)
    
    print(f"\nTask 4 completed! Results saved to results/")
    print("Files generated:")
    print("  - task4_ner_analysis.png: Entity analysis charts")
    print("  - task4_cooccurrence_matrix.png: Entity co-occurrence matrix")
    print("  - task4_entity_visualization.html: Interactive entity visualization")
    print("  - task4_results.txt: Detailed results")
    print("  - task4_entities.csv: All extracted entities")
    print("\nNote: This uses sample data. For real implementation, download CoNLL2003 dataset from Kaggle.")
    
    return final_results

if __name__ == "__main__":
    named_entity_recognition()