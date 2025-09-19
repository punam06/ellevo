# Natural Language Processing Tasks

This project implements 5 comprehensive NLP tasks as specified in the requirements document. Each task includes complete implementation, evaluation, and visualization components.

## Project Structure

```
project/
├── data/                          # Place datasets here
├── notebooks/                     # Jupyter notebooks for exploration
├── results/                       # Output files, models, and visualizations
├── tasks/                         # Individual task implementations
│   ├── task1_sentiment_analysis.py
│   ├── task2_news_classification.py
│   ├── task3_fake_news_detection.py
│   ├── task4_named_entity_recognition.py
│   └── task5_topic_modeling.py
├── utils/                         # Utility functions
│   └── nlp_utils.py
├── requirements.txt               # Python dependencies
├── run_tasks.py                  # Main script to run all tasks
└── README.md                     # This file
```

## Tasks Overview

### Task 1: Sentiment Analysis on Product Reviews
- **Dataset**: Sample data (recommend IMDb Reviews or Amazon Product Reviews)
- **Techniques**: TF-IDF, Logistic Regression, Naive Bayes
- **Features**: Text preprocessing, model comparison, word clouds, feature analysis
- **Output**: Accuracy metrics, confusion matrix, visualizations

### Task 2: News Category Classification
- **Dataset**: Sample data (recommend AG News Dataset)
- **Techniques**: TF-IDF, Logistic Regression, Random Forest, SVM
- **Features**: Multi-class classification, feature analysis, category-wise word clouds
- **Output**: Model comparison, confusion matrix, accuracy metrics

### Task 3: Fake News Detection
- **Dataset**: Sample data (recommend Fake and Real News Dataset)
- **Techniques**: TF-IDF with n-grams, Logistic Regression, SVM, Naive Bayes
- **Features**: Text pattern analysis, linguistic feature extraction
- **Output**: F1-score, precision/recall, text characteristic analysis

### Task 4: Named Entity Recognition (NER)
- **Dataset**: Sample data (recommend CoNLL2003 Dataset)
- **Techniques**: spaCy NER models, entity extraction and analysis
- **Features**: Entity visualization, co-occurrence analysis, model comparison
- **Output**: Entity statistics, interactive visualizations, CSV exports

### Task 5: Topic Modeling
- **Dataset**: Sample data (recommend BBC News Dataset)
- **Techniques**: Latent Dirichlet Allocation (LDA), Non-negative Matrix Factorization (NMF)
- **Features**: Model comparison, topic visualization, document-topic assignments
- **Output**: Topic word clouds, model evaluation, topic coherence

## Setup and Installation

### 1. Environment Setup

```bash
# Create virtual environment
python3 -m venv nlp_env

# Activate virtual environment
source nlp_env/bin/activate  # On macOS/Linux
# or
nlp_env\Scripts\activate     # On Windows

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Download spaCy model
python -m spacy download en_core_web_sm
```

### 2. Running the Tasks

#### Run All Tasks
```bash
python run_tasks.py
```

#### Run Individual Tasks
```bash
python run_tasks.py 1  # Task 1: Sentiment Analysis
python run_tasks.py 2  # Task 2: News Classification
python run_tasks.py 3  # Task 3: Fake News Detection
python run_tasks.py 4  # Task 4: Named Entity Recognition
python run_tasks.py 5  # Task 5: Topic Modeling
```

#### Run Tasks Directly
```bash
cd tasks
python task1_sentiment_analysis.py
python task2_news_classification.py
python task3_fake_news_detection.py
python task4_named_entity_recognition.py
python task5_topic_modeling.py
```

## Recommended Datasets

For production use, download these datasets from Kaggle:

1. **Task 1**: [IMDb Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
2. **Task 2**: [AG News Classification](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset)
3. **Task 3**: [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
4. **Task 4**: [CoNLL2003 NER Dataset](https://www.kaggle.com/datasets/abhinavwalia95/entity-annotated-corpus)
5. **Task 5**: [BBC News Classification](https://www.kaggle.com/datasets/alfathterry/bbc-full-text-document-classification)

Place downloaded datasets in the `data/` folder and modify the data loading functions in each task file.

## Dependencies

### Core Libraries
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms
- **matplotlib**: Data visualization
- **seaborn**: Statistical data visualization

### NLP Libraries
- **nltk**: Natural language processing toolkit
- **spacy**: Advanced NLP library
- **wordcloud**: Word cloud generation

### Additional Libraries
- **xgboost**: Gradient boosting framework
- **lightgbm**: Gradient boosting framework
- **jupyter**: Interactive notebooks

## Features

### Text Preprocessing
- Lowercasing and cleaning
- Stopword removal
- Lemmatization
- N-gram feature extraction
- TF-IDF vectorization

### Model Evaluation
- Cross-validation
- Multiple evaluation metrics
- Confusion matrices
- Feature importance analysis

### Visualizations
- Word clouds
- Bar charts and histograms
- Confusion matrices
- Topic visualizations
- Entity relationship maps

### Export Capabilities
- Model persistence (pickle)
- CSV result exports
- PNG/HTML visualizations
- Detailed text reports

## Results and Output

Each task generates:

1. **Text Reports**: Detailed results in `results/taskX_results.txt`
2. **Visualizations**: Charts and plots saved as PNG files
3. **Model Files**: Trained models saved as pickle files
4. **Data Exports**: Results exported as CSV files
5. **Interactive Outputs**: HTML visualizations where applicable

## Performance Notes

- All tasks include sample data for immediate execution
- Processing time varies based on dataset size
- Memory usage scales with vocabulary size and number of documents
- GPU acceleration not required but can improve performance for large datasets

## Customization

### Adding New Datasets
1. Place dataset files in the `data/` folder
2. Modify the `load_sample_data()` function in the relevant task file
3. Adjust preprocessing parameters as needed

### Modifying Models
- Edit model parameters in the respective task files
- Add new algorithms by importing them and including in the model comparison
- Adjust evaluation metrics based on your requirements

### Extending Functionality
- Add new preprocessing steps in `utils/nlp_utils.py`
- Create additional visualization functions
- Implement custom evaluation metrics

## Troubleshooting

### Common Issues

1. **spaCy model not found**: Run `python -m spacy download en_core_web_sm`
2. **NLTK data missing**: Run the NLTK download commands in setup
3. **Memory errors**: Reduce `max_features` in vectorizers or use smaller datasets
4. **Import errors**: Ensure virtual environment is activated and dependencies are installed

### Performance Optimization

1. **Large datasets**: Use sampling or batch processing
2. **Memory constraints**: Reduce feature dimensions or use sparse matrices
3. **Speed improvements**: Use faster algorithms or parallel processing

## License

This project is for educational purposes. Please respect the licenses of the recommended datasets.

## Contributing

Feel free to:
- Add new NLP tasks
- Improve existing implementations
- Add support for additional datasets
- Enhance visualizations
- Optimize performance

## Contact

For questions or issues, please refer to the project documentation or create an issue in the repository.