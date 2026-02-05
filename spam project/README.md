# Spam Classification System

A machine learning application built with Python that classifies emails and SMS messages as spam or authentic using NLP techniques and various classification algorithms.

## Features

- **Natural Language Processing (NLP)**: Tokenization, stop-word removal, and stemming using NLTK
- **TF-IDF Vectorization**: Converting text to numerical features for machine learning
- **Multiple ML Algorithms**: Logistic Regression, Naive Bayes, SVM, and Random Forest
- **Interactive Visualization**: Using Matplotlib, Seaborn, and Plotly
- **Real-time Prediction**: Streamlit web application for live spam detection
- **Model Evaluation**: Comprehensive metrics and performance comparison

## Installation

1. Clone or download this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit application:

```bash
streamlit run main.py
```

## Project Structure

```
spam-classification/
│
├── main.py                 # Main Streamlit application
├── preprocessing.py        # Text preprocessing utilities
├── model.py               # ML model training and evaluation
├── requirements.txt       # Python dependencies
└── README.md             # Project documentation
```

## How It Works

### 1. Text Preprocessing
- Converts text to lowercase
- Removes special characters, URLs, and punctuation
- Tokenizes the text
- Removes stop words
- Applies stemming using Porter Stemmer

### 2. Feature Engineering
- Uses TF-IDF (Term Frequency-Inverse Document Frequency) to convert text to numerical vectors
- Employs n-grams (1-2) to capture phrases and word combinations

### 3. Model Training
- Compares multiple algorithms: Logistic Regression, Naive Bayes, SVM, and Random Forest
- Evaluates models using accuracy, precision, recall, and F1-score
- Selects the best performing model for deployment

### 4. Real-time Prediction
- Provides an interactive web interface for spam detection
- Shows prediction confidence and probabilities
- Supports both single and batch predictions

## Usage

1. Run the application:
```bash
streamlit run main.py
```

2. Navigate through the different sections:
   - **Home**: Overview of the application
   - **Data Exploration**: Analyze the dataset and distributions
   - **Model Training**: Train and evaluate ML models
   - **Real-time Prediction**: Test the model with custom inputs

3. In the "Real-time Prediction" section:
   - Enter an email/SMS message
   - Click "Predict" to see if it's classified as spam or ham (authentic)
   - View the preprocessing steps and prediction confidence

## Technologies Used

- **Python**: Core programming language
- **Streamlit**: Web application framework
- **Scikit-learn**: Machine learning algorithms
- **NLTK**: Natural language processing
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Matplotlib/Seaborn/Plotly**: Data visualization
- **Joblib**: Model serialization

## Algorithms Implemented

- **Logistic Regression**: Linear model for binary classification
- **Multinomial Naive Bayes**: Probabilistic classifier based on Bayes' theorem
- **Support Vector Machine (SVM)**: Linear kernel for classification
- **Random Forest**: Ensemble method using decision trees

## Dataset

The application includes a sample dataset with labeled examples of spam and ham messages. For production use, you can replace this with a larger, more comprehensive dataset.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is open source and available under the MIT License.