import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('stemmers/porter')
except LookupError:
    nltk.download('porter_test')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Set page configuration
st.set_page_config(
    page_title="Spam Classification System",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced appearance
st.markdown("""
<style>
    /* Navigation Bar */
    .navbar {
        position: fixed;
        top: 0;
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        z-index: 999;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .nav-item {
        color: white;
        padding: 10px 20px;
        margin: 0 5px;
        border-radius: 5px;
        cursor: pointer;
        transition: all 0.3s ease;
        font-weight: 500;
    }
    
    .nav-item:hover {
        background-color: rgba(255, 255, 255, 0.2);
        transform: translateY(-2px);
    }
    
    .popup-menu {
        position: absolute;
        top: 50px;
        background: white;
        border-radius: 10px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
        min-width: 200px;
        z-index: 1000;
        opacity: 0;
        visibility: hidden;
        transform: translateY(-10px);
        transition: all 0.3s ease;
    }
    
    .popup-menu.show {
        opacity: 1;
        visibility: visible;
        transform: translateY(0);
    }
    
    .popup-item {
        padding: 12px 20px;
        color: #333;
        text-decoration: none;
        display: block;
        border-bottom: 1px solid #eee;
    }
    
    .popup-item:hover {
        background-color: #f0f2f6;
        color: #667eea;
    }
    
    /* Header styles */
    .main-header {
        font-size: 3rem;
        color: #667eea;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 3px 3px 6px rgba(102, 126, 234, 0.3);
        margin-top: 80px; /* Space for navbar */
        background: linear-gradient(45deg, #667eea, #764ba2, #f093fb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #764ba2;
        text-align: center;
        margin-bottom: 1rem;
        background: linear-gradient(45deg, #f093fb, #f5576c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Card styles */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 16px rgba(102, 126, 234, 0.3);
        color: white;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    /* Prediction styles */
    .prediction-spam {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 16px rgba(255, 107, 107, 0.3);
        color: white;
        border-left: 8px solid #dc3545;
    }
    
    .prediction-ham {
        background: linear-gradient(135deg, #51cf66 0%, #40c057 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 16px rgba(81, 207, 102, 0.3);
        color: white;
        border-left: 8px solid #198754;
    }
    
    /* Button styles */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 25px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(102, 126, 234, 0.4);
    }
    
    /* Input styles */
    .stTextArea>div>textarea {
        border-radius: 10px;
        border: 2px solid #667eea;
        padding: 15px;
    }
    
    /* Sidebar */
    .sidebar-title {
        font-size: 1.2rem;
        font-weight: bold;
        color: #667eea;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    /* Container */
    .main-container {
        margin-top: 80px; /* Space for navbar */
        padding: 20px;
    }
    
    /* Animated background */
    body {
        background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
        height: 100vh;
    }
    
    @keyframes gradientBG {
        0% {
            background-position: 0% 50%;
        }
        50% {
            background-position: 100% 50%;
        }
        100% {
            background-position: 0% 50%;
        }
    }
</style>
<script>
    function togglePopup(menuId) {
        const menu = document.getElementById(menuId);
        menu.classList.toggle('show');
    }
    
    // Close popup when clicking outside
    document.addEventListener('click', function(event) {
        const popups = document.querySelectorAll('.popup-menu');
        popups.forEach(function(popup) {
            if (!event.target.closest('.nav-item')) {
                popup.classList.remove('show');
            }
        });
    });
    
    // Update URL when clicking on navigation items
    function navigateTo(page) {
        // Update the URL hash to trigger navigation
        window.location.hash = page;
        window.location.reload();
    }
</script>
""", unsafe_allow_html=True)

# Navigation Bar HTML
st.markdown("""
<div class="navbar">
    <div style="display: flex; align-items: center;">
        <h2 style="color: white; margin: 0; font-size: 1.5rem;">📧 Spam Classifier</h2>
    </div>
    <div style="display: flex; gap: 10px;">
        <div class="nav-item" onclick="togglePopup('home-popup')">🏠 Home</div>
        <div class="nav-item" onclick="togglePopup('explore-popup')">📊 Explore</div>
        <div class="nav-item" onclick="togglePopup('train-popup')">🧠 Train</div>
        <div class="nav-item" onclick="togglePopup('predict-popup')">🔮 Predict</div>
    </div>
</div>

<!-- Popup Menus -->
<div id="home-popup" class="popup-menu">
    <a href="#" onclick="navigateTo('home')" class="popup-item">🏠 Home</a>
    <a href="#" onclick="navigateTo('about')" class="popup-item">ℹ️ About</a>
    <a href="#" onclick="navigateTo('features')" class="popup-item">✨ Features</a>
</div>

<div id="explore-popup" class="popup-menu">
    <a href="#" onclick="navigateTo('exploration')" class="popup-item">📈 Data Exploration</a>
    <a href="#" onclick="navigateTo('visualizations')" class="popup-item">📊 Visualizations</a>
    <a href="#" onclick="navigateTo('statistics')" class="popup-item">📊 Statistics</a>
</div>

<div id="train-popup" class="popup-menu">
    <a href="#" onclick="navigateTo('training')" class="popup-item">🤖 Train Models</a>
    <a href="#" onclick="navigateTo('evaluation')" class="popup-item">📋 Evaluate</a>
    <a href="#" onclick="navigateTo('comparison')" class="popup-item">📊 Compare Models</a>
</div>

<div id="predict-popup" class="popup-menu">
    <a href="#" onclick="navigateTo('realtime')" class="popup-item">⚡ Real-time Predict</a>
    <a href="#" onclick="navigateTo('batch')" class="popup-item">📦 Batch Predict</a>
    <a href="#" onclick="navigateTo('analysis')" class="popup-item">🔍 Analysis</a>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = None
if 'classifier' not in st.session_state:
    st.session_state.classifier = None
if 'accuracy' not in st.session_state:
    st.session_state.accuracy = 0
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Home"

class SpamClassifier:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
    def preprocess_text(self, text):
        """Preprocess text: lowercase, tokenize, remove stopwords, stem"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and stem
        filtered_tokens = []
        for token in tokens:
            if token not in self.stop_words:
                stemmed_token = self.stemmer.stem(token)
                filtered_tokens.append(stemmed_token)
        
        return ' '.join(filtered_tokens)
    
    def prepare_data(self, texts):
        """Prepare data by preprocessing"""
        processed_texts = []
        for text in texts:
            processed_text = self.preprocess_text(text)
            processed_texts.append(processed_text)
        return processed_texts

def load_sample_data():
    """Load sample data for demonstration"""
    # Sample data - in a real scenario, you'd load from a CSV file
    sample_data = {
        'text': [
            "Congratulations! You've won $1000! Click here to claim now!",
            "Hey, are we still meeting for lunch tomorrow?",
            "FREE MONEY! Act now! Limited time offer!",
            "Can you send me the report by end of day?",
            "Win big prizes! Call now!",
            "Thanks for the great presentation today",
            "URGENT: Your account will be suspended!",
            "Let's schedule a meeting for next week",
            "Get rich quick! Make money fast!",
            "The meeting has been moved to 3 PM"
        ],
        'label': ['spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham']
    }
    df = pd.DataFrame(sample_data)
    return df

def train_models(df):
    """Train multiple models and return the best one"""
    classifier = SpamClassifier()
    
    # Preprocess the text data
    processed_texts = classifier.prepare_data(df['text'].tolist())
    
    # Prepare features and labels
    X = processed_texts
    y = df['label'].map({'spam': 1, 'ham': 0})  # 1 for spam, 0 for ham
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Create and train different models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Naive Bayes': MultinomialNB(),
        'SVM': SVC(kernel='linear', random_state=42)
    }
    
    best_model = None
    best_accuracy = 0
    best_name = ""
    
    results = {}
    
    for name, model in models.items():
        # Create pipeline with TF-IDF vectorizer
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('classifier', model)
        ])
        
        # Train the model
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = pipeline.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = {
            'pipeline': pipeline,
            'accuracy': accuracy,
            'y_test': y_test,
            'y_pred': y_pred
        }
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = pipeline
            best_name = name
    
    return results, best_model, best_name, best_accuracy, X_test, y_test, y_pred

def visualize_results(results, X_test, y_test, y_pred):
    """Create visualizations for model performance"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Model Comparison', 'Confusion Matrix', 'Text Length Distribution', 'Prediction Distribution'),
        specs=[[{"type": "bar"}, {"type": "heatmap"}],
               [{"type": "histogram"}, {"type": "pie"}]]
    )
    
    # Model comparison
    model_names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in model_names]
    
    fig.add_trace(
        go.Bar(x=model_names, y=accuracies, name='Accuracy'),
        row=1, col=1
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig.add_trace(
        go.Heatmap(z=cm, x=['Ham', 'Spam'], y=['Ham', 'Spam'], colorscale='Blues', showscale=False),
        row=1, col=2
    )
    
    # Text length distribution
    text_lengths = [len(text) for text in X_test]
    fig.add_trace(
        go.Histogram(x=text_lengths, nbinsx=20, name='Text Length'),
        row=2, col=1
    )
    
    # Prediction distribution
    prediction_counts = pd.Series(y_pred).value_counts()
    fig.add_trace(
        go.Pie(labels=['Ham', 'Spam'], values=prediction_counts.values, name='Predictions'),
        row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=False, title_text="Model Performance Analysis")
    return fig

def main():
    st.markdown('<h1 class="main-header">📧 Spam Classification System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Machine Learning Application for Email/SMS Spam Detection</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown('<div class="sidebar-title">Navigation</div>', unsafe_allow_html=True)
    app_mode = st.sidebar.selectbox("Choose the app mode",
                                   ["Home", "Data Exploration", "Model Training", "Real-time Prediction"])
    
    if app_mode == "Home":
        st.header("Welcome to Spam Classification System")
        st.write("""
        This application uses Natural Language Processing (NLP) and Machine Learning techniques to classify emails and SMS messages as spam or authentic.
        
        **Features:**
        - Advanced text preprocessing with NLTK
        - TF-IDF vectorization
        - Multiple ML algorithms for comparison
        - Real-time prediction capabilities
        - Interactive visualizations
        
        **Technologies Used:**
        - Python, Streamlit, Scikit-learn
        - NLTK for NLP preprocessing
        - Matplotlib/Seaborn for visualization
        """)
        
        # Display sample data
        st.subheader("Sample Data Preview")
        df = load_sample_data()
        st.dataframe(df.head(10))
        
        # Show preprocessing example
        st.subheader("Text Preprocessing Example")
        sample_text = "Congratulations! You've won $1000! Click here to claim now!"
        classifier = SpamClassifier()
        processed_text = classifier.preprocess_text(sample_text)
        st.write(f"**Original:** {sample_text}")
        st.write(f"**Processed:** {processed_text}")
    
    elif app_mode == "Data Exploration":
        st.header("Data Exploration & Visualization")
        
        # Load data
        df = load_sample_data()
        
        # Display basic statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(label="Total Messages", value=len(df))
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(label="Spam Messages", value=len(df[df['label'] == 'spam']))
            st.markdown('</div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(label="Ham Messages", value=len(df[df['label'] == 'ham']))
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Label distribution
        st.subheader("Label Distribution")
        label_counts = df['label'].value_counts()
        fig = px.pie(values=label_counts.values, names=label_counts.index, title="Distribution of Spam vs Ham")
        st.plotly_chart(fig, use_container_width=True)
        
        # Text length analysis
        st.subheader("Text Length Analysis")
        df['text_length'] = df['text'].apply(len)
        fig = px.histogram(df, x='text_length', color='label', title="Distribution of Text Lengths")
        st.plotly_chart(fig, use_container_width=True)
        
        # Word cloud simulation (using most common words)
        st.subheader("Most Common Words in Spam vs Ham")
        df_spam = df[df['label'] == 'spam']
        df_ham = df[df['label'] == 'ham']
        
        # Simple word frequency analysis
        all_spam_words = ' '.join(df_spam['text']).lower()
        all_ham_words = ' '.join(df_ham['text']).lower()
        
        spam_words = all_spam_words.split()
        ham_words = all_ham_words.split()
        
        spam_freq = pd.Series(spam_words).value_counts().head(10)
        ham_freq = pd.Series(ham_words).value_counts().head(10)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Top Words in Spam:**")
            st.bar_chart(spam_freq)
        with col2:
            st.write("**Top Words in Ham:**")
            st.bar_chart(ham_freq)
    
    elif app_mode == "Model Training":
        st.header("Model Training & Evaluation")
        
        if st.button("Train Models"):
            with st.spinner("Training models... This may take a moment."):
                df = load_sample_data()
                results, best_model, best_name, best_accuracy, X_test, y_test, y_pred = train_models(df)
                
                # Store in session state
                st.session_state.model_trained = True
                st.session_state.best_model = best_model
                st.session_state.best_name = best_name
                st.session_state.accuracy = best_accuracy
                
                st.success(f"Best Model: {best_name} with Accuracy: {best_accuracy:.4f}")
                
                # Display model comparison
                st.subheader("Model Comparison")
                comparison_df = pd.DataFrame({
                    'Model': list(results.keys()),
                    'Accuracy': [results[name]['accuracy'] for name in results.keys()]
                })
                st.dataframe(comparison_df.style.highlight_max(axis=0))
                
                # Detailed metrics for best model
                st.subheader(f"Detailed Metrics for {best_name}")
                best_result = results[best_name]
                st.text(f"Classification Report:\n{classification_report(best_result['y_test'], best_result['y_pred'])}")
                
                # Visualizations
                st.subheader("Performance Visualizations")
                fig = visualize_results(results, X_test, y_test, y_pred)
                st.plotly_chart(fig, use_container_width=True)
        
        # Show training status
        if st.session_state.model_trained:
            st.info(f"✅ Model Trained: {st.session_state.best_name} with Accuracy: {st.session_state.accuracy:.4f}")
    
    elif app_mode == "Real-time Prediction":
        st.header("Real-time Spam Detection")
        
        if not st.session_state.model_trained:
            st.warning("⚠️ Please train the model first in the 'Model Training' section.")
            return
        
        # Input for prediction
        user_input = st.text_area("Enter email/SMS text for spam detection:", height=150)
        
        if st.button("Predict"):
            if user_input.strip() != "":
                with st.spinner("Analyzing..."):
                    classifier = SpamClassifier()
                    processed_input = classifier.preprocess_text(user_input)
                    
                    # Make prediction
                    prediction = st.session_state.best_model.predict([processed_input])[0]
                    probability = st.session_state.best_model.predict_proba([processed_input])[0]
                    
                    # Display result
                    if prediction == 1:  # Spam
                        st.markdown('<div class="prediction-spam">', unsafe_allow_html=True)
                        st.subheader("🚨 SPAM DETECTED!")
                        st.write(f"**Confidence:** {max(probability):.4f}")
                        st.write(f"**Probability of Spam:** {probability[1]:.4f}")
                        st.write(f"**Probability of Ham:** {probability[0]:.4f}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:  # Ham
                        st.markdown('<div class="prediction-ham">', unsafe_allow_html=True)
                        st.subheader("✅ AUTHENTIC MESSAGE")
                        st.write(f"**Confidence:** {max(probability):.4f}")
                        st.write(f"**Probability of Spam:** {probability[1]:.4f}")
                        st.write(f"**Probability of Ham:** {probability[0]:.4f}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Show preprocessing
                    st.subheader("Preprocessing Result")
                    st.write(f"**Original:** {user_input}")
                    st.write(f"**Processed:** {processed_input}")
            else:
                st.warning("Please enter some text for prediction.")
        
        # Batch prediction example
        st.subheader("Batch Prediction Example")
        sample_messages = [
            "Congratulations! You've won $1000! Click here to claim now!",
            "Hey, are we still meeting for lunch tomorrow?",
            "FREE MONEY! Act now! Limited time offer!"
        ]
        
        if st.button("Run Batch Prediction"):
            classifier = SpamClassifier()
            processed_inputs = classifier.prepare_data(sample_messages)
            
            predictions = st.session_state.best_model.predict(processed_inputs)
            probabilities = st.session_state.best_model.predict_proba(processed_inputs)
            
            results_df = pd.DataFrame({
                'Message': sample_messages,
                'Prediction': ['SPAM' if pred == 1 else 'HAM' for pred in predictions],
                'Spam Probability': [prob[1] for prob in probabilities],
                'Ham Probability': [prob[0] for prob in probabilities]
            })
            
            st.dataframe(results_df)

if __name__ == "__main__":
    main()