import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import joblib
import warnings
warnings.filterwarnings('ignore')

class SpamDetectionModel:
    def __init__(self):
        self.models = {
            'logistic_regression': LogisticRegression(random_state=42),
            'naive_bayes': MultinomialNB(),
            'svm': SVC(kernel='linear', random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        self.best_model = None
        self.best_model_name = None
        self.vectorizer = None
        self.is_trained = False
    
    def prepare_data(self, df, text_column='text', label_column='label'):
        """
        Prepare data for training by separating features and labels
        """
        X = df[text_column].values
        y = df[label_column].map({'spam': 1, 'ham': 0}).values  # Convert to binary
        return X, y
    
    def train_models(self, X, y, test_size=0.2):
        """
        Train multiple models and return the best performing one
        """
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
        
        # Initialize results dictionary
        results = {}
        
        # Train each model
        for name, model in self.models.items():
            # Create pipeline with TF-IDF vectorizer
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')),
                ('classifier', model)
            ])
            
            # Train the model
            pipeline.fit(X_train, y_train)
            
            # Make predictions
            y_pred = pipeline.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Store results
            results[name] = {
                'pipeline': pipeline,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'y_test': y_test,
                'y_pred': y_pred,
                'X_test': X_test
            }
        
        # Find the best model based on accuracy
        best_model_name = max(results, key=lambda x: results[x]['accuracy'])
        self.best_model = results[best_model_name]['pipeline']
        self.best_model_name = best_model_name
        self.is_trained = True
        
        return results, best_model_name
    
    def evaluate_model(self, model_name, results):
        """
        Get detailed evaluation of a specific model
        """
        if model_name in results:
            result = results[model_name]
            return {
                'model_name': model_name,
                'accuracy': result['accuracy'],
                'precision': result['precision'],
                'recall': result['recall'],
                'f1_score': result['f1_score'],
                'classification_report': classification_report(result['y_test'], result['y_pred']),
                'confusion_matrix': confusion_matrix(result['y_test'], result['y_pred'])
            }
        else:
            return None
    
    def predict(self, texts):
        """
        Predict spam/ham for given texts
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if isinstance(texts, str):
            texts = [texts]
        
        predictions = self.best_model.predict(texts)
        probabilities = self.best_model.predict_proba(texts)
        
        # Convert predictions to labels
        labels = ['ham' if pred == 0 else 'spam' for pred in predictions]
        
        return {
            'labels': labels,
            'predictions': predictions,
            'probabilities': probabilities
        }
    
    def save_model(self, filepath):
        """
        Save the trained model to disk
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        joblib.dump({
            'model': self.best_model,
            'model_name': self.best_model_name
        }, filepath)
    
    def load_model(self, filepath):
        """
        Load a trained model from disk
        """
        loaded = joblib.load(filepath)
        self.best_model = loaded['model']
        self.best_model_name = loaded['model_name']
        self.is_trained = True

def load_sample_data():
    """
    Load sample data for demonstration
    In a real scenario, you would load from a CSV file with columns 'text' and 'label'
    """
    data = {
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
            "The meeting has been moved to 3 PM",
            "Your Amazon package is on the way!",
            "Meeting reminder: Quarterly review at 2 PM",
            "You have won a lottery! Send your bank details now!",
            "Can you pick up milk on your way home?",
            "Click here for amazing deals!",
            "See you at the conference next week",
            "Your credit card has been charged $500",
            "Happy birthday! Hope you have a great day!",
            "Limited time offer! Act now!",
            "Please find the attachment for your review"
        ],
        'label': [
            'spam', 'ham', 'spam', 'ham', 'spam',
            'ham', 'spam', 'ham', 'spam', 'ham',
            'ham', 'ham', 'spam', 'ham', 'spam',
            'ham', 'spam', 'ham', 'spam', 'ham'
        ]
    }
    return pd.DataFrame(data)

if __name__ == "__main__":
    # Example usage
    print("Loading sample data...")
    df = load_sample_data()
    print(f"Loaded {len(df)} samples")
    
    # Initialize model
    spam_model = SpamDetectionModel()
    
    # Prepare data
    X, y = spam_model.prepare_data(df)
    
    # Train models
    print("Training models...")
    results, best_model_name = spam_model.train_models(X, y)
    
    # Print results
    print(f"\nBest model: {best_model_name}")
    print("All model performances:")
    for name, result in results.items():
        print(f"{name}: Accuracy = {result['accuracy']:.4f}")
    
    # Test prediction
    test_texts = [
        "Free money! Click here now!",
        "Can we reschedule our meeting?"
    ]
    
    predictions = spam_model.predict(test_texts)
    print(f"\nTest predictions:")
    for i, text in enumerate(test_texts):
        print(f"Text: {text}")
        print(f"Prediction: {predictions['labels'][i]}")
        print(f"Probability: {predictions['probabilities'][i]}")
        print("---")