import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline

class CareerPathModel:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.education_encoder = LabelEncoder()
        self.skills_vectorizer = TfidfVectorizer()
        self.interests_vectorizer = TfidfVectorizer()
        self.models = {}
        self.career_paths = []
        
    def load_data(self, data_path):
        """Load career data from CSV file"""
        df = pd.read_csv(data_path)
        return df
    
    def preprocess_data(self, df):
        """Preprocess the data for training"""
        # Encode categorical variables
        df['education_encoded'] = self.education_encoder.fit_transform(df['education'])
        
        # Process skills (convert to TF-IDF features)
        skills_features = self.skills_vectorizer.fit_transform(df['skills'])
        skills_df = pd.DataFrame(skills_features.toarray(), 
                               columns=[f"skill_{i}" for i in range(skills_features.shape[1])])
        
        # Process interests (convert to TF-IDF features)
        interests_features = self.interests_vectorizer.fit_transform(df['interests'])
        interests_df = pd.DataFrame(interests_features.toarray(), 
                                  columns=[f"interest_{i}" for i in range(interests_features.shape[1])])
        
        # Combine all features
        feature_columns = (['education_encoded'] + 
                          list(skills_df.columns) + 
                          list(interests_df.columns))
        
        X = pd.concat([df[['education_encoded']], skills_df, interests_df], axis=1)
        y = self.label_encoder.fit_transform(df['career_path'])
        
        self.career_paths = self.label_encoder.classes_
        
        return X, y, feature_columns
    
    def train_models(self, X, y):
        """Train multiple ML models"""
        # Split data - use simple split for this dataset size
        if len(y) > 50:  # Use stratification only for larger datasets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        else:  # Simple split for small dataset
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )
        
        # Logistic Regression Model
        print("Training Logistic Regression...")
        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        lr_model.fit(X_train, y_train)
        
        # Random Forest Model
        print("Training Random Forest...")
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        
        # Store models and performance
        self.models['logistic_regression'] = {
            'model': lr_model,
            'accuracy': accuracy_score(y_test, lr_model.predict(X_test))
        }
        
        self.models['random_forest'] = {
            'model': rf_model,
            'accuracy': accuracy_score(y_test, rf_model.predict(X_test))
        }
        
        print("Model Training Complete!")
        print(f"Logistic Regression Accuracy: {self.models['logistic_regression']['accuracy']:.3f}")
        print(f"Random Forest Accuracy: {self.models['random_forest']['accuracy']:.3f}")
        
        return X_test, y_test
    
    def predict_proba(self, features, model_name='random_forest'):
        """Get prediction probabilities for given features"""
        model = self.models[model_name]['model']
        probabilities = model.predict_proba([features])[0]
        return dict(zip(self.career_paths, probabilities))
    
    def save_model(self, filepath):
        """Save the trained model and encoders"""
        model_data = {
            'models': self.models,
            'label_encoder': self.label_encoder,
            'education_encoder': self.education_encoder,
            'skills_vectorizer': self.skills_vectorizer,
            'interests_vectorizer': self.interests_vectorizer,
            'career_paths': self.career_paths
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        if os.path.exists(filepath):
            model_data = joblib.load(filepath)
            self.models = model_data['models']
            self.label_encoder = model_data['label_encoder']
            self.education_encoder = model_data['education_encoder']
            self.skills_vectorizer = model_data['skills_vectorizer']
            self.interests_vectorizer = model_data['interests_vectorizer']
            self.career_paths = model_data['career_paths']
            print(f"Model loaded from {filepath}")
            return True
        return False

def main():
    """Main function to train and save the model"""
    # Create model instance
    career_model = CareerPathModel()
    
    # Load data
    data_path = os.path.join('data', 'career_data.csv')
    df = career_model.load_data(data_path)
    
    # Preprocess data
    X, y, feature_columns = career_model.preprocess_data(df)
    
    # Train models
    X_test, y_test = career_model.train_models(X, y)
    
    # Save model
    model_path = 'career_model.pkl'
    career_model.save_model(model_path)
    
    print("\nTraining completed successfully!")
    print(f"Available career paths: {list(career_model.career_paths)}")

if __name__ == "__main__":
    main()