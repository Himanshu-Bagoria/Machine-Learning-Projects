"""
Demo script for the Spam Classification System
This demonstrates how to use the model and preprocessing components
"""

from preprocessing import TextPreprocessor
from model import SpamDetectionModel, load_sample_data
import pandas as pd

def demo_preprocessing():
    """Demonstrate the text preprocessing capabilities"""
    print("=== TEXT PREPROCESSING DEMO ===")
    preprocessor = TextPreprocessor()
    
    sample_texts = [
        "Congratulations! You've won $1000! Click here to claim now! http://bit.ly/freemoney",
        "Hey, are we still meeting for lunch tomorrow?",
        "FREE MONEY! Act now! Limited time offer! Call now!!!"
    ]
    
    print("\nOriginal texts:")
    for i, text in enumerate(sample_texts):
        print(f"{i+1}. {text}")
    
    print("\nProcessed texts:")
    processed_texts = preprocessor.preprocess_batch(sample_texts)
    for i, text in enumerate(processed_texts):
        print(f"{i+1}. {text}")
    
    print()

def demo_model_training():
    """Demonstrate the model training and evaluation"""
    print("=== MODEL TRAINING DEMO ===")
    
    # Load sample data
    df = load_sample_data()
    print(f"Loaded {len(df)} samples")
    print(f"Spam count: {sum(df['label'] == 'spam')}, Ham count: {sum(df['label'] == 'ham')}")
    
    # Initialize model
    spam_model = SpamDetectionModel()
    
    # Prepare data
    X, y = spam_model.prepare_data(df)
    
    # Train models
    print("\nTraining models...")
    results, best_model_name = spam_model.train_models(X, y)
    
    # Print results
    print(f"\nBest model: {best_model_name}")
    print("All model performances:")
    for name, result in results.items():
        print(f"  {name}: Accuracy = {result['accuracy']:.4f}, F1-Score = {result['f1_score']:.4f}")
    
    print()

def demo_predictions():
    """Demonstrate the prediction capabilities"""
    print("=== PREDICTION DEMO ===")
    
    # First, train a model
    df = load_sample_data()
    spam_model = SpamDetectionModel()
    X, y = spam_model.prepare_data(df)
    results, best_model_name = spam_model.train_models(X, y)
    
    # Test prediction
    test_texts = [
        "Free money! Click here now!",
        "Can we reschedule our meeting?",
        "CONGRATULATIONS! You are our lucky winner! Claim your prize!",
        "Please review the attached document"
    ]
    
    print(f"Using best model: {best_model_name}")
    print("\nTest predictions:")
    predictions = spam_model.predict(test_texts)
    for i, text in enumerate(test_texts):
        print(f"Text: {text}")
        print(f"Prediction: {predictions['labels'][i]}")
        print(f"Spam Probability: {predictions['probabilities'][i][1]:.4f}")
        print(f"Ham Probability: {predictions['probabilities'][i][0]:.4f}")
        print("---")

def main():
    """Main demo function"""
    print("SPAM CLASSIFICATION SYSTEM - DEMONSTRATION")
    print("="*50)
    
    demo_preprocessing()
    demo_model_training()
    demo_predictions()
    
    print("\nThe demo has completed!")
    print("\nTo run the full web application, execute:")
    print("  streamlit run main.py")

if __name__ == "__main__":
    main()