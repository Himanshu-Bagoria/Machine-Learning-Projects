import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string

class TextPreprocessor:
    def __init__(self):
        # Download required NLTK data if not present
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
        
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        """
        Clean text by removing special characters and extra whitespaces
        """
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove user @ references and '#' from tweet
        text = re.sub(r'@\w+|#', '', text)
        # Remove punctuations and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Remove extra whitespaces
        text = ' '.join(text.split())
        return text
    
    def preprocess_text(self, text):
        """
        Complete preprocessing pipeline:
        1. Clean text
        2. Tokenize
        3. Remove stopwords
        4. Stem words
        """
        # Clean the text
        cleaned_text = self.clean_text(text)
        
        # Convert to lowercase
        cleaned_text = cleaned_text.lower()
        
        # Tokenize
        tokens = word_tokenize(cleaned_text)
        
        # Remove stopwords and apply stemming
        processed_tokens = []
        for token in tokens:
            if token not in self.stop_words and token not in string.punctuation:
                stemmed_token = self.stemmer.stem(token)
                processed_tokens.append(stemmed_token)
        
        return ' '.join(processed_tokens)
    
    def preprocess_batch(self, texts):
        """
        Preprocess a batch of texts
        """
        processed_texts = []
        for text in texts:
            processed_text = self.preprocess_text(text)
            processed_texts.append(processed_text)
        return processed_texts

# Example usage
if __name__ == "__main__":
    preprocessor = TextPreprocessor()
    
    sample_texts = [
        "Congratulations! You've won $1000! Click here to claim now! http://bit.ly/freemoney",
        "Hey, are we still meeting for lunch tomorrow?",
        "FREE MONEY! Act now! Limited time offer! Call now!!!"
    ]
    
    print("Original texts:")
    for i, text in enumerate(sample_texts):
        print(f"{i+1}. {text}")
    
    print("\nProcessed texts:")
    processed_texts = preprocessor.preprocess_batch(sample_texts)
    for i, text in enumerate(processed_texts):
        print(f"{i+1}. {text}")