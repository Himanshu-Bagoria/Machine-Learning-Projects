**🎯 Career Path Predictor
**An AI-powered Streamlit application that predicts possible career paths based on skills, education, and interests.
The app uses machine learning models to generate probability scores for different career options and visualizes transitions with an interactive Sankey diagram.

**🌐 Live Demo
**👉 https://career-path-predictor.onrender.com

**🚀 Features
**- User Input: Enter skills, education level, and interests.
- ML Predictions: Suggests career paths with probability scores.
- Interactive Visualization: Sankey diagram showing transitions from education → skills → career outcomes.
- Personalized Report: Option to download career prediction as PDF.
- Recruiter-Friendly UI: Clean design with modern Streamlit components.

**🛠️ Technologies Used
**- Python: Core programming language
- Streamlit: Web application framework
- Scikit-learn: Machine learning algorithms
- Plotly: Sankey diagram visualization
- Pandas / NumPy: Data manipulation and numerical computing

**📂 Project Structur**

career-path-predictor/
│
├── app.py                # Main Streamlit application
├── model.py              # ML model training and prediction
├── preprocessing.py      # Input preprocessing utilities
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation

**🔎 How It Works**
- Input Collection
- User provides skills, education level, and interests.
- Feature Engineering
- Encodes inputs into numerical vectors for ML processing.
- Model Training & Prediction
- Trains ML models (Logistic Regression, Random Forest, etc.).
- Generates probability scores for career paths.
- Visualization
- Displays results with bar charts and Sankey diagram.
- Report Generation
- Allows users to download a personalized career prediction report.
