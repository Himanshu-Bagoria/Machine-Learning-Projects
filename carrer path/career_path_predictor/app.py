import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from model_trainer import CareerPathModel
from utils import (
    create_probability_chart, 
    create_sankey_diagram, 
    generate_career_explanation,
    create_pdf_report,
    get_career_icon,
    get_career_description,
    prepare_features_for_prediction
)

# Configure page
st.set_page_config(
    page_title="Career Path Predictor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .career-card {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #f8f9fa;
    }
    .prediction-box {
        background-color: #e3f2fd;
        border-left: 5px solid #1f77b4;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .explanation-box {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model"""
    model_path = 'career_model.pkl'
    if not os.path.exists(model_path):
        # Train model if not exists
        st.info("Training model for the first time... This may take a moment.")
        from model_trainer import main as train_model
        train_model()
    
    career_model = CareerPathModel()
    career_model.load_model(model_path)
    return career_model

def main():
    # Load model
    model = load_model()
    
    # Sidebar
    with st.sidebar:
        st.markdown("<h2 style='text-align: center;'>🎯 Career Path Predictor</h2>", unsafe_allow_html=True)
        st.markdown("---")
        
        st.markdown("### About This App")
        st.info("""
        This application predicts your most suitable career paths based on your:
        - **Skills**: Technical and soft skills
        - **Education**: Academic background
        - **Interests**: Areas of personal interest
        
        The predictions are generated using machine learning models trained on career data.
        """)
        
        st.markdown("### How It Works")
        st.success("""
        1. Enter your profile information
        2. Get AI-powered career predictions
        3. View detailed probability analysis
        4. Download your personalized report
        """)
        
        st.markdown("### Career Paths Available")
        career_list = "\n".join([f"• {career}" for career in model.career_paths])
        st.markdown(career_list)
        
        st.markdown("---")
        st.markdown("### 💡 Tips")
        st.warning("""
        - Be honest about your skills and interests
        - The more accurate your input, the better the predictions
        - Use this as a starting point for career exploration
        """)
    
    # Main content
    st.markdown("<h1 class='main-header'>🎯 Career Path Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.2rem; color: #666;'>Discover your ideal career path with AI-powered predictions</p>", unsafe_allow_html=True)
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📋 Input Profile", "📊 Predictions", "📥 Report"])
    
    with tab1:
        st.markdown("<h2 class='sub-header'>Enter Your Profile Information</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎓 Education Level")
            education = st.selectbox(
                "Select your highest education level:",
                ["High School", "Bachelor", "Master", "PhD"],
                help="Choose your highest completed or currently pursuing education level"
            )
            
            st.markdown("### 💼 Skills")
            all_skills = [
                "Python", "Java", "JavaScript", "SQL", "R", "C++", "C#", "HTML", "CSS",
                "React", "Angular", "Vue.js", "Node.js", "Django", "Flask", "Spring",
                "Machine Learning", "Deep Learning", "AI", "NLP", "Computer Vision",
                "Data Analysis", "Statistics", "Excel", "Tableau", "Power BI",
                "AWS", "Azure", "Google Cloud", "Docker", "Kubernetes",
                "Git", "Linux", "Windows", "Agile", "Scrum", "Project Management",
                "Business Analysis", "Financial Analysis", "Marketing", "Digital Marketing",
                "UI/UX Design", "Figma", "Adobe Creative Suite", "Content Writing",
                "SEO", "Social Media", "Cybersecurity", "Ethical Hacking",
                "Mobile Development", "Android", "iOS", "React Native", "Flutter"
            ]
            
            selected_skills = st.multiselect(
                "Select your skills (choose 3-10):",
                all_skills,
                help="Select the skills you possess or are proficient in"
            )
            
            if len(selected_skills) < 3:
                st.warning("Please select at least 3 skills for better predictions")
            elif len(selected_skills) > 10:
                st.warning("Please select no more than 10 skills")
        
        with col2:
            st.markdown("### 🎯 Interests")
            all_interests = [
                "Technology", "Business", "Creative", "Finance", "Healthcare",
                "Education", "Engineering", "Marketing", "Data Science", "AI/ML",
                "Design", "Development", "Analytics", "Research", "Management",
                "Entrepreneurship", "Innovation", "Problem Solving", "Leadership",
                "Communication", "Teamwork", "Project Planning"
            ]
            
            selected_interests = st.multiselect(
                "Select your interests (choose 2-5):",
                all_interests,
                help="Select areas that genuinely interest you"
            )
            
            if len(selected_interests) < 2:
                st.warning("Please select at least 2 interests")
            elif len(selected_interests) > 5:
                st.warning("Please select no more than 5 interests")
            
            st.markdown("### 📊 Prediction Settings")
            model_choice = st.radio(
                "Select prediction model:",
                ["Random Forest (Recommended)", "Logistic Regression"],
                help="Random Forest typically provides more accurate predictions"
            )
            
            show_confidence = st.checkbox("Show confidence intervals", value=True)
    
    # Store user inputs in session state
    st.session_state.user_inputs = {
        'education': education,
        'skills': selected_skills,
        'interests': selected_interests,
        'model': 'random_forest' if 'Random Forest' in model_choice else 'logistic_regression'
    }
    
    with tab2:
        if len(selected_skills) >= 3 and len(selected_interests) >= 2:
            st.markdown("<h2 class='sub-header'>Career Predictions</h2>", unsafe_allow_html=True)
            
            # Prepare features for prediction
            features = prepare_features_for_prediction(
                education, selected_skills, selected_interests, model
            )
            
            # Get predictions
            predictions = model.predict_proba(features, st.session_state.user_inputs['model'])
            
            # Display top predictions
            st.markdown("### 🏆 Top Career Recommendations")
            
            col1, col2, col3 = st.columns(3)
            
            sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
            top_3 = sorted_predictions[:3]
            
            for i, (career, prob) in enumerate(top_3):
                with [col1, col2, col3][i]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3 style="margin: 0; color: #1f77b4;">{get_career_icon(career)} {career}</h3>
                        <h1 style="margin: 10px 0; color: #ff7f0e;">{prob:.1%}</h1>
                        <p style="margin: 0; color: #666;">Match Probability</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Visualization section
            st.markdown("### 📈 Detailed Analysis")
            
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                st.markdown("#### Probability Distribution")
                prob_chart = create_probability_chart(predictions, top_n=8)
                st.plotly_chart(prob_chart, use_container_width=True)
            
            with chart_col2:
                st.markdown("#### Career Path Flow")
                sankey_chart = create_sankey_diagram(
                    education, selected_skills, selected_interests, predictions, top_n=4
                )
                st.plotly_chart(sankey_chart, use_container_width=True)
            
            # Detailed career information
            st.markdown("### 📋 Career Details")
            
            for career, prob in sorted_predictions[:5]:
                with st.expander(f"{get_career_icon(career)} {career} ({prob:.1%})"):
                    st.markdown(f"**Description:** {get_career_description(career)}")
                    st.progress(float(prob))
                    
                    if show_confidence:
                        st.markdown(f"**Confidence Level:** {prob:.1%}")
                        if prob > 0.7:
                            st.success("High confidence match!")
                        elif prob > 0.4:
                            st.info("Moderate confidence match")
                        else:
                            st.warning("Lower confidence - consider exploring related fields")
        
        else:
            st.info("Please complete your profile in the 'Input Profile' tab to see predictions.")
    
    with tab3:
        if len(selected_skills) >= 3 and len(selected_interests) >= 2:
            st.markdown("<h2 class='sub-header'>Personalized Career Report</h2>", unsafe_allow_html=True)
            
            # Generate career explanations
            predictions = model.predict_proba(features, st.session_state.user_inputs['model'])
            sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
            
            career_explanations = {}
            for career, prob in sorted_predictions[:3]:
                explanation = generate_career_explanation(
                    career, education, selected_skills, selected_interests, prob
                )
                career_explanations[career] = explanation
            
            # Display explanations
            st.markdown("### 🎯 Why These Careers Fit You")
            
            for career, explanation in career_explanations.items():
                st.markdown(f"<div class='explanation-box'>{explanation}</div>", unsafe_allow_html=True)
            
            # PDF Download
            st.markdown("### 📥 Download Your Report")
            
            if st.button("Generate PDF Report", type="primary"):
                with st.spinner("Creating your personalized report..."):
                    pdf_bytes = create_pdf_report(
                        st.session_state.user_inputs,
                        predictions,
                        career_explanations
                    )
                    
                    st.download_button(
                        label="Download PDF Report",
                        data=pdf_bytes,
                        file_name="career_path_prediction_report.pdf",
                        mime="application/pdf",
                        type="secondary"
                    )
            
            # Additional resources
            st.markdown("### 📚 Next Steps & Resources")
            st.info("""
            **Recommended Actions:**
            - Research the top 2-3 career paths in detail
            - Identify skill gaps and create learning plans
            - Connect with professionals in your target fields
            - Gain relevant experience through projects or volunteering
            - Consider informational interviews
            
            **Learning Resources:**
            - Online courses (Coursera, Udemy, edX)
            - Professional certifications
            - Industry conferences and meetups
            - Mentorship programs
            """)
            
        else:
            st.info("Please complete your profile in the 'Input Profile' tab to generate a report.")

if __name__ == "__main__":
    main()