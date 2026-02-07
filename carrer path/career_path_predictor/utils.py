import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import base64
from io import BytesIO
from fpdf import FPDF
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

# Career path icons and descriptions
CAREER_INFO = {
    'Data Scientist': {
        'icon': '🔬',
        'description': 'Analyze complex data to extract insights and build predictive models',
        'why_text': 'Your combination of analytical skills and technology interest aligns perfectly with data-driven decision making.'
    },
    'Web Developer': {
        'icon': '🌐',
        'description': 'Build and maintain websites and web applications',
        'why_text': 'Your technical skills and creative approach make you well-suited for building digital experiences.'
    },
    'Software Engineer': {
        'icon': '💻',
        'description': 'Design, develop, and maintain software applications and systems',
        'why_text': 'Your programming skills and systematic approach are ideal for software development.'
    },
    'Business Analyst': {
        'icon': '📊',
        'description': 'Analyze business processes and recommend solutions to improve efficiency',
        'why_text': 'Your analytical mindset and business interest make you perfect for bridging technical and business needs.'
    },
    'Marketing Manager': {
        'icon': '📈',
        'description': 'Plan and execute marketing strategies to promote products and services',
        'why_text': 'Your creative and strategic thinking aligns well with market-driven decision making.'
    },
    'AI Engineer': {
        'icon': '🤖',
        'description': 'Develop and implement artificial intelligence and machine learning solutions',
        'why_text': 'Your advanced technical skills and interest in cutting-edge technology make AI engineering ideal.'
    },
    'UI/UX Designer': {
        'icon': '🎨',
        'description': 'Create user interfaces and experiences for digital products',
        'why_text': 'Your creative skills and user-focused approach make design your natural calling.'
    },
    'Content Strategist': {
        'icon': '📝',
        'description': 'Develop and manage content strategy across various platforms',
        'why_text': 'Your communication skills and strategic thinking are perfect for content development.'
    },
    'Financial Analyst': {
        'icon': '💰',
        'description': 'Analyze financial data and provide investment recommendations',
        'why_text': 'Your analytical skills and interest in numbers make financial analysis a strong fit.'
    },
    'Project Manager': {
        'icon': '📋',
        'description': 'Plan, execute, and oversee projects from conception to completion',
        'why_text': 'Your organizational skills and leadership abilities are perfect for project management.'
    },
    'Cybersecurity Analyst': {
        'icon': '🛡️',
        'description': 'Protect systems and networks from security threats and attacks',
        'why_text': 'Your technical expertise and attention to detail make cybersecurity your ideal domain.'
    },
    'Mobile Developer': {
        'icon': '📱',
        'description': 'Create applications for mobile devices and platforms',
        'why_text': 'Your programming skills and user-focused approach are perfect for mobile development.'
    },
    'Cloud Engineer': {
        'icon': '☁️',
        'description': 'Design and manage cloud computing infrastructure and services',
        'why_text': 'Your technical infrastructure skills and systematic approach align well with cloud systems.'
    },
    'Digital Marketing Specialist': {
        'icon': '🎯',
        'description': 'Execute digital marketing campaigns across online channels',
        'why_text': 'Your analytical and creative skills make you well-suited for data-driven marketing.'
    }
}

def create_probability_chart(predictions, top_n=5):
    """Create a bar chart showing career probabilities"""
    # Sort predictions by probability
    sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    top_predictions = sorted_predictions[:top_n]
    
    careers = [item[0] for item in top_predictions]
    probabilities = [item[1] for item in top_predictions]
    
    # Create interactive plotly chart
    fig = go.Figure(data=[
        go.Bar(
            x=probabilities,
            y=careers,
            orientation='h',
            marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(careers)],
            text=[f'{p:.1%}' for p in probabilities],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title='Top Career Path Predictions',
        xaxis_title='Probability',
        yaxis_title='Career Path',
        height=400,
        xaxis=dict(tickformat='.0%', range=[0, 1])
    )
    
    return fig

def create_sankey_diagram(education, skills, interests, predictions, top_n=3):
    """Create a Sankey diagram showing the flow from inputs to career paths"""
    # Get top predictions
    sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    top_predictions = sorted_predictions[:top_n]
    
    # Create nodes
    nodes = []
    node_mapping = {}
    
    # Education node
    nodes.append(education)
    node_mapping[education] = 0
    
    # Skills nodes
    skill_list = skills.split(',') if isinstance(skills, str) else skills
    skill_list = [s.strip() for s in skill_list][:3]  # Top 3 skills
    for i, skill in enumerate(skill_list):
        nodes.append(skill)
        node_mapping[skill] = i + 1
    
    # Interests nodes
    interest_list = interests.split(',') if isinstance(interests, str) else interests
    interest_list = [i.strip() for i in interest_list][:2]  # Top 2 interests
    start_idx = len(skill_list) + 1
    for i, interest in enumerate(interest_list):
        nodes.append(interest)
        node_mapping[interest] = start_idx + i
    
    # Career nodes
    start_idx = len(skill_list) + len(interest_list) + 1
    for i, (career, prob) in enumerate(top_predictions):
        nodes.append(career)
        node_mapping[career] = start_idx + i
    
    # Create links
    links = []
    
    # Education to skills
    for skill in skill_list:
        links.append({
            'source': node_mapping[education],
            'target': node_mapping[skill],
            'value': 0.3
        })
    
    # Skills to interests
    for skill in skill_list:
        for interest in interest_list:
            links.append({
                'source': node_mapping[skill],
                'target': node_mapping[interest],
                'value': 0.2
            })
    
    # Interests to careers
    for interest in interest_list:
        for career, prob in top_predictions:
            links.append({
                'source': node_mapping[interest],
                'target': node_mapping[career],
                'value': prob * 0.5
            })
    
    # Create Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=nodes,
            color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", 
                   "#8c564b", "#e377c2", "#7f7f7f"][:len(nodes)]
        ),
        link=dict(
            source=[link['source'] for link in links],
            target=[link['target'] for link in links],
            value=[link['value'] for link in links]
        )
    )])
    
    fig.update_layout(title_text="Career Path Flow Diagram", font_size=10, height=500)
    
    return fig

def generate_career_explanation(career_path, education, skills, interests, probability):
    """Generate explanation text for why a career path is recommended"""
    base_text = CAREER_INFO.get(career_path, {}).get('why_text', 
        'Your profile shows strong alignment with this career path based on your education, skills, and interests.')
    
    skills_list = skills.split(',') if isinstance(skills, str) else skills
    interests_list = interests.split(',') if isinstance(interests, str) else interests
    
    explanation = f"""
    **{career_path}** ({CAREER_INFO.get(career_path, {}).get('icon', '💼')})
    
    **Why this career fits you:**
    {base_text}
    
    **Your Profile Match:**
    - Education: {education}
    - Key Skills: {', '.join(skills_list[:3])}
    - Primary Interests: {', '.join(interests_list[:2])}
    
    **Confidence Level:** {probability:.1%}
    
    **Next Steps:**
    1. Research specific requirements for {career_path}
    2. Identify skill gaps and create learning plan
    3. Connect with professionals in this field
    4. Gain relevant experience through projects or internships
    """
    
    return explanation

def create_pdf_report(user_inputs, predictions, career_explanations):
    """Create PDF report of career predictions"""
    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 15)
            self.cell(0, 10, 'Career Path Prediction Report', 0, 1, 'C')
            self.ln(10)
        
        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
    
    pdf = PDF()
    pdf.add_page()
    pdf.set_font('Arial', '', 12)
    
    # User inputs section
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Your Profile', 0, 1)
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 8, f'Education: {user_inputs["education"]}', 0, 1)
    pdf.cell(0, 8, f'Skills: {", ".join(user_inputs["skills"])}', 0, 1)
    pdf.cell(0, 8, f'Interests: {", ".join(user_inputs["interests"])}', 0, 1)
    pdf.ln(10)
    
    # Top predictions
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Top Career Predictions', 0, 1)
    pdf.set_font('Arial', '', 12)
    
    sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:5]
    for i, (career, prob) in enumerate(sorted_predictions, 1):
        pdf.cell(0, 8, f'{i}. {career} - {prob:.1%}', 0, 1)
    
    pdf.ln(10)
    
    # Detailed explanations
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Career Recommendations', 0, 1)
    pdf.set_font('Arial', '', 12)
    
    for career, explanation in career_explanations.items():
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 8, career, 0, 1)
        pdf.set_font('Arial', '', 10)
        # Split explanation into lines
        lines = explanation.split('\n')
        for line in lines:
            if line.strip():
                pdf.cell(0, 6, line.strip(), 0, 1)
        pdf.ln(5)
    
    # Save to bytes
    pdf_bytes = pdf.output(dest='S').encode('latin-1')
    return pdf_bytes

def get_career_icon(career_path):
    """Get icon for a career path"""
    return CAREER_INFO.get(career_path, {}).get('icon', '💼')

def get_career_description(career_path):
    """Get description for a career path"""
    return CAREER_INFO.get(career_path, {}).get('description', 'Career path description not available.')

def prepare_features_for_prediction(education, skills, interests, model):
    """Prepare user inputs for model prediction"""
    # Encode education
    try:
        education_encoded = model.education_encoder.transform([education])[0]
    except ValueError:
        education_encoded = 0  # Default to first category
    
    # Process skills
    skills_text = ', '.join(skills) if isinstance(skills, list) else skills
    skills_features = model.skills_vectorizer.transform([skills_text]).toarray()[0]
    
    # Process interests
    interests_text = ', '.join(interests) if isinstance(interests, list) else interests
    interests_features = model.interests_vectorizer.transform([interests_text]).toarray()[0]
    
    # Combine all features
    features = np.concatenate([[education_encoded], skills_features, interests_features])
    
    return features

# Task status tracking removed for clean deployment