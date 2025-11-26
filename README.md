# Problem Statement: 

Social media has become a primary platform where individuals openly express their emotions, thoughts, and mental struggles. Among these expressions, some posts may indicate suicidal ideation or an immediate risk of self-harm. Manual monitoring at scale is nearly impossible due to the enormous volume of daily posts, language variability, and emotional complexity. As a result, there is a critical need for an automated system capable of identifying suicidal intent from text-based content in real time.

This project aims to build an AI-driven text classification system to automatically detect suicide-related posts using Natural Language Processing (NLP) and Deep Learning (LSTM/Bidirectional LSTM). The model analyzes social media text, learns emotional patterns, and predicts whether a post expresses suicidal or non-suicidal intent. This solution can potentially support mental health organizations, helplines, and online safety systems to intervene early and prevent tragic outcomes.

# Objectives: 

--> To preprocess and analyze text data collected from social media platforms.
--> To develop a deep learning-based model using LSTM/Bidirectional LSTM for suicide ideation detection.
--> To deploy the model as a web application using Streamlit.
--> To evaluate the model's performance and provide insights for potential improvements.

# Scope:
--> Detect only binary classification: Suicide vs Non-suicide text.
--> Focus on English-language posts.
--> Model trained on manually labeled dataset.

# Expected Impact: 
--> Early detection of suicidal ideation.
--> Support for mental health organizations and helplines.
--> Potential prevention of tragic outcomes.

# TechStack: 
--> Python
--> Streamlit
--> Keras
--> Numpy
--> Pandas
--> Matplotlib
--> Seaborn
--> Scikit-learn
--> NLTK
--> Tensorflow
