# Sentiment-Analysis
**COMPANY**: CODTECH IT SOLUTIONS 
**NAME**: TANYA DEEP 
**INTERN ID**: CT08WBB 
**DOMAIN**: DATA ANALYSIS 
**DURATION**: 4 WEEKS 
**MENTOR**: NEELA SANTOSH

**DESCRIPTION**
This Python script performs sentiment analysis on Amazon review data. It loads the data, converts numerical ratings into "positive," "negative," or "neutral" sentiment labels, and cleans the review text by removing special characters and stop words. The cleaned text is then transformed into numerical features using TF-IDF vectorization. A Multinomial Naive Bayes classifier is trained on this data to predict sentiment. The model's accuracy and classification report are printed, and a bar plot visualizes the sentiment distribution. Essentially, the script prepares text data, trains a sentiment prediction model, and evaluates its performance.

Data Preparation: Loads reviews and converts ratings to sentiment labels.
Text Cleaning: Removes noise from review text.
Feature Extraction: Converts text to numerical data.
Model Training: Trains a Naive Bayes classifier.
Evaluation & Visualization: Assesses model accuracy and displays sentiment distribution.

**OUTPUT**
Accuracy: 0.75
              precision    recall  f1-score   support

    negative       1.00      0.01      0.03        74
    positive       0.75      1.00      0.86       219
    accuracy                           0.75       293
    macro avg       0.88      0.51      0.44       293
    weighted avg       0.81      0.75      0.65       293
