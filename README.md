# Predicting Purchase Intent of Customers for Targeted Marketing Using Clickstream Data 

## Project Objectives 

This project focuses on analyzing customer journey sequences from e-commerce clickstream data of customers in Indonesia to improve targeted advertising and enhance user experience. The goal is to determine whether a browsing session leads to a purchase, framing the problem as a binary classification task. 

To evaluate which feature types best predict conversion, we compared models trained on static, temporal, and sequential features individually, and then all combined. Our objectives include: 

- Identify which clickstream features (temporal, static, or sequential) best predict purchase conversions. 
- Evaluate the impact of combining these feature types on model performance using AutoML (DataRobot). 
- Generate actionable insights into user behavior patterns (e.g., session length, funnel depth) that correlate with conversion or drop-off.

## Dataset Used
Source:  https://www.kaggle.com/datasets/bytadit/transactional-ecommerce

## Feature Engineering Techniques 

The first step we took for the clickstream, transaction and products data was to merge it and create new metrics for feature engineering to perform predictive modeling for finding purchase intent or user conversion in Data Robot/AutoML. Hence, our decision to have our target variable be “converted”. 

1. Temporal Features:  For temporal based feature modeling, we wanted to see the impact of user session-based data on the user conversion or purchase intent.  Each action is encoded with a set of features that describe user interaction, such as event type (view, cart, etc.), product details before user makes by finding the total number of events, session duration, average session time to provide insights on overall engagement.
2. Static Features:  Static features are derived from stable, non-time-dependent attributes that describe either the user, the session, or the product in a summarized or categorical manner.  They capture foundational context that can help characterize the overall profile of a user or interaction, such as customer demographic information.
3.  Sequential Features Since the heart of clickstream data is, following a user’s journey, we created a new feature ‘funnel sequence’, based on the event type journey in a session per customer, where we create a string capturing the order of event types for sequential modeling.  We believed this feature would be useful for determining user drop off, and for applying deep learning which we hope to look at in future modeling. 

## Leveraging DataRobot for Determing Customer Purchase Conversion 
The final dataset is a cleaned, merged, and enriched representation of user sessions, ready for supervised classification modeling in Data Robot. It combines time-aware behavioral metrics, sequential journey indicators, and all features were scaled, binned, or encoded appropriately for ingestion into AutoML/DataRobot . 

### Model Selection Process: 
- Light Gradient Boosting on ElasticNet Predictions (Best) 
- eXtreme Gradient Boosted Trees Classifier with Early Stopping (Fast Feature Binning)  
- RandomForest Classifier (Gini) 
- Keras Slim Residual Neural Network Classifier using Training Schedule (1 Layer: 64 Units)

