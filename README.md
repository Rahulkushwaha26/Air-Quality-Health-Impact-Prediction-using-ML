# AI-Powered Air Quality & Health Impact Prediction

## About the Project
Air pollution is one of the most serious public health challenges in India today. This project builds a complete Big Data analytics pipeline to predict the Air Quality Index category from real-world hourly pollutant sensor readings collected across nine major Indian cities. Instead of relying on manual monitoring, the system uses supervised machine learning classification algorithms to automatically classify air quality into six categories — Good, Satisfactory, Moderate, Poor, Very Poor, and Severe.
The dataset is sourced from the Central Pollution Control Board of India and contains 135,513 hourly records spanning 2017 to 2020, covering 14 pollutant features across cities including Delhi, Mumbai, Kolkata, Chennai, Patna, Hyderabad, Jaipur, Amaravati, and Visakhapatnam.

## Problem Statement
Given hourly pollutant concentration readings from automated monitoring stations, the goal is to classify each observation into one of six Air Quality Index categories. The dataset presents inherent class imbalance — Moderate and Satisfactory categories dominate while Severe has only 3,171 records — making accurate classification of minority classes especially important since they pose the greatest health risk.


## Pipeline Overview
Raw dataset input of 135,513 records is passed through the following steps:

1. Exploratory Data Analysis(EDA) — distribution, correlation, and city-wise AQI analysis
2. Quality checks — null values, duplicate rows, and infinite value detection
3. Outlier removal using the Interquartile Range method — 53,517 rows removed
4. Feature engineering — Datetime column dropped as non-predictive
5. Label encoding — City and AQI Bucket columns encoded to integers
6. Feature scaling — Standard Scaler applied to all 14 input features
7. Train-test split — 80 percent training (65,596 records) and 20 percent testing (16,400 records)
8. Model training — 7 ML classifiers and 1 Deep Neural Network
9. Evaluation — Accuracy, Precision, Recall, and F1-Score per class

## Models Implemented
Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, K-Nearest Neighbours, Support Vector Machine, Naive Bayes, and a Deep Neural Network built using TensorFlow and Keras.

## Results
1. Gradient Boosting - 100.00%
2. Decision Tree (depth 5) - 100.00%
3. Random Forest (100 trees) - 99.99%
4. Logistic Regression- 99.46%
5. Support Vector Machine - 92.18%
6. K-Nearest Neighbours - 88.43%
7. Naive Bayes - 83.74%
8. Deep Neural Network - 99%+
- Random Forest and Gradient Boosting significantly outperform all prior works reviewed in the literature, which reported a best accuracy of 96.1 percent on similar datasets.


## Key Findings
 - PM2.5 and PM10 are the two most dominant predictors of AQI category
 - Outlier removal using IQR reduced the dataset from 135,513 to 81,996 clean records
 - Naive Bayes performed worst due to the feature independence assumption being violated by correlated pollutant data
 - Gradient Boosting and Random Forest achieved near-perfect classification even on the minority Severe category

## Dataset
Source: Central Pollution Control Board of India via Kaggle
Cities covered: Delhi, Mumbai, Kolkata, Chennai, Patna, Hyderabad, Jaipur, Amaravati, Visakhapatnam
Records: 135,513 hourly observations (2017 to 2020)
Features: 14 pollutants including PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene
Target: AQI Bucket — Good, Satisfactory, Moderate, Poor, Very Poor, Severe
Tech Stack
Python 3, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, TensorFlow, Keras, Jupyter Notebook

## Project Structure
- data/
    cleaned_air_quality_data.csv
 - src/
   AQI_Project_Code
 - output/
    confusion_matrices/
    model_accuracy_comparison
README.md

Jaypee Institute of Information Technology, Noida
Department of Computer Science and Engineering and Information Technology
M.Tech in Artificial Intelligence and Data Science, 2026
