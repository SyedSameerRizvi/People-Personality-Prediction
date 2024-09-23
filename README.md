# Predicting MBTI Personality Types from Demographics and Interests

## Project Overview
This project aims to predict an individual's MBTI (Myers-Briggs Type Indicator) personality type based on a set of demographic and interest-related features. The data is synthetic and designed for personality research purposes, with over 100,000 samples and various input features, including age, gender, education level, and personality-related scores.

## Dataset
The dataset used in this project contains the following features:

- **Age**: A continuous variable representing the age of the individual.
- **Gender**: A categorical variable indicating the gender of the individual. Possible values are 'Male' and 'Female'.
- **Education**: A binary variable, A value of 1 indicates the individual has at least a graduate-level education (or higher), and 0 indicates an undergraduate, high school level or Uneducated.
- **Interest**: A categorical variable representing the individual's primary area of interest.
- **Introversion Score**: A continuous variable ranging from 0 to 10, representing the individual's tendency toward introversion versus extraversion. Higher scores indicate a greater tendency toward extraversion.
- **Sensing Score**: A continuous variable ranging from 0 to 10, representing the individual's preference for sensing versus intuition. Higher scores indicate a preference for sensing.
- **Thinking Score**: A continuous variable ranging from 0 to 10, indicating the individual's preference for thinking versus feeling. Higher scores indicate a preference for thinking.
- **Judging Score**: A continuous variable ranging from 0 to 10, representing the individual's preference for judging versus perceiving. Higher scores indicate a preference for judging.
- **Personality**: Target that contains People Personality Type

The target variable is the individual's MBTI type, which can be one of the 16 personality types, e.g., ISTJ, ENFP, etc

The dataset can found here: [People Personality type](https://www.kaggle.com/datasets/stealthtechnologies/predict-people-personality-types/data)

## Objective
The objective of this project is to train machine learning models to predict MBTI personality types using demographic and personality data. The target accuracy is 90% or higher.

## Approach
The project involves several steps:
1. **Exploratory Data Analysis (EDA)**: Visualization and initial analysis of the dataset to identify patterns and correlations.
2. **Preprocessing**: Handling missing values, feature encoding, and data scaling as required.
3. **Modeling**: Using various machine learning algorithms to predict MBTI types, including:
   - K-Nearest Neighbors (KNN)
   - Random Forest Classifier
   - XGBoost Classifier
4. **Evaluation**: Measuring model performance using accuracy, precision, recall, F1 score, and ROC-AUC.

## Tools & Libraries
The following tools and libraries are used in this project:
- Python (NumPy, Pandas, Matplotlib, Seaborn)
- Scikit-learn for machine learning models and evaluation
- XGBoost for gradient boosting classification
- GridSearchCV and RandomizedSearchCV for hyperparameter tuning

## Results
The goal is to achieve a predictive accuracy of 90% or higher. The project uses a combination of model performance metrics to evaluate the success of the models, including confusion matrices, classification reports, and ROC curves.

## Future Work
Future steps include:
- Tuning the models further using advanced hyperparameter optimization techniques.
- Applying deep learning approaches such as neural networks for improved accuracy.
- Extending the dataset to include more features, such as additional personality dimensions.
