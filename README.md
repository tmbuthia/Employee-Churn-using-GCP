# Employee-Churn-using-GCP
This project aims to predict employee churn using machine learning. It utilizes data from BigQuery, trains a Random Forest model using PyCaret, and stores predictions back in BigQuery.
## Overview

Employee churn, or turnover, is a critical metric for organizations aiming to retain talent. This project provides an end-to-end pipeline to analyze a dataset related to employee churn and build predictive models to classify or estimate churn likelihood.
## Key Features
1. Data Acquisition and Preparation:
  Data Source: Historical employee data from Excel spreadsheets.
  Data Ingestion: Import data into a BigQuery database. The data consist of two csvs that are joined using a Union to form on table for querying
  ```sql
  SELECT *,"Original" as Type FROM `hale-life-398519.EmployeeChurn.tbl_hr_data` 
  UNION ALL
  SELECT *,"Pilot" as Type FROM `hale-life-398519.EmployeeChurn.new_employee`
  ```
  Data Cleaning: Handle missing values, outliers, and inconsistencies.
  Exploratory Data Analysis (EDA) for uncovering patterns and trends.
  Feature Engineering: Create relevant features for model training.Feature importance analysis is performed to identify the most influential features in predicting churn.
  A feature importance table is stored in a BigQuery table 
![image](https://github.com/user-attachments/assets/da481679-5817-4da3-b54f-e3901174c2ed)

3. Model Development:
  Data Splitting: Divide the data into training and testing sets. Employee churn data is sourced from a BigQuery table
  Model Training: Train the model on the training data. PyCaret is used for model training and evaluation.
  Model Evaluation: Evaluate the model's performance using metrics like accuracy, precision, recall, and F1-score.A Random Forest model is selected as the best performing model based on comparison with other models.
  Model Tuning: Fine-tune hyperparameters to improve model performance.

5. Model Deployment and Prediction:
  Deploy Model: Deploy the trained model to a cloud-based environment.
  Prediction: Use the deployed model to predict the likelihood of attrition for new employees.
  Result Analysis: Analyze the predicted probabilities and identify employees at risk of attrition.

6. Dashboard Creation:
  Data Visualization: Create a dashboard in Looker Studio to visualize key insights.
  Dashboard Components:
  KPI: Overall attrition rate
  Bar Chart: Attrition rate by department
  Pie Chart: Distribution of predicted attrition probabilities
  Table: List of employees at high risk of attrition
  Filters: Allow users to filter data by department, tenure, or other relevant factors.


7. Insights and Recommendations:
  Identify Risk Factors: Determine the key factors contributing to employee attrition.
  Propose Interventions: Recommend strategies to mitigate attrition, such as targeted training, improved compensation, or enhanced work-life balance.
  Monitor Performance: Continuously monitor the model's performance and retrain as needed.

Tools and Technologies:
Google Cloud Platform (GCP)|BigQuery|Google Colab|Python|Pandas|NumPy|Scikit-learn|Looker Studio

Dependencies
To run the notebook, ensure you have the following Python libraries installed:
pandas|numpy|matplotlib|seaborn|scikit-learn

The notebook evaluates multiple machine learning models and provides insights into factors influencing employee churn. It also includes performance metrics and visualizations to assist in model interpretation.

Future Work
Incorporate advanced feature selection techniques.
Test additional machine learning models, such as ensemble methods.
Evaluate the impact of imbalanced classes using techniques like SMOTE.
Deploy the model using a web application framework like Flask or FastAPI.
License
This project is licensed under the MIT License. See the LICENSE file for details.
Acknowledgments
Open-source contributors for Python libraries.
Data providers for the dataset used in this analysis.


