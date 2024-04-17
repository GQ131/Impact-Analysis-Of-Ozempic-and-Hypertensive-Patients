# Impact Analysis of Ozempic on Obese and Hypertensive Patients

This repository contains a comprehensive analysis aimed at understanding the effects of Ozempic on a specific patient group aged 40-75, diagnosed with obesity and hypertension. Due to confidentiality agreements, the actual datasets provided by Ozempic for this study are not included within this repository. However, the methodologies, code, visualizations, and narratives derived from the analysis vividly illustrate our approach and findings.

## Overview:

This project is structured around several key analytical phases, designed to explore the causal relationships between Ozempic treatment and health outcomes within the target population. Through rigorous data processing, exploratory data analysis (EDA), and the application of advanced statistical models, we aim to accurately estimate the treatment effect of Ozempic.
Project Components:

* Exploratory Data Analysis (EDA): Initiated with an in-depth examination of the dataset's structure, missing values, and key statistics. This phase included generating visualizations to understand variable distributions, inter-variable relationships, and identifying data anomalies.
* Data Preprocessing and Merging: Prepared the datasets for analysis by addressing missing data, encoding categorical variables, and merging the Medical and Prescription datasets while ensuring data integrity for subsequent analysis.
* Causal Analysis Setup: Discussed potential endogeneity issues impacting the estimation of Ozempic's treatment effect. Outlined strategies to tackle these challenges, emphasizing the importance of accurate treatment effect estimation.
* Model Development: Implemented a Double-Lasso/Treatment Effect Lasso approach to estimate Ozempic's treatment effect, taking into account the endogeneity of treatment assignment. The variable selection process was thoroughly justified to ensure the model's integrity.
* Model Evaluation and Interpretation: The model's performance was critically evaluated, with a detailed interpretation of the estimated treatment effects. Insights into Ozempic's impact on the target patient population were derived from this evaluation.
* Additional Insights: Enhanced the analysis with demographic factors from census data and additional Ozempic information, providing a richer context for understanding the treatment effects.

## Key Insights:

* Identified causal relationships between Ozempic treatment and health outcomes in obese and hypertensive patients
* Addressed and adjusted for endogeneity to accurately estimate treatment effects
* Provided comprehensive insights into how demographic factors might influence these effects.

## Technologies and Techniques Used:

* Python for data analysis and modeling
* Pandas and NumPy for data manipulation
* Matplotlib and Seaborn for data visualization
* Scikit-learn for implementing advanced statistical models
