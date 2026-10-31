# Groundwater Quality Analysis and Classification Using Machine Learning

## Project Overview

This project focuses on analysing groundwater quality data using Python and machine learning techniques. The main objective is to explore the relationship between different chemical parameters and pH levels, and to develop models that can predict pH values and classify groundwater into Acidic, Neutral, and Alkaline categories.

The project covers the complete machine learning workflow, starting from data cleaning and exploratory data analysis to model training and evaluation.

## Dataset Information

- Original dataset size: 446,489 rows
- Original number of columns: 29
- Initial sample selected for analysis: 100,000 rows
- Final cleaned dataset: 17,608 rows
- Final columns: 21

The dataset contains groundwater chemical measurements, including parameters such as Calcium, Chloride, Conductivity, Fluorine, Sodium, Magnesium, Nitrate, Sulfate, and Hardness.

**Note:** Add the verified original data.gov.in dataset URL here once its exact source page is confirmed.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Spyder IDE

## Project Workflow

### 1. Data Cleaning and Preprocessing

The original dataset required several preprocessing steps before model development.

- Selected a sample of 100,000 records.
- Removed rows with missing District values.
- Removed columns containing more than 80% missing values.
- Removed records with missing pH values.
- Removed invalid pH readings outside the 0–14 range.
- Removed records with missing Carbonate or Sodium Absorption Ratio values.
- Filled remaining missing values in selected chemical features using median imputation.

After preprocessing, the final dataset contained 17,608 rows and 21 columns.

### 2. Exploratory Data Analysis (EDA)

EDA was performed to understand the distribution of groundwater chemical parameters and their relationships with pH.

The analysis included:

- Histograms for understanding feature distributions.
- Boxplots for examining outliers.
- Correlation heatmaps for identifying relationships.
- Scatter plots for studying relationships between chemical parameters and pH.
- Statistical analysis of selected features.

Some notable correlations with pH were:

| Chemical Parameter | Correlation with pH |
|---|---:|
| Fluorine | +0.1713 |
| Sodium | +0.1071 |
| Potassium | +0.0819 |
| Hydrogencarbonate | +0.0714 |
| Calcium | -0.1534 |
| Hardness | -0.0678 |

### 3. Machine Learning Models

Five machine learning models were implemented.

**Regression Models**

1. Simple Linear Regression
   - Used Fluorine as the input feature.
   - Predicted continuous pH values.

2. Polynomial Regression
   - Used Fluorine, Calcium, Sodium, and Hydrogencarbonate.
   - Applied a polynomial transformation of degree 2.

**Classification Models**

3. Logistic Regression

4. Decision Tree Classifier

5. Support Vector Machine (SVM)

For classification, groundwater samples were grouped using the following pH categories:

- Acidic: pH below 6.5
- Neutral: pH from 6.5 to 8.5
- Alkaline: pH above 8.5

The classification models used six chemical features:
Fluorine, Calcium, Sodium, Hydrogencarbonate, Potassium, and Sulfate.

An 80:20 train-test split was used with random_state=42.

Class weighting was applied to the classification models to address class imbalance.

## Model Performance

### Regression Results

| Model | MAE | MSE | RMSE | R² |
|---|---:|---:|---:|---:|
| Simple Linear Regression | 0.3444 | 0.1985 | 0.4455 | 0.0361 |
| Polynomial Regression | 0.3310 | 0.1871 | 0.4325 | 0.0914 |

Polynomial Regression performed better than Simple Linear Regression across the reported regression metrics.

### Classification Results

| Model | Accuracy |
|---|---:|
| Logistic Regression | 66.41% |
| Decision Tree | 90.46% |
| SVM | 70.22% |

The Decision Tree achieved the highest overall classification accuracy.

However, the dataset contained an imbalanced distribution of pH categories. The Decision Tree performed particularly well on the Neutral category but had lower recall for Acidic and Alkaline samples.

Logistic Regression and SVM achieved better recall for the minority categories, demonstrating why class-wise evaluation is important alongside overall accuracy.

## Key Learnings

- Gained practical experience in handling a large real-world dataset.
- Understood the importance of data cleaning before model training.
- Learned how to analyse relationships between chemical parameters using EDA.
- Implemented both regression and classification algorithms.
- Compared machine learning models using appropriate evaluation metrics.
- Understood the effect of class imbalance on classification performance.

## Limitations

- The regression models achieved relatively low R² scores, indicating limited explanatory power.
- The classification dataset had a significant class imbalance.
- Minority-category predictions were less reliable than Neutral-category predictions.
- The models use selected chemical parameters and do not represent a complete assessment of drinking-water safety.

## Conclusion

This project provided hands-on experience with the complete machine learning pipeline, from data preprocessing and exploratory analysis to model development and evaluation.

The results showed that different algorithms perform differently depending on the task and evaluation metric. It also highlighted the importance of understanding the dataset before interpreting model performance.

---

**Project Type:** Machine Learning / Data Analysis

**Tools:** Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

**IDE:** Spyder
