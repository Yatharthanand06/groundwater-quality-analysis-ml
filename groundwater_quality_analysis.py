
import pandas as pd
import numpy as np

# Step 1: Load the dataset
df = pd.read_csv(
    "C:/Users/Dell/Downloads/pythongroundwaterqty_dataset_final.csv"
)

print("First five rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns)

print("\nDataset information:")
print(df.info())



# Step 1: Select 100000 random rows
df = df.sample(n=100000, random_state=42)

# Step 2: Remove missing District rows
df = df.dropna(subset=['District'])

# Step 3: Calculate missing percentage
missing_percent = df.isnull().sum() / len(df) * 100

print("\nMissing percentages:")
print(missing_percent.sort_values(ascending=False))

# Step 4: Remove columns having more than 80% missing values
columns_to_drop = missing_percent[
    missing_percent > 80
].index

print("\nColumns removed:")
print(columns_to_drop)

df = df.drop(columns=columns_to_drop)

print("\nDataset shape after removing columns:")
print(df.shape)


# Step 5: Rename remaining columns

df.rename(columns={
    'Amount Of Carbonate (UOM:), Scaling Factor:1': 'Carbonate',
    'Amount Of Calcium (UOM:), Scaling Factor:1': 'Calcium',
    'Amount Of Chloride (UOM:), Scaling Factor:1': 'Chloride',
    'Amount Of Electrical Conductivity (UOM:), Scaling Factor:1': 'Conductivity',
    'Amount Of Fluorine (UOM:), Scaling Factor:1': 'Fluorine',
    'Amount Of Hydrogencarbonate (UOM:), Scaling Factor:1': 'Hydrogencarbonate',
    'Amount Of Potassium (UOM:), Scaling Factor:1': 'Potassium',
    'Amount Of Magnesium (UOM:), Scaling Factor:1': 'Magnesium',
    'Amount Of Nitrate (UOM:), Scaling Factor:1': 'Nitrate',
    'Amount Of Sodium (UOM:), Scaling Factor:1': 'Sodium',
    'Amount Of Sodium Absorption Ratio (UOM:), Scaling Factor:1': 'Sodium_Absorption_Ratio',
    'Amount Of Sulfate (UOM:), Scaling Factor:1': 'Sulfate',
    'Amount Of Hardness Total (UOM:MgCaCO3/L(calciumhardness)), Scaling Factor:1': 'Hardness',
    'Amount Of Potential Of Hydrogen (UOM:pH), Scaling Factor:1': 'pH'
}, inplace=True)

# Display renamed columns
print("\nRenamed column names:")
print(df.columns.tolist())




# Step 6: Check negative values in chemical columns

chemical_cols = [
    'Carbonate',
    'Calcium',
    'Chloride',
    'Conductivity',
    'Fluorine',
    'Hydrogencarbonate',
    'Potassium',
    'Magnesium',
    'Nitrate',
    'Sodium',
    'Sodium_Absorption_Ratio',
    'Sulfate',
    'Hardness',
    'pH'
]

# Step 7: Check pH values

print("\nMinimum pH:", df['pH'].min())
print("Maximum pH:", df['pH'].max())

print("Missing pH values:", df['pH'].isnull().sum())

# Count invalid pH values
invalid_ph = df[
    (df['pH'] < 0) | (df['pH'] > 14)
]

print("Invalid pH readings:", len(invalid_ph))
# Step 8: Remove rows with missing pH values

print("Rows before removing missing pH:", len(df))

df = df.dropna(subset=['pH'])

print("Rows after removing missing pH:", len(df))

print("Remaining missing pH values:", df['pH'].isnull().sum())

# Step 9: Remove invalid pH readings

print("Rows before removing invalid pH:", len(df))

df = df[
    (df['pH'] >= 0) &
    (df['pH'] <= 14)
]

print("Rows after removing invalid pH:", len(df))

# Verify pH range
print("Minimum pH:", df['pH'].min())
print("Maximum pH:", df['pH'].max())

# Step 10: Check remaining missing values

print("\nMissing values in each column:")
print(df.isnull().sum())

# Calculate missing percentage
missing_percent = (df.isnull().sum() / len(df)) * 100

print("\nMissing percentage after pH filtering:")
print(missing_percent.sort_values(ascending=False))


# Create temporary dataset
df_test = df.dropna(
    subset=['Carbonate', 'Sodium_Absorption_Ratio']
)

# Check missing values in remaining columns
print("\nMissing values after removing rows:")

print(df_test.isnull().sum())

# Calculate missing percentages
missing_after = (
    df_test.isnull().sum() / len(df_test)
) * 100

print("\nMissing percentages after removing rows:")
print(missing_after.sort_values(ascending=False))

# Check new dataset size
print("\nNew dataset shape:", df_test.shape)



# Step 11: Apply the selected dataset
df = df_test.copy()

# Chemical columns used as input features
chemical_cols = [
    'Calcium',
    'Chloride',
    'Conductivity',
    'Fluorine',
    'Hydrogencarbonate',
    'Potassium',
    'Magnesium',
    'Nitrate',
    'Sodium',
    'Sulfate',
    'Hardness'
]

# Fill missing values using median
for col in chemical_cols:
    df[col] = df[col].fillna(df[col].median())

# Check missing values after filling
print("\nMissing values after median imputation:")
print(df[chemical_cols].isnull().sum())

# Check final dataset shape
print("\nFinal dataset shape:", df.shape)



# Step 12: Correlation analysis

correlation = df[chemical_cols + ['pH']].corr()

print("\nCorrelation matrix:")
print(correlation)

# Check correlation of each feature with pH
print("\nCorrelation with pH:")
print(correlation['pH'].sort_values(ascending=False))


# Step 13: Correlation heatmap

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 8))

sns.heatmap(
    correlation,
    annot=True,
    cmap='coolwarm',
    fmt='.2f',
    linewidths=0.5
)

plt.title("Correlation Heatmap of Groundwater Parameters")
plt.tight_layout()
plt.show()


# Step 14: Detect outliers using boxplots

plt.figure(figsize=(14, 8))

df[chemical_cols].boxplot()

plt.title("Boxplot of Groundwater Chemical Parameters")
plt.ylabel("Measurement Values")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Step 15: Detect outliers using IQR method

print("\nOutlier count in each column:")

for col in chemical_cols:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    outliers = df[
        (df[col] < lower_limit) |
        (df[col] > upper_limit)
    ]

    print(col, ":", len(outliers))


# Step 16: Check distribution of chemical parameters

plt.figure(figsize=(15, 10))

df[chemical_cols].hist(
    bins=30,
    figsize=(15, 10)
)

plt.suptitle("Distribution of Groundwater Chemical Parameters")

plt.tight_layout()
plt.show()


# Step 17: Calculate skewness

print("\nSkewness of chemical parameters:")

skew_values = df[chemical_cols].skew()

print(skew_values.sort_values(ascending=False))


# Step 18: Check pH distribution

plt.figure(figsize=(8, 5))


plt.hist(df['pH'], bins=30)

plt.title("Distribution of Groundwater pH")
plt.xlabel("pH Value")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

print("pH skewness:", df['pH'].skew())
print("pH mean:", df['pH'].mean())
print("pH median:", df['pH'].median())


from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


#--------------------Simple linear regression-----------------------
# Select independent and dependent variables

X = df[['Fluorine']]
y = df['pH']

# Split dataset

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# Create model

simple_model = LinearRegression()

# Train model

simple_model.fit(X_train, y_train)

# Prediction

y_pred = simple_model.predict(X_test)

# Evaluation

print("Simple Linear Regression")

print("MAE:", mean_absolute_error(y_test, y_pred))

print("MSE:", mean_squared_error(y_test, y_pred))

print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

print("R-squared:", r2_score(y_test, y_pred))


#---------------------------Polynomial regression----------------------
# Select features

selected_cols = [
    'Fluorine',
    'Calcium',
    'Sodium',
    'Hydrogencarbonate'
]

X = df[selected_cols]
y = df['pH']

# Split dataset

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# Convert features into polynomial features

poly = PolynomialFeatures(degree=2)

X_train_poly = poly.fit_transform(X_train)

X_test_poly = poly.transform(X_test)

# Create model

poly_model = LinearRegression()

# Train model

poly_model.fit(X_train_poly, y_train)

# Prediction

y_pred = poly_model.predict(X_test_poly)

# Evaluation

print("Polynomial Regression")

print("MAE:", mean_absolute_error(y_test, y_pred))

print("MSE:", mean_squared_error(y_test, y_pred))

print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

print("R-squared:", r2_score(y_test, y_pred))


#---------------------Logistics regression---------------------------
# Create pH categories

def classify_ph(ph):
    if ph < 6.5:
        return 'Acidic'
    elif ph <= 8.5:
        return 'Neutral'
    else:
        return 'Alkaline'

df['pH_Category'] = df['pH'].apply(classify_ph)

# Select independent variables

X = df[['Fluorine', 'Calcium', 'Sodium',
        'Hydrogencarbonate', 'Potassium', 'Sulfate']]

# Select dependent variable

y = df['pH_Category']

# Split dataset

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# Create model

log_model = LogisticRegression(
    max_iter=5000,
    class_weight='balanced'
)

# Train model

log_model.fit(X_train, y_train)

# Prediction

y_pred = log_model.predict(X_test)

# Evaluation

print("Logistic Regression")

print("Accuracy:", accuracy_score(y_test, y_pred))

print("Confusion Matrix:")

print(confusion_matrix(y_test, y_pred))

print("Classification Report:")

print(classification_report(y_test, y_pred, zero_division=0))



#----------------------------Decision tree classifier---------------------------
# Select independent variables

X = df[['Fluorine', 'Calcium', 'Sodium',
        'Hydrogencarbonate', 'Potassium', 'Sulfate']]

# Select dependent variable

y = df['pH_Category']

# Split dataset

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# Create model
dt_model = DecisionTreeClassifier(
    random_state=42,
    class_weight='balanced'
)

# Train model

dt_model.fit(X_train, y_train)

# Prediction

y_pred = dt_model.predict(X_test)

# Evaluation

print("Decision Tree Classification")

print("Accuracy:", accuracy_score(y_test, y_pred))

print("Confusion Matrix:")

print(confusion_matrix(y_test, y_pred))

print("Classification Report:")

print(classification_report(y_test, y_pred))


#-----------------------------------SVM---------------------------------
# Select independent variables

X = df[['Fluorine', 'Calcium', 'Sodium',
        'Hydrogencarbonate', 'Potassium', 'Sulfate']]

# Select dependent variable

y = df['pH_Category']

# Split dataset

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# Create model

svm_model = SVC(class_weight='balanced')

# Train model

svm_model.fit(X_train, y_train)

# Prediction

y_pred = svm_model.predict(X_test)

# Evaluation

print("Support Vector Machine")

print("Accuracy:", accuracy_score(y_test, y_pred))

print("Confusion Matrix:")

print(confusion_matrix(y_test, y_pred))

print("Classification Report:")

print(classification_report(y_test, y_pred,zero_division=0))








