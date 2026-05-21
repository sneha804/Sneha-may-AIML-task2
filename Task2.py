import numpy as np
import pandas as pd

# Load dataset
df = pd.read_csv("student_scores.csv")

# Display first 5 rows
print(df.head())

# NumPy operations
math_scores = np.array(df['math_score'])

print("Mean:", np.mean(df['math_score']))
print("Median:", np.median(df['math_score']))
print("Maximum:", np.max(df['math_score']))
print("Minimum:", np.min(df['math_score']))

# Normalize scores
normalized_scores = (
    (df['math_score'] - df['math_score'].min()) /
    (df['math_score'].max() - df['math_score'].min())
)

# Data types
print(df.dtypes)

# Missing values
print(df.isnull().sum())

# Attendance below 70
print(df[df['attendance'] < 70])

# Convert columns
numeric_cols = ['age', 'math_score', 'science_score', 'attendance']

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col].fillna(df[col].mean(), inplace=True)

# Handle categorical values
df['gender'].fillna(df['gender'].mode()[0], inplace=True)
df['name'].fillna("Unknown", inplace=True)

# Correct gender spelling
df['gender'] = df['gender'].replace({
    'Malee': 'Male',
    'Femal e': 'Female'
})

# Convert date
df['exam_date'] = pd.to_datetime(df['exam_date'], errors='coerce')

# Outlier handling
Q1 = df['math_score'].quantile(0.25)
Q3 = df['math_score'].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df['math_score'] = np.where(
    df['math_score'] > upper,
    upper,
    np.where(df['math_score'] < lower, lower, df['math_score'])
)

# Remove duplicates
df = df.drop_duplicates()

# Average score
df['average_score'] = (
    df['math_score'] + df['science_score']
) / 2

# Top 5 students
top_students = df.sort_values(
    by='average_score',
    ascending=False
).head(5)

print(top_students[['name', 'average_score']])

# Correlation
correlation = df['attendance'].corr(df['average_score'])
print("Correlation:", correlation)

# Group by gender
grouped = df.groupby('gender')['average_score'].mean()

print(grouped)