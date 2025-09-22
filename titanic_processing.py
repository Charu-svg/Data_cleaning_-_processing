# 1) Import libraries
import pandas as pd            # handles tables (dataframes)
import numpy as np             # numbers and math helpers
import matplotlib.pyplot as plt# plotting library
import seaborn as sns          # nicer plots on top of matplotlib
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# 2) Load example Titanic dataset (built into seaborn)
df = sns.load_dataset('titanic')

# 3) Quick look
df.head()
# show structure and missing values
print("Shape (rows, columns):", df.shape)
print("\nColumns and types:")
print(df.dtypes)
print("\nMissing values per column:")
print(df.isnull().sum())

# statistical summary for numeric columns
df.describe()
# 1) Distribution of ages
plt.figure(figsize=(6,3))
sns.histplot(df['age'], kde=False)
plt.title('Age distribution')
plt.show()

# 2) Boxplot for fare to see outliers
plt.figure(figsize=(6,3))
sns.boxplot(x=df['fare'])
plt.title('Fare boxplot')
plt.show()

# 3) Missing values heatmap (quick visual)
plt.figure(figsize=(6,4))
sns.heatmap(df.isnull(), cbar=False)
plt.title('Missing values map (yellow = missing)')
plt.show()
# 1) Copy data so original remains untouched
df_clean = df.copy()

# 2) Fill missing 'age' with median (typical choice)
df_clean['age'] = df_clean['age'].fillna(df_clean['age'].median())

# 3) Fill missing 'embarked' with the most common value (mode)
df_clean['embarked'] = df_clean['embarked'].fillna(df_clean['embarked'].mode()[0])

# 4) Drop columns with many missing values or not useful for modeling
# 'deck' has many missing values in this dataset
df_clean = df_clean.drop(columns=['deck'])

# 5) Confirm missing values reduced
print(df_clean.isnull().sum())
# 1) Convert 'sex' to numeric: male=0, female=1
df_clean['sex'] = df_clean['sex'].map({'male':0, 'female':1})

# 2) Convert 'embarked' to one-hot columns (drop_first avoids redundant column)
df_clean = pd.get_dummies(df_clean, columns=['embarked'], drop_first=True)

# 3) Convert 'class' (which is ordered) using label mapping (1st, 2nd, 3rd)
df_clean['class'] = df_clean['class'].map({'First':3, 'Second':2, 'Third':1})

# 4) Check the new top rows
df_clean.head()
# 1) Pick numeric columns to scale
num_cols = ['age', 'fare']

# 2) Create scaler and fit_transform
scaler = StandardScaler()
df_clean[num_cols] = scaler.fit_transform(df_clean[num_cols])

# 3) Check scaled values
df_clean[num_cols].head()
# 1) Compute Q1, Q3 and IQR for 'fare'
Q1 = df_clean['fare'].quantile(0.25)
Q3 = df_clean['fare'].quantile(0.75)
IQR = Q3 - Q1

# 2) Define filter: keep rows within [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
df_no_outliers = df_clean[(df_clean['fare'] >= lower) & (df_clean['fare'] <= upper)]

# 3) Print sizes to see how many rows were removed
print("Original rows:", df_clean.shape[0])
print("After outlier removal:", df_no_outliers.shape[0])
# 1) Quick info
print(df_no_outliers.info())

# 2) Save cleaned dataset so you can reuse it later
df_no_outliers.to_csv('titanic_clean.csv', index=False)
print("Saved cleaned dataset to titanic_clean.csv")
# 1) Choose target and features
data = df_no_outliers.copy()
data = data.dropna(subset=['survived'])  # ensure no missing target
X = data[['sex','age','fare','class']]   # features (simple set)
y = data['survived']                     # target

# 2) Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3) Train a logistic regression
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# 4) Predict and evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
