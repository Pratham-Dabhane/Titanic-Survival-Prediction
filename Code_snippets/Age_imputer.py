import pandas as pd
from sklearn.impute import SimpleImputer

# Sample dataset with missing values in the 'Age' column
data = {
    'Name': ['John', 'Anna', 'Mike', 'Sara', 'Tom'],
    'Age': [28, 22, None, 35, None]
}

df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)

# Initialize the SimpleImputer to fill missing 'Age' values with the mean
imputer = SimpleImputer(strategy="mean")

# Reshape Age column to a 2D array for SimpleImputer
# Since we only want to fill missing values in the 'Age' column
df['Age'] = imputer.fit_transform(df[['Age']])

print("\nDataFrame after imputing missing 'Age' values with mean:")
print(df)
