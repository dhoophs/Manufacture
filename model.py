import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV


# Load the dataset
df = pd.read_csv('synthetic_manufacturing_data.csv')

# Features and target
X = df[['Temperature', 'Run_Time']]  # Features
y = df['Downtime_Flag']  # Target (Downtime Flag)

# Split the data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train['Temp_Run_Interaction'] = X_train['Temperature'] * X_train['Run_Time']
X_test['Temp_Run_Interaction'] = X_test['Temperature'] * X_test['Run_Time']


scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the model (Logistic Regression)
model = LogisticRegression(class_weight='balanced' , C=0.1, max_iter=100, solver='liblinear')

# Train the model
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"F1-score: {f1:.2f}")

scores = cross_val_score(model, X, y, cv=5, scoring='f1')
print(f"Cross-validation F1-score: {scores.mean():.2f}")




