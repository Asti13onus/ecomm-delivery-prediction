# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the dataset
df = pd.read_csv("C:/Users/91807/E_Commerce.csv")

# Displaying the first few rows of the dataset
print(df.head())

# Checking the shape of the dataset
print(df.shape)

# Checking data types of the columns
print(df.dtypes)

# Dropping the 'ID' column
df.drop(['ID'], axis=1, inplace=True)

# Checking for null/missing values
print(df.isnull().sum())

# Checking for duplicate values
print(df.duplicated().sum())

# Descriptive statistics of the dataset
print(df.describe())

# Displaying the first few rows after preprocessing
print(df.head())

# Visualizing Gender Distribution
plt.figure()
plt.pie(df['Gender'].value_counts(), labels=['F', 'M'], autopct='%1.1f%%', startangle=90)
plt.title('Gender Distribution')
plt.show()

# Visualizing various distributions and counts
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
sns.histplot(df['Weight_in_gms'], ax=ax[0], kde=True).set_title('Weight Distribution')
sns.countplot(x='Product_importance', data=df, ax=ax[1]).set_title('Product Importance')
sns.histplot(df['Cost_of_the_Product'], ax=ax[2], kde=True).set_title('Cost of the Product')
plt.show()

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
sns.countplot(x='Warehouse_block', data=df, ax=ax[0]).set_title('Warehouse Block')
sns.countplot(x='Mode_of_Shipment', data=df, ax=ax[1]).set_title('Mode of Shipment')
sns.countplot(x='Reached.on.Time_Y.N', data=df, ax=ax[2]).set_title('Reached on Time')
plt.show()

fig, ax = plt.subplots(2, 2, figsize=(15, 10))
sns.countplot(x='Customer_care_calls', data=df, ax=ax[0, 0]).set_title('Customer Care Calls')
sns.countplot(x='Customer_rating', data=df, ax=ax[0, 1]).set_title('Customer Rating')
sns.countplot(x='Prior_purchases', data=df, ax=ax[1, 0]).set_title('Prior Purchases')
sns.histplot(x='Discount_offered', data=df, ax=ax[1, 1], kde=True).set_title('Discount Offered')
plt.show()

sns.countplot(x='Gender', data=df, hue='Reached.on.Time_Y.N').set_title('Gender vs Reached on Time')
plt.show()

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
sns.violinplot(x='Reached.on.Time_Y.N', y='Weight_in_gms', data=df, ax=ax[0]).set_title('Weight Distribution')
sns.countplot(x='Product_importance', data=df, ax=ax[1], hue='Reached.on.Time_Y.N').set_title('Product Importance')
sns.violinplot(x='Reached.on.Time_Y.N', y='Cost_of_the_Product', data=df, ax=ax[2]).set_title('Cost of the Product')
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(15, 5))
sns.countplot(x='Warehouse_block', data=df, ax=ax[0], hue='Reached.on.Time_Y.N').set_title('Warehouse Block')
sns.countplot(x='Mode_of_Shipment', data=df, ax=ax[1], hue='Reached.on.Time_Y.N').set_title('Mode of Shipment')
plt.show()

fig, ax = plt.subplots(2, 2, figsize=(15, 10))
sns.countplot(x='Customer_care_calls', data=df, ax=ax[0, 0], hue='Reached.on.Time_Y.N').set_title('Customer Care Calls')
sns.countplot(x='Customer_rating', data=df, ax=ax[0, 1], hue='Reached.on.Time_Y.N').set_title('Customer Rating')
sns.countplot(x='Prior_purchases', data=df, ax=ax[1, 0], hue='Reached.on.Time_Y.N').set_title('Prior Purchases')
sns.violinplot(x='Reached.on.Time_Y.N', y='Discount_offered', data=df, ax=ax[1, 1]).set_title('Discount Offered')
plt.show()

# Label encoding categorical features
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
cols = ['Warehouse_block', 'Mode_of_Shipment', 'Product_importance', 'Gender']

for col in cols:
    df[col] = le.fit_transform(df[col])
    print(col, df[col].unique())

# Heatmap of correlations
plt.figure(figsize=(10, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

# Violin plot for Customer care calls vs Cost of the Product
sns.violinplot(x='Customer_care_calls', y='Cost_of_the_Product', data=df)
plt.show()

# Splitting the dataset into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df.drop('Reached.on.Time_Y.N', axis=1),
                                                    df['Reached.on.Time_Y.N'], test_size=0.2, random_state=0)

# Training and tuning a Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

rfc = RandomForestClassifier()
param_grid = {
    'max_depth': [4, 8, 12, 16],
    'min_samples_leaf': [2, 4, 6, 8],
    'min_samples_split': [2, 4, 6, 8],
    'criterion': ['gini', 'entropy'],
    'random_state': [0, 42]
}

grid = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='accuracy')
grid.fit(X_train, y_train)

# Best parameters
print('Best parameters: ', grid.best_params_)

# Random Forest Classifier with best parameters
rfc = RandomForestClassifier(criterion='gini', max_depth=8, min_samples_leaf=8,
                             min_samples_split=2, random_state=42)
rfc.fit(X_train, y_train)

# Training accuracy
print('Training accuracy: ', rfc.score(X_train, y_train))

# Predicting the test set results
rfc_pred = rfc.predict(X_test)

# Training and tuning a Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()
param_grid = {
    'max_depth': [2, 4, 6, 8],
    'min_samples_leaf': [2, 4, 6, 8],
    'min_samples_split': [2, 4, 6, 8],
    'criterion': ['gini', 'entropy'],
    'random_state': [0, 42]
}

grid = GridSearchCV(estimator=dtc, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='accuracy')
grid.fit(X_train, y_train)

# Best parameters
print('Best parameters: ', grid.best_params_)

# Decision Tree Classifier with best parameters
dtc = DecisionTreeClassifier(criterion='gini', max_depth=6, min_samples_leaf=6,
                             min_samples_split=2, random_state=0, class_weight='balanced')
dtc.fit(X_train, y_train)

# Training accuracy
print('Training accuracy: ', dtc.score(X_train, y_train))

# Predicting the test set results
dtc_pred = dtc.predict(X_test)

# Training and evaluating a Logistic Regression model
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(X_train, y_train)

# Training accuracy
print('Training accuracy: ', lr.score(X_train, y_train))

# Predicting the test set results
lr_pred = lr.predict(X_test)

# Training and evaluating a KNN Classifier
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

# Training accuracy
print('Training accuracy: ', knn.score(X_train, y_train))

# Predicting the test set results
knn_pred = knn.predict(X_test)

# Evaluating the models
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

fig, ax = plt.subplots(2, 2, figsize=(15, 10))
sns.heatmap(confusion_matrix(y_test, rfc_pred), annot=True, cmap='coolwarm', ax=ax[0, 0]).set_title('Random Forest Classifier')
sns.heatmap(confusion_matrix(y_test, dtc_pred), annot=True, cmap='coolwarm', ax=ax[0, 1]).set_title('Decision Tree Classifier')
sns.heatmap(confusion_matrix(y_test, lr_pred), annot=True, cmap='coolwarm', ax=ax[1, 0]).set_title('Logistic Regression')
sns.heatmap(confusion_matrix(y_test, knn_pred), annot=True, cmap='coolwarm', ax=ax[1, 1]).set_title('KNN Classifier')
plt.show()

# Classification report
print('Random Forest Classifier: \n', classification_report(y_test, rfc_pred))
print('Decision Tree Classifier: \n', classification_report(y_test, dtc_pred))
print('Logistic Regression: \n', classification_report(y_test, lr_pred))
print('KNN Classifier: \n', classification_report(y_test, knn_pred))

# Comparing model accuracies
models = ['Random Forest Classifier', 'Decision Tree Classifier', 'Logistic Regression', 'KNN Classifier']
accuracy = [accuracy_score(y_test, rfc_pred), accuracy_score(y_test, dtc_pred),
            accuracy_score(y_test, lr_pred), accuracy_score(y_test, knn_pred)]

plt.figure(figsize=(10, 5))
sns.barplot(x=models, y=accuracy, palette='magma').set_title('Model Comparison')
plt.xticks(rotation=90)
plt.ylabel('Accuracy')
plt.show()

