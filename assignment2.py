# Student ID: 13890353
# Student Name: Michael

import os  # Import os module to handle file paths
import streamlit as st  # Import Streamlit library for building the web app
import pandas as pd  # Import pandas library for data loading and manipulation
from sklearn.model_selection import train_test_split  # Import function to split dataset into train and test sets
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier  # Import Random Forest and Extra Trees classifiers
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree classifier
from sklearn.metrics import accuracy_score, precision_score, f1_score  # Import evaluation metrics for classification

st.title("White Wine Classification")  # Display the main title of the web application

file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "winequality-white.csv")  # Build absolute path to the dataset file
df = pd.read_csv(file_path, sep=";")  # Read the white wine CSV dataset using semicolon separator
st.subheader("Dataset Table")  # Display subheader for the dataset preview table
st.dataframe(df.head(10))  # Show the first 10 rows of the dataset as an interactive table

st.subheader("Quality Distribution")  # Display subheader for the quality distribution chart
st.bar_chart(df["quality"].value_counts().sort_index())  # Plot a bar chart showing sample counts per quality score

X = df.drop("quality", axis=1)  # Select the 11 physicochemical attributes as input features
y = df["quality"]  # Select the quality column as the target output variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Split data into 80% training and 20% testing sets

models = {  # Create a dictionary containing three classification models
    "Random Forest": RandomForestClassifier(random_state=42),  # Initialize Random Forest classifier with fixed random state
    "Extra Trees": ExtraTreesClassifier(random_state=42),  # Initialize Extra Trees classifier with fixed random state
    "Decision Tree": DecisionTreeClassifier(random_state=42),  # Initialize Decision Tree classifier with fixed random state
}  # Close the models dictionary

choice = st.sidebar.selectbox("Choose Model", list(models.keys()))  # Add a sidebar dropdown widget to choose a model
model = models[choice]  # Retrieve the selected classification model from the dictionary
model.fit(X_train, y_train)  # Train the selected model using the training data
y_pred = model.predict(X_test)  # Predict quality scores on the test dataset

st.subheader("Model Evaluation")  # Display subheader for model evaluation metrics
st.write("Accuracy:", round(accuracy_score(y_test, y_pred), 4))  # Calculate and display rounded accuracy score
st.write("Precision:", round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4))  # Calculate and display weighted precision score
st.write("F1 Score:", round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4))  # Calculate and display weighted F1 score

inputs = {}  # Initialize an empty dictionary to store user feature inputs
for col in X.columns:  # Loop through each of the 11 wine attributes
    inputs[col] = st.sidebar.slider(col, float(df[col].min()), float(df[col].max()), float(df[col].mean()))  # Create a sidebar slider widget for each attribute

prediction = model.predict(pd.DataFrame([inputs]))[0]  # Predict the white wine quality from user inputs
st.subheader("Classification Result")  # Display subheader for the prediction result
st.success(f"Predicted Wine Quality: {prediction}")  # Display the predicted wine quality score in a success box