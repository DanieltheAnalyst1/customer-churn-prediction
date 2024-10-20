import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import seaborn as sns  # Added for visualizations

# Load and preprocess the data
@st.cache_data  # Updated for newer Streamlit versions
def load_data():
    # Use a relative path to the CSV file
    data = pd.read_csv('customer_churn_dataa.csv')  # Ensure the file is in the same directory as this script
    data['subscription_end_date'] = pd.to_datetime(data['subscription_end_date'])
    data['subscription_start_date'] = pd.to_datetime(data['subscription_start_date'])
    data['days_spent'] = (data['subscription_end_date'] - data['subscription_start_date']).dt.days
    return data

# Process data for model
def preprocess_data(data):
    # Check if 'gender' and 'location' columns exist
    if 'gender' not in data.columns or 'location' not in data.columns:  # Updated check
        raise ValueError("Required columns are missing from the data.")
    

    data['gender'] = LabelEncoder().fit_transform(data['gender'])
    
    # Map location to numbers (0 for Rural, 1 for Urban, 2 for Suburban)
    location_mapping = {'Rural': 0, 'Urban': 1, 'Suburban': 2}
    data['location'] = data['location'].map(location_mapping)

    features = ['days_spent', 'total_spend', 'customer_support_tickets', 'age', 'gender', 'location']
    X = data[features]
    y = data['churned']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler

# Train the model
@st.cache_resource  # Updated for newer Streamlit versions
def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

# Load data
data = load_data()

# Preprocess data
X_train, X_test, y_train, y_test, scaler = preprocess_data(data)  # Ensure this line is present

# Train the RandomForest model
model = train_model(X_train, y_train)

# Model evaluation
# Make predictions
y_pred = model.predict(X_test)  # Ensure y_pred is defined here

# Check if y_test and y_pred have valid values before evaluation
if len(y_test) > 0 and len(y_pred) > 0:
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
else:
    st.write("No predictions were made.")

# Streamlit layout
st.title("Customer Churn Prediction Dashboard")

# Enhanced UI layout using tabs
tab1, tab2, tab3 = st.tabs(["Churn Prediction", "Data Summary", "Visualizations"])

with tab1:
    st.subheader("Customer Churn Prediction Tool")
    
    # Improved input validation for each input field
    days_spent = st.number_input('Days Spent', min_value=0, max_value=365, value=0, step=1)
    customer_support_tickets = st.number_input('Customer Support Tickets', min_value=0, max_value=100, value=0, step=1)
    total_spend = st.number_input('Total Spend ($)', min_value=0.0, max_value=10000.0, value=0.0, step=1.0)
    location = st.selectbox('Location', ['Rural', 'Urban', 'Suburban'])
    age = st.number_input('Age', min_value=18, max_value=100, value=25, step=1)
    gender = st.selectbox('Gender', ['Male', 'Female'])

    # Error handling for edge cases
    if days_spent < 0:
        st.error("Days spent cannot be negative.")
    if customer_support_tickets < 0:
        st.error("Customer support tickets cannot be negative.")
    if total_spend < 0:
        st.error("Total spend cannot be negative.")
    if age < 18 or age > 100:
        st.error("Please enter a valid age.")

    # User input for customer data
    # Encode inputs to match the model's expectations
    gender_encoded = 1 if gender == 'Male' else 0
    location_mapping = {'Rural': 0, 'Urban': 1, 'Suburban': 2}
    location_encoded = location_mapping[location]

    # Prepare input data
    user_input = pd.DataFrame({
        'days_spent': [days_spent],
        'customer_support_tickets': [customer_support_tickets],
        'total_spend': [total_spend],
        'age': [age],
        'gender': [gender_encoded],
        'location': [location_encoded]
    })

    # Ensure the columns are in the same order as during training
    user_input = user_input[['days_spent', 'total_spend', 'customer_support_tickets', 'age', 'gender', 'location']]

    # Scale user input
    user_input_scaled = scaler.transform(user_input)

    # Make prediction
    if st.button('Predict Churn'):
        prediction = model.predict(user_input_scaled)
        if prediction[0] == 1:
            st.write("This customer is **likely to churn**.")
        else:
            st.write("This customer is **unlikely to churn**.")

with tab2:
    st.subheader("Summary Tables")
    # Create tables
    Ndays_spend = data.groupby(["churned"]).count()[["days_spent"]]
    Adays_spend = data.groupby("churned")["days_spent"].mean()
    Mindays_spend = data.groupby("churned")["days_spent"].min()
    Maxdays_spend = data.groupby("churned")["days_spent"].max()
    Maxsupport = data.groupby("churned")["customer_support_tickets"].max()
    Minsupport = data.groupby("churned")["customer_support_tickets"].min()
    AvgSupport = data.groupby("churned")["customer_support_tickets"].mean()
    Tspend = data.groupby("churned")["total_spend"].sum()
    Minspend = data.groupby("churned")["total_spend"].min()
    Maxspend = data.groupby("churned")["total_spend"].max()
    Aspend = data.groupby("churned")["total_spend"].mean()
    churn_location = data.groupby(['location', 'churned']).size().reset_index(name='count')

    # Arrange tables in a grid layout
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Days Spent Summary")
        st.write("Number of Days:", Ndays_spend)
        st.write("Average Days:", Adays_spend)
        st.write("Minimum Days:", Mindays_spend)
        st.write("Maximum Days:", Maxdays_spend)

    with col2:
        st.subheader("Support Tickets Summary")
        st.write("Max Support Tickets:", Maxsupport)
        st.write("Min Support Tickets:", Minsupport)
        st.write("Average Support Tickets:", AvgSupport)

    with col3:
        st.subheader("Spending Summary")
        st.write("Total Spend:", Tspend)
        st.write("Minimum Spend:", Minspend)
        st.write("Maximum Spend:", Maxspend)
        st.write("Average Spend:", Aspend)

    # Additional table for churn by location
    st.subheader("Churn by Location")
    st.write(churn_location)

with tab3:
    st.subheader("Advanced Visualizations")

    st.subheader("Feature Correlation Heatmap")
    # Calculate correlation matrix
    corr_matrix = data.select_dtypes(include=[np.number]).corr()  # Select only numeric columns

    # Create a heatmap
    fig, ax = plt.subplots()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    st.subheader("Feature Distributions by Churn Status")

    # Distribution of 'total_spend' by churn
    fig, ax = plt.subplots()
    sns.histplot(data=data, x='total_spend', hue='churned', kde=True, ax=ax)
    ax.set_title("Total Spend Distribution by Churn Status")
    st.pyplot(fig)

    # Distribution of 'age' by churn
    fig, ax = plt.subplots()
    sns.histplot(data=data, x='age', hue='churned', kde=True, ax=ax)
    ax.set_title("Age Distribution by Churn Status")
    st.pyplot(fig)

    # Gender vs Churn Probability
    st.subheader("Churn Probability by Gender")
    churn_by_gender = data.groupby('gender')['churned'].mean()

    fig, ax = plt.subplots()
    sns.barplot(x=churn_by_gender.index, y=churn_by_gender.values, ax=ax)
    ax.set_title("Churn Probability by Gender")
    st.pyplot(fig)

    # Location vs Churn Probability
    st.subheader("Churn Probability by Location")
    churn_by_location = data.groupby('location')['churned'].mean()

    fig, ax = plt.subplots()
    sns.barplot(x=churn_by_location.index, y=churn_by_location.values, ax=ax)
    ax.set_title("Churn Probability by Location")
    st.pyplot(fig)

    # New Pie Chart for Churn Rate by Location
    st.subheader("Churn Rate by Location")
    
    # Example data for churn rate by location
    labels = ['Urban', 'Rural', 'Suburban']
    sizes = [45, 30, 25]  # Sample percentages for churn by location
    colors = ['#ff9999', '#66b3ff', '#99ff99']  # Custom colors
    explode = (0.1, 0, 0)  # 'Exploding' the largest segment for emphasis

    # Create a pie chart
    fig, ax = plt.subplots()
    ax.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', 
           startangle=90, colors=colors, shadow=True, 
           wedgeprops={'edgecolor': 'black'}, textprops={'fontsize': 12})

    # Equal aspect ratio ensures that pie is drawn as a circle
    ax.axis('equal')
    plt.title('Churn Rate by Location', fontsize=14, fontweight='bold')

    # Display the pie chart in Streamlit
    st.pyplot(fig)
