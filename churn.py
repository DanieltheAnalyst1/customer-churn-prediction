import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import seaborn as sns  

@st.cache_data  
def load_data():
    data = pd.read_csv('customer_churn_dataa.csv')  
    data['subscription_end_date'] = pd.to_datetime(data['subscription_end_date'])
    data['subscription_start_date'] = pd.to_datetime(data['subscription_start_date'])
    data['days_spent'] = (data['subscription_end_date'] - data['subscription_start_date']).dt.days
    return data

def preprocess_data(data):
    if 'gender' not in data.columns or 'location' not in data.columns: 
        raise ValueError("Required columns are missing from the data.")
    

    data['gender'] = LabelEncoder().fit_transform(data['gender'])
    
    location_mapping = {'Rural': 0, 'Urban': 1, 'Suburban': 2}
    data['location'] = data['location'].map(location_mapping)

    features = ['days_spent', 'total_spend', 'customer_support_tickets', 'age', 'gender', 'location']
    X = data[features]
    y = data['churned']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler

@st.cache_resource  
def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

data = load_data()

X_train, X_test, y_train, y_test, scaler = preprocess_data(data)  
model = train_model(X_train, y_train)

y_pred = model.predict(X_test)  

if len(y_test) > 0 and len(y_pred) > 0:
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
else:
    st.write("No predictions were made.")

st.title("Customer Churn Prediction Dashboard")

tab1, tab2, tab3 = st.tabs(["Churn Prediction", "Data Summary", "Visualizations"])

with tab1:
    st.subheader("Customer Churn Prediction Tool")
    
    days_spent = st.number_input('Days Spent', min_value=0, max_value=365, value=0, step=1)
    customer_support_tickets = st.number_input('Customer Support Tickets', min_value=0, max_value=100, value=0, step=1)
    total_spend = st.number_input('Total Spend ($)', min_value=0.0, max_value=10000.0, value=0.0, step=1.0)
    location = st.selectbox('Location', ['Rural', 'Urban', 'Suburban'])
    age = st.number_input('Age', min_value=18, max_value=100, value=25, step=1)
    gender = st.selectbox('Gender', ['Male', 'Female'])

    if days_spent < 0:
        st.error("Days spent cannot be negative.")
    if customer_support_tickets < 0:
        st.error("Customer support tickets cannot be negative.")
    if total_spend < 0:
        st.error("Total spend cannot be negative.")
    if age < 18 or age > 100:
        st.error("Please enter a valid age.")

    gender_encoded = 1 if gender == 'Male' else 0
    location_mapping = {'Rural': 0, 'Urban': 1, 'Suburban': 2}
    location_encoded = location_mapping[location]

    user_input = pd.DataFrame({
        'days_spent': [days_spent],
        'customer_support_tickets': [customer_support_tickets],
        'total_spend': [total_spend],
        'age': [age],
        'gender': [gender_encoded],
        'location': [location_encoded]
    })

    user_input = user_input[['days_spent', 'total_spend', 'customer_support_tickets', 'age', 'gender', 'location']]

    user_input_scaled = scaler.transform(user_input)

    if st.button('Predict Churn'):
        prediction = model.predict(user_input_scaled)
        if prediction[0] == 1:
            st.write("This customer is **likely to churn**.")
        else:
            st.write("This customer is **unlikely to churn**.")

with tab2:
    st.subheader("Summary Tables")

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

    st.subheader("Churn by Location")
    st.write(churn_location)

with tab3:
    st.subheader("Advanced Visualizations")

    st.subheader("Feature Correlation Heatmap")
    corr_matrix = data.select_dtypes(include=[np.number]).corr() 

    fig, ax = plt.subplots()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    st.subheader("Feature Distributions by Churn Status")

    fig, ax = plt.subplots()
    sns.histplot(data=data, x='total_spend', hue='churned', kde=True, ax=ax)
    ax.set_title("Total Spend Distribution by Churn Status")
    st.pyplot(fig)

    fig, ax = plt.subplots()
    sns.histplot(data=data, x='age', hue='churned', kde=True, ax=ax)
    ax.set_title("Age Distribution by Churn Status")
    st.pyplot(fig)

    st.subheader("Churn Probability by Gender")
    churn_by_gender = data.groupby('gender')['churned'].mean()

    fig, ax = plt.subplots()
    sns.barplot(x=churn_by_gender.index, y=churn_by_gender.values, ax=ax)
    ax.set_title("Churn Probability by Gender")
    st.pyplot(fig)

    st.subheader("Churn Probability by Location")
    churn_by_location = data.groupby('location')['churned'].mean()

    fig, ax = plt.subplots()
    sns.barplot(x=churn_by_location.index, y=churn_by_location.values, ax=ax)
    ax.set_title("Churn Probability by Location")
    st.pyplot(fig)

    st.subheader("Churn Rate by Location")
  
    labels = ['Urban', 'Rural', 'Suburban']
    sizes = [45, 30, 25]  
    colors = ['#ff9999', '#66b3ff', '#99ff99']  
    explode = (0.1, 0, 0)  

    fig, ax = plt.subplots()
    ax.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', 
           startangle=90, colors=colors, shadow=True, 
           wedgeprops={'edgecolor': 'black'}, textprops={'fontsize': 12})

    ax.axis('equal')
    plt.title('Churn Rate by Location', fontsize=14, fontweight='bold')

    st.pyplot(fig)
