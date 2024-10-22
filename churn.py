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
    
    days_spent = st.number_input('Days Spent', min_value=0, max_value=1000, value=0, step=1)
    customer_support_tickets = st.number_input('Customer Support Tickets', min_value=0, max_value=200, value=0, step=1)
    total_spend = st.number_input('Total Spend ($)', min_value=0.0, max_value=100000.0, value=0.0, step=1.0)
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

    summary_df = data.groupby("churned").agg({
        "days_spent": ['count', 'mean', 'min', 'max'],
        "customer_support_tickets": ['mean', 'min', 'max'],
        "total_spend": ['sum', 'min', 'max', 'mean']
    })
    st.write("Data Summary Based on Churned Customers:")
    st.dataframe(summary_df)

    st.subheader("Number of High-risk Customers")
    churn_prob = model.predict_proba(X_test)[:, 1]
    high_risk_customers = X_test[churn_prob > 0.7]
    st.write("Number of customers with a churn probability greater than 70%:", len(high_risk_customers))
    
    churn_location = data.groupby(['location', 'churned']).size().reset_index(name='count')
    st.subheader("Churn by Location")
    st.dataframe(churn_location)

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
    ax.set_title("Total Amount Spent by Customer Distribution by Churn Status")
    ax.set_xlabel("Total Amount Spent by Customer") 
    st.pyplot(fig)

    fig, ax = plt.subplots()
    sns.histplot(data=data, x='age', hue='churned', kde=True, ax=ax)
    ax.set_title("Age of Customer Distribution by Churn Status")
    ax.set_xlabel("Age of Customer")
    st.pyplot(fig)

    st.subheader("Churn Probability by Gender")
    churn_by_gender = data.groupby('gender')['churned'].mean()
    
    for gender_value, gender_label in zip([0, 1], ['Female', 'Male']):
        fig, ax = plt.subplots()
        gender_churn_prob = churn_by_gender[gender_value]
        sns.barplot(x=[gender_label], y=[gender_churn_prob], ax=ax)
        ax.set_title(f"Churn Probability by {gender_label}")
        st.pyplot(fig)

    st.subheader("Churn Probability by Location")
    churn_by_location = data.groupby('location')['churned'].mean()

    for loc_value, loc_label in zip([0, 1, 2], ['Rural', 'Urban', 'Suburban']):
        fig, ax = plt.subplots()
        loc_churn_prob = churn_by_location[loc_value]
        sns.barplot(x=[loc_label], y=[loc_churn_prob], ax=ax)
        ax.set_title(f"Churn Probability by {loc_label}")
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

    churn_prob = model.predict_proba(X_test)[:, 1]
    high_risk_customers = X_test[churn_prob > 0.7]
    st.write("Number of high-risk customers:", len(high_risk_customers))

    st.subheader("Actionable Strategies to Retain Customers")
    st.write("""
        - **Targeted Promotions**: Offer discounts or loyalty rewards to customers in high churn segments (e.g., customers in Urban areas or with high customer support tickets).
        - **Improved Customer Support**: Prioritize outreach to customers with many support tickets to resolve their issues faster.
        - **Tailored Engagement**: Provide personalized offers or incentives to customers with low total spend and high churn probability.
    """)

