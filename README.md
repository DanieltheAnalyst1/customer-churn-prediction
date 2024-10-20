### GitHub Repository Documentation: Customer Churn Prediction Dashboard

---

# Customer Churn Prediction Dashboard

This project provides a **Customer Churn Prediction Tool** built using **Streamlit**, a user-friendly web interface for machine learning models. The dashboard leverages **Random Forest Classifier** to predict customer churn based on various input factors such as `days_spent`, `total_spend`, `age`, `gender`, and `location`.

### Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model](#model)
- [Visualizations](#visualizations)
- [Technologies Used](#technologies-used)
- [License](#license)

---

## Overview

Customer churn prediction is essential for businesses aiming to retain customers and maintain growth. This dashboard allows users to input specific customer data, and it predicts whether the customer is likely to churn. The project also includes data summaries and visualizations to help decision-makers understand factors related to customer churn.

---

## Features

### 1. **Churn Prediction Tool**  
The main functionality of the dashboard allows users to input customer details and predict whether they are likely to churn using a trained Random Forest Classifier.

### 2. **Data Summary Tables**  
Provides detailed data summaries related to churn, including:
   - Days spent with the company
   - Customer support tickets
   - Spending habits
   - Churn by location

### 3. **Advanced Visualizations**  
Includes insightful visualizations such as:
   - Correlation heatmap between features
   - Distribution of age and spending by churn status
   - Churn rate by gender and location

### 4. **Interactive UI**  
The user-friendly interface lets users input customer data, check summaries, and explore visual insights without the need for any technical background.

---

## Project Structure

```bash
.
├── customer_churn_dataa.csv   # Dataset file
├── churn_prediction.py        # Main Streamlit app file
├── README.md                  # Project documentation
└── requirements.txt           # Required dependencies
```

---

## Installation

1. Clone this repository:

    ```bash
    git clone https://github.com/your_username/churn-prediction-dashboard.git
    ```

2. Navigate to the project directory:

    ```bash
    cd churn-prediction-dashboard
    ```

3. Create a virtual environment and activate it:

    ```bash
    python -m venv env
    source env/bin/activate  # For Windows: env\Scripts\activate
    ```

4. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

1. Ensure that the dataset `customer_churn_dataa.csv` is placed in the project directory.

2. Run the Streamlit application:

    ```bash
    streamlit run churn_prediction.py
    ```

3. The dashboard should now be accessible at `http://localhost:8501/`. You can input customer data and check predictions, view summary statistics, and explore visualizations.

---

## Model

- **Model Type**: Random Forest Classifier
- **Features Used**: 
   - `days_spent`: The number of days a customer has spent with the company.
   - `total_spend`: Total amount the customer has spent.
   - `customer_support_tickets`: Number of tickets raised by the customer.
   - `age`, `gender`, and `location`: Basic demographic information.

The model was trained using a 70-30 train-test split, and the features were scaled using `StandardScaler` to improve performance.

---

## Visualizations

The dashboard includes the following visualizations:

1. **Correlation Heatmap**: Visualizes the correlation between numerical features.
2. **Distribution of `total_spend` and `age` by churn**: Compares spending and age between churned and non-churned customers.
3. **Churn Probability by Gender and Location**: Bar charts showing the probability of churn based on demographic features.
4. **Churn Rate by Location**: A pie chart highlighting the churn rate across different locations (Urban, Rural, Suburban).

---

## Technologies Used

- **Python**: Main programming language.
- **Streamlit**: For building the web app.
- **Pandas**: For data manipulation.
- **Scikit-learn**: For machine learning.
- **Matplotlib & Seaborn**: For data visualizations.

---
