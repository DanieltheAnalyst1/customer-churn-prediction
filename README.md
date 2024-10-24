---

# Customer Churn Prediction Dashboard

This project provides a **Customer Churn Prediction Tool** built using **Streamlit**, a user-friendly web interface for machine learning models. The dashboard leverages a **Random Forest Classifier** to predict customer churn based on various input factors such as `days_spent`, `total_spend`, `age`, `gender`, and `location`.

> **Disclaimer:**  
> The data used in this project was randomly generated in Python as a sample, intended to demonstrate the capabilities of data analysis. The results are illustrative only. Your business will have its own unique data and outcomes, but the key takeaway is the value of leveraging data-driven insights to reduce churn and improve retention.

### Table of Contents
- [Overview](#overview)
- [Live Demo](#live-demo)
- [Features](#features)
- [Disclaimer](#disclaimer)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model](#model)
- [Visualizations](#visualizations)
- [Technologies Used](#technologies-used)
- [License](#license)

---

## Overview

Customer churn prediction is essential for businesses aiming to retain customers and maintain growth. This dashboard allows users to input specific customer data and predicts whether the customer is likely to churn. The project also includes data summaries and visualizations to help decision-makers understand factors related to customer churn.

For a quick overview and to interact with the tool, check out the **live demo** linked below.

---

## Live Demo

Explore the deployed Customer Churn Prediction Dashboard [here](https://customer-churn-prediction-bdb7vappbwxqlbndynb5qhd.streamlit.app/).

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

## Disclaimer

The data used in this project was randomly generated in Python as a sample, designed to demonstrate what can be achieved with data analysis. The results are purely illustrative. Your business will have its own unique data and outcomes, but the key takeaway is the potential of data-driven insights to reduce churn and improve customer retention.

---

## Project Structure

```bash
.
├── customer_churn_data.csv    # Dataset file
├── churn_prediction.py        # Main Streamlit app file
├── README.md                  # Project documentation
└── requirements.txt           # Python dependencies
```

--- 
