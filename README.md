# 🎓 Campus Placement Analytics & Prediction System

An interactive Machine Learning dashboard for analyzing student placement trends and predicting placement outcomes based on academic performance, technical skills, projects, internships, and other student attributes.

## 🚀 Project Overview

This project analyzes a synthetic dataset of 1,000 student records to explore factors associated with campus placement outcomes.

A Random Forest classification model is trained to predict whether a student is likely to be placed based on their profile.

The project also includes an interactive Streamlit dashboard for data visualization and real-time predictions.

## ✨ Features

- 📊 Exploratory Data Analysis
- 📈 Placement trend visualizations
- 🤖 Machine Learning placement prediction
- 🌳 Random Forest Classification
- 🎯 Placement probability prediction
- 🖥️ Interactive Streamlit dashboard
- 📋 Dataset preview
- 📌 Feature importance analysis

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Streamlit
- Joblib

## 📂 Project Structure

```text
campus-placement-analytics/
│
├── data/
│   └── placement_data.csv
│
├── src/
│   ├── generate_dataset.py
│   ├── analyze_data.py
│   ├── visualize_data.py
│   └── train_model.py
│
├── app.py
├── placement_model.pkl
├── requirements.txt
├── README.md
└── .gitignore