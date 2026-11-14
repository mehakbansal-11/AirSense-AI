# 🌫️ AirSense AI

### Air Quality Analysis, Prediction and Pollution Pattern Clustering

AirSense AI is a Python-based machine learning project designed to analyze historical air quality data, estimate the Air Quality Index (AQI), classify air quality into categories, and explore pollution patterns across cities.

## 🎯 Project Objectives

- Analyze historical air quality records.
- Estimate AQI using machine learning.
- Classify air quality into predefined categories.
- Group cities based on pollution patterns.
- Visualize air quality trends and model performance.

## ✨ Features

### 1. Interactive Dashboard
View dataset statistics, average AQI, air quality category distributions, city-wise AQI comparisons, and monthly trends.

### 2. AQI Prediction
Enter pollutant readings to obtain an estimated AQI value using a trained regression model.

### 3. AQI Classification
Predict the air quality category using a machine learning classification model.
 
### 4. City Clustering
Explore pollution patterns by grouping cities using K-Means clustering.

### 5. Model Evaluation
View regression metrics, classification performance, a confusion matrix, and feature importance.

### 6. Dataset Explorer
Preview historical air quality records and download the dataset.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Plotly
- Machine Learning

## 🤖 Machine Learning Models

| Model | Purpose |
|---|---|
| Random Forest Regressor | AQI estimation |
| Random Forest Classifier | AQI category prediction |
| K-Means Clustering | Grouping cities by pollution patterns |

## 📂 Project Structure

```text
AirSense-AI/
├── app.py
├── requirements.txt
├── README.md
└── data/
    ├── city_day_clean.csv
    └── .gitkeep
```

## ⚙️ Installation and Setup

### Prerequisites
- Python installed on your computer
- pip package manager

### Step 1: Clone the repository

```bash
git clone https://github.com/mehakbansal-11/AirSense-AI.git
```

### Step 2: Open the project directory

```bash
cd AirSense-AI
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

## 📊 Model Evaluation

AirSense AI evaluates its regression model using MAE, RMSE, and R². Classification performance is assessed using accuracy, macro F1-score, and a confusion matrix.

The application uses an 80:20 train-test split for model evaluation.

## ⚠️ Limitations

- Predictions are based on historical dataset patterns.
- The application does not provide official live AQI readings.
- Prediction quality depends on the training data and model performance.
- The application should not be used for emergency or medical decisions.

## 👩‍💻 Author

**Mehak**

B.Tech Computer Science Engineering Student  
Aspiring Data Analyst | Python | Excel | SQL | Machine Learning

## 🌱 Future Improvements

- Deploy the application online.
- Add regularly updated air quality data.
- Improve model performance through further experimentation.
- Enhance interactive visualizations.

---

*AirSense AI is an educational machine learning project for air quality analysis and prediction.*
