# 🌫️ AirSense AI

### Air Quality Analysis, Prediction & Pollution Pattern Clustering

AirSense AI is a Python-based machine learning project designed to analyze historical air quality data, estimate the Air Quality Index (AQI), classify air quality into categories, and explore pollution patterns across cities.

The application provides an interactive dashboard that helps users explore air quality records, visualize pollution trends, generate model-based predictions, and understand patterns in air pollution data.

## 🚀 Live Demo

**Try the application here:**

👉 [Open AirSense AI Dashboard](https://airsense-ai-wwmtvgwqwn2xiigrezhafd.streamlit.app/)

## 🎯 Project Objectives

- Analyze historical air quality records.
- Estimate AQI using machine learning.
- Classify air quality into predefined categories.
- Identify pollution patterns using city clustering.
- Visualize air quality trends and model performance.
- Present air quality information through an interactive web application.

## ✨ Key Features

### 📊 1. Interactive Dashboard

Explore air quality data through summary statistics and visualizations, including average AQI, air quality category distributions, city-wise comparisons, and monthly trends.

### 🔮 2. AQI Prediction

Enter pollutant readings to obtain an estimated AQI value using a trained machine learning regression model.

### 🌍 3. AQI Classification

Predict the air quality category using a machine learning classification model.

### 🧩 4. City Clustering

Explore pollution patterns by grouping cities using the K-Means clustering algorithm.

### 📈 5. Model Evaluation

Explore model performance through relevant regression and classification metrics, a confusion matrix, and feature importance visualizations.

### 🗂️ 6. Dataset Explorer

Preview historical air quality records and download the dataset for further exploration.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical computations |
| Scikit-learn | Machine learning models and evaluation |
| Streamlit | Interactive web application |
| Plotly | Interactive data visualizations |

## 🤖 Machine Learning Models

AirSense AI uses three machine learning approaches for different analytical tasks.

| Model | Purpose |
|---|---|
| Random Forest Regressor | AQI estimation |
| Random Forest Classifier | AQI category prediction |
| K-Means Clustering | Grouping observations to explore pollution patterns |

### 📊 Model Evaluation Metrics

The application evaluates its models using relevant performance metrics.

**Regression metrics**
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R-squared (R²)

**Classification metrics**
- Accuracy
- Macro F1-score
- Confusion matrix

The application uses an 80:20 train-test split for model evaluation.

*Actual model performance depends on the dataset, selected features, and training process.*

## 📁 Project Structure

```text
AirSense-AI/
│
├── app.py
├── requirements.txt
├── README.md
│
└── data/
    ├── city_day_clean.csv
    └── .gitkeep
```

## ⚙️ Installation and Setup

Follow these steps to run AirSense AI locally.

### Prerequisites

- Python installed on your computer
- pip package manager
- Git

### Step 1: Clone the repository

```bash
git clone https://github.com/mehakbansal-11/AirSense-AI.git
```

### Step 2: Navigate to the project directory

```bash
cd AirSense-AI
```

### Step 3: Install the required dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the application

```bash
streamlit run app.py
```

The application will start locally and provide a URL that you can open in your web browser.

## 📂 Dataset

The project uses a historical air quality dataset stored at:

`data/city_day_clean.csv`

The dataset is used to explore air quality records and support the machine learning workflows implemented in the application.

Predictions and analytical results depend on the available dataset and the models used by the application.

## ⚠️ Limitations

- Predictions are based on historical dataset patterns and model outputs.
- The application does not provide official live AQI readings.
- Prediction quality depends on the training data, selected features, and model performance.
- Clustering results depend on the data and preprocessing used.
- The application is intended for educational and analytical purposes and should not be used as the sole basis for emergency or medical decisions.

## 🌱 Future Improvements

- Integrate regularly updated air quality data.
- Experiment with additional machine learning approaches.
- Improve model performance through further validation and experimentation.
- Enhance interactive visualizations and pollution trend analysis.
- Expand city-level comparisons and analytical features.

## 👩‍💻 Author

**Mehak**  
B.Tech Computer Science Engineering Student

Interested in Data Analytics, Python, Machine Learning, and building data-driven applications.

🔗 **GitHub:** [mehakbansal-11](https://github.com/mehakbansal-11)  
🔗 **LinkedIn:** [Connect with Mehak](https://www.linkedin.com/in/mehak-bansal-a58280380/)

---

*AirSense AI is an educational machine learning project focused on air quality analysis, prediction, and pollution pattern exploration.*
