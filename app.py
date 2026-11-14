from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / 'data' / 'city_day_clean.csv'
FEATURES = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene']
BUCKET_ORDER = ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe']

st.set_page_config(page_title='AirSense AI', page_icon='🌫️', layout='wide')

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=['Date'])
    required = ['City', 'Date', 'AQI', 'AQI_Bucket'] + FEATURES
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'Dataset is missing required columns: {missing}')
    df = df.dropna(subset=required).copy()
    df['AQI_Bucket'] = df['AQI_Bucket'].astype(str)
    return df

@st.cache_resource
def train_models():
    df = load_data()
    X = df[FEATURES]
    y_reg = df['AQI'].astype(float)
    Xtr, Xte, yrtr, yrte = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    reg = RandomForestRegressor(n_estimators=120, min_samples_leaf=2, random_state=42, n_jobs=-1)
    reg.fit(Xtr, yrtr)
    rp = np.clip(reg.predict(Xte), 0, 500)
    reg_metrics = {'MAE': mean_absolute_error(yrte, rp), 'RMSE': np.sqrt(mean_squared_error(yrte, rp)), 'R²': r2_score(yrte, rp)}

    y_cls = df['AQI_Bucket']
    Xtrc, Xtec, ytrc, ytec = train_test_split(X, y_cls, test_size=0.2, random_state=42, stratify=y_cls)
    clf = RandomForestClassifier(n_estimators=120, min_samples_leaf=2, random_state=42, n_jobs=-1, class_weight='balanced_subsample')
    clf.fit(Xtrc, ytrc)
    cp = clf.predict(Xtec)
    cls_metrics = {'Accuracy': accuracy_score(ytec, cp), 'Macro F1': f1_score(ytec, cp, average='macro')}
    cm = confusion_matrix(ytec, cp, labels=clf.classes_)

    city_features = df.groupby('City')[['AQI'] + FEATURES].mean()
    scaler = StandardScaler()
    Z = scaler.fit_transform(city_features)
    km = KMeans(n_clusters=4, n_init=20, random_state=42)
    city_features['Cluster'] = km.fit_predict(Z)
    cluster_means = city_features.groupby('Cluster')['AQI'].mean().sort_values(ascending=False)
    rank = {cluster: i for i, cluster in enumerate(cluster_means.index)}
    labels = ['Higher pollution', 'High-moderate pollution', 'Moderate pollution', 'Lower pollution']
    city_features['Pollution group'] = city_features['Cluster'].map(lambda x: labels[rank[x]])
    return reg, clf, reg_metrics, cls_metrics, cm, list(clf.classes_), city_features.reset_index()

try:
    df = load_data()
    reg, clf, reg_metrics, cls_metrics, cm, cls_labels, city_clusters = train_models()
except Exception as exc:
    st.error(f'Could not load the dataset or train models: {exc}')
    st.stop()

st.title('🌫️ AirSense AI')
st.caption('Air Quality Index analysis, prediction, and city pollution-pattern clustering')

with st.sidebar:
    st.header('Project overview')
    st.write('**Dataset:** cleaned city-day air-quality records')
    st.write('**Tasks:** AQI regression · AQI-category classification · city clustering')
    page = st.radio('Navigate', ['Dashboard', 'Predict AQI', 'City Clusters', 'Model Evaluation', 'Dataset'])

if page == 'Dashboard':
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Records', f'{len(df):,}')
    c2.metric('Cities', f'{df.City.nunique()}')
    c3.metric('Average AQI', f'{df.AQI.mean():.1f}')
    c4.metric('Date range', f'{df.Date.min():%Y}–{df.Date.max():%Y}')
    left, right = st.columns(2)
    with left:
        counts = df['AQI_Bucket'].value_counts().reindex(BUCKET_ORDER).fillna(0).reset_index()
        counts.columns = ['AQI category', 'Records']
        st.plotly_chart(px.bar(counts, x='AQI category', y='Records', title='Records by AQI category', color='AQI category'), use_container_width=True)
    with right:
        city_avg = df.groupby('City', as_index=False)['AQI'].mean().sort_values('AQI', ascending=False)
        st.plotly_chart(px.bar(city_avg, x='City', y='AQI', title='Average AQI by city'), use_container_width=True)
    monthly = df.assign(Month=df.Date.dt.to_period('M').dt.to_timestamp()).groupby('Month', as_index=False)['AQI'].mean()
    st.plotly_chart(px.line(monthly, x='Month', y='AQI', title='Monthly average AQI across records'), use_container_width=True)

elif page == 'Predict AQI':
    st.subheader('Estimate AQI from pollutant readings')
    st.info('Enter pollutant values in the same units and conventions as the training dataset. This is a model estimate, not an official live AQI reading.')
    defaults = df[FEATURES].median(numeric_only=True).to_dict()
    with st.form('prediction_form'):
        cols = st.columns(3)
        values = {}
        for i, feature in enumerate(FEATURES):
            with cols[i % 3]:
                values[feature] = st.number_input(feature, min_value=0.0, value=float(defaults[feature]), step=1.0, format='%.2f')
        submitted = st.form_submit_button('Predict')
    if submitted:
        row = pd.DataFrame([[values[f] for f in FEATURES]], columns=FEATURES)
        aqi = float(np.clip(reg.predict(row)[0], 0, 500))
        category = str(clf.predict(row)[0])
        a, b = st.columns(2)
        a.metric('Estimated AQI', f'{aqi:.1f}')
        b.metric('Predicted category', category)
        st.caption('Predictions depend on the historical dataset and model; do not use this prototype for health or emergency decisions.')

elif page == 'City Clusters':
    st.subheader('Cities grouped by average pollution pattern')
    st.write('K-Means groups cities using their mean AQI and mean pollutant readings. Group names are descriptive labels assigned by average AQI rank.')
    st.plotly_chart(px.scatter(city_clusters, x='AQI', y='PM2.5', color='Pollution group', hover_name='City', size='PM10', title='City-level pollution patterns (average values)'), use_container_width=True)
    st.dataframe(city_clusters[['City', 'AQI', 'PM2.5', 'PM10', 'Pollution group']].sort_values('AQI', ascending=False).round(2), use_container_width=True)

elif page == 'Model Evaluation':
    st.subheader('Hold-out test performance')
    st.write('The data is split into training and test sets using an 80:20 random split and random_state=42. Metrics describe this split only.')
    c1, c2, c3 = st.columns(3)
    c1.metric('Regression MAE', f"{reg_metrics['MAE']:.2f}")
    c2.metric('Regression RMSE', f"{reg_metrics['RMSE']:.2f}")
    c3.metric('Regression R²', f"{reg_metrics['R²']:.3f}")
    d1, d2 = st.columns(2)
    d1.metric('Classification accuracy', f"{cls_metrics['Accuracy']:.1%}")
    d2.metric('Classification macro F1', f"{cls_metrics['Macro F1']:.3f}")
    cm_df = pd.DataFrame(cm, index=cls_labels, columns=cls_labels)
    st.write('**Classification confusion matrix** (rows = actual; columns = predicted)')
    st.dataframe(cm_df, use_container_width=True)
    importances = pd.DataFrame({'Pollutant': FEATURES, 'Importance': clf.feature_importances_}).sort_values('Importance', ascending=False)
    st.plotly_chart(px.bar(importances, x='Importance', y='Pollutant', orientation='h', title='Classification feature importance'), use_container_width=True)

elif page == 'Dataset':
    st.subheader('Dataset preview and summary')
    st.write(f'{len(df):,} rows · {df.shape[1]} columns · {df.City.nunique()} cities')
    st.dataframe(df.head(100), use_container_width=True)
    st.write('Missing values by column')
    st.dataframe(df.isna().sum().rename('Missing values').to_frame(), use_container_width=True)
    st.download_button('Download dataset CSV', data=df.to_csv(index=False).encode('utf-8'), file_name='city_day_clean.csv', mime='text/csv')
