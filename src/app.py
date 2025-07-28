import streamlit as st
import pandas as pd
import numpy as np
import os
from data_preprocessing import SDGDataPreprocessor
from supervised_models import SDGFloodRegressor
from visualization import FloodRiskVisualizer

st.set_page_config(page_title="Disaster Impact Prediction Dashboard", layout="wide")
st.title("🌊 Disaster Impact Prediction & Dashboard")

# Ensure results directory exists
os.makedirs('results', exist_ok=True)

# Sidebar: Data selection/upload
st.sidebar.header("Data Input")
data_option = st.sidebar.radio("Choose data source:", ("Use project data", "Upload your own Excel file"))

if data_option == "Use project data":
    filepaths = [
        'data/Goal11.xlsx',
        'data/Goal13.xlsx'
    ]
    preprocessor = SDGDataPreprocessor()
    full_data = preprocessor.load_and_merge(filepaths)
    X_train, X_test, y_train, y_test, feature_names = preprocessor.preprocess_pipeline(filepaths)
    data_loaded = True
elif data_option == "Upload your own Excel file":
    uploaded_file = st.sidebar.file_uploader("Upload Excel file", type=["xlsx"])
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        st.write("Sample of uploaded data:", df.head())
        preprocessor = SDGDataPreprocessor()
        # Save uploaded file to disk for compatibility
        temp_path = 'data/uploaded_data.xlsx'
        df.to_excel(temp_path, index=False)
        filepaths = [temp_path]
        full_data = preprocessor.load_and_merge(filepaths)
        X_train, X_test, y_train, y_test, feature_names = preprocessor.preprocess_pipeline(filepaths)
        data_loaded = True
    else:
        st.warning("Please upload an Excel file to proceed.")
        data_loaded = False
else:
    data_loaded = False

if data_loaded:
    # Train or load model
    reg = SDGFloodRegressor()
    performance_df = reg.train_all_models(X_train, y_train, X_test, y_test, feature_names)
    best_model = reg.best_model
    predictions = best_model.predict(X_test)
    feature_importance = reg.get_feature_importance(top_n=10)
    if feature_importance is None:
        feature_importance = pd.DataFrame({'feature': feature_names, 'importance': [0]*len(feature_names)})

    # Tabs for Prediction and Dashboard
    tab1, tab2 = st.tabs(["🔮 Predict", "📊 Dashboard"])

    with tab1:
        st.header("Make Predictions")
        st.write("You can upload a new Excel file or use the test set for predictions.")
        pred_option = st.radio("Prediction data:", ("Use test set", "Upload new data"))
        if pred_option == "Use test set":
            pred_data = X_test
            pred_true = y_test
            pred_pred = predictions
            st.write("### Test Set Predictions")
            st.write(pd.DataFrame({
                'Actual': pred_true,
                'Predicted': pred_pred
            }).head(20))
            st.scatter_chart(pd.DataFrame({'Actual': pred_true, 'Predicted': pred_pred}))
        else:
            user_file = st.file_uploader("Upload Excel file for prediction", type=["xlsx"], key="pred_upload")
            if user_file is not None:
                user_df = pd.read_excel(user_file)
                st.write("Sample of uploaded data:", user_df.head())
                # Preprocess user data
                user_X, _ = preprocessor.filter_and_prepare(user_df)
                user_X_scaled = preprocessor.scale_features(user_X)
                user_pred = best_model.predict(user_X_scaled)
                st.write("### Predictions on Uploaded Data")
                st.write(pd.DataFrame({'Predicted': user_pred}))

    with tab2:
        st.header("Model Dashboard & Visualizations")
        st.subheader("Model Performance Metrics")
        st.dataframe(performance_df)
        st.subheader("Feature Importance")
        st.bar_chart(feature_importance.set_index('feature'))
        st.subheader("Predicted vs Actual (Test Set)")
        import matplotlib.pyplot as plt
        fig1 = plt.figure()
        FloodRiskVisualizer().plot_predicted_vs_actual(y_test, predictions, save_path=None)
        st.pyplot(fig1)
        st.subheader("Total People Affected by Area (Top 10)")
        fig2 = plt.figure()
        top_areas = full_data.groupby('GeoAreaName')['Value'].sum().sort_values(ascending=False).head(10)
        top_areas.plot(kind='bar', color='skyblue')
        plt.title('Total People Affected by Disaster by Area (Top 10)')
        plt.ylabel('Total Affected')
        plt.tight_layout()
        st.pyplot(fig2)
        st.subheader("Total People Affected Over Time")
        fig3 = plt.figure()
        full_data.groupby('TimePeriod')['Value'].sum().plot()
        plt.title('Total People Affected by Disaster Over Time')
        plt.ylabel('Total Affected')
        plt.xlabel('Year')
        plt.tight_layout()
        st.pyplot(fig3)
        st.subheader("Raw Data Sample")
        st.dataframe(full_data.head(20))
else:
    st.info("Please select or upload data to begin.") 