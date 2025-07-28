# 🌍🤖 AI for Sustainable Development: Localized Flash Flood Risk Prediction

## Project Overview

This project addresses **SDG 13 (Climate Action)** and **SDG 11 (Sustainable Cities and Communities)** by developing a machine learning model to predict flash flood risks in small towns. The solution is tailored for smaller communities that are often overlooked but equally vulnerable to climate change impacts.  

  Use the link to view :
                                     **https://flood-prediction-25.streamlit.app/**

## 🎯 Problem Statement

Small towns face unique challenges in flood prediction:
- Limited historical data compared to major cities
- Complex terrain interactions with rainfall patterns
- Lack of sophisticated monitoring infrastructure
- Increasing frequency of extreme weather events due to climate change

## 🧠 AI Solution

Our solution uses **Supervised Learning** models to predict the number of people affected by disasters, using real-world data from SDG indicators.

### Supervised Learning Components:
- **Random Forest Regression**
- **Gradient Boosting Regression**
- **Neural Network Regression**
- **Linear Regression**

## 📊 Features Used

- **GeoAreaName**: Name of the geographic area
- **TimePeriod**: Year or time period
- **Source**: Data source
- **Nature**: Nature of the disaster
- **Reporting Type**: Type of reporting

The target variable is **Value**: the number of people affected by disaster.

## 🛠️ Technical Implementation

### Technologies Used:
- **Python 3.8+**
- **Scikit-learn**: Machine learning algorithms
- **Pandas & NumPy**: Data manipulation
- **Matplotlib & Seaborn**: Data visualization
- **Streamlit**: Interactive web dashboard
- **Plotly**: Interactive plots
- **openpyxl**: Excel file support

## 📁 Project Structure

```
sdg/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── data/                          # Dataset directory (Excel files)
│   ├── Goal11.xlsx
│   └── Goal13.xlsx
├── src/                           # Source code
│   ├── data_preprocessing.py      # Data cleaning and preparation
│   ├── supervised_models.py       # Supervised learning models
│   ├── visualization.py           # Data and results visualization
│   └── app.py                     # Streamlit web app
├── results/                       # Output files (plots, dashboards)
│   └── ...
```

## 🚀 Quick Start

### 1. Install Dependencies
```sh
pip install -r requirements.txt
```

### 2. Run the Interactive Web Dashboard
```sh
streamlit run src/app.py
```
- Open the link provided in your terminal (usually http://localhost:8501).
- Use the “Predict” tab to upload data and get predictions.
- Use the “Dashboard” tab to view model performance, feature importance, and data visualizations.

### 3. (Optional) Generate Batch Visualizations
If you want to generate all plots and dashboards as files:
```sh
python src/visualize_results.py
```
Check the `results/` folder for output files.

## 🖥️ Interactive Web Dashboard

- **Prediction Tab**: Upload your own Excel data or use the test set to get predictions from the trained model.
- **Dashboard Tab**: View model performance (MAE, RMSE, R²), feature importance, predicted vs actual, affected by area, affected over time, and a sample of the raw data.

## 🌐 Deploying Online

You can deploy this app for free on [Streamlit Cloud](https://streamlit.io/cloud):
1. Push your project to GitHub.
2. Sign in to Streamlit Cloud and create a new app from your repo.
3. Set the main file to `src/app.py`.
4. Add any required secrets or environment variables if needed.

## 📈 Model Performance

Model performance is shown in the dashboard tab after training on your data. Metrics include:
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error
- **R²**: Coefficient of Determination

## 🤝 Contributing

This project is open for contributions! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request
4. Follow the code of conduct

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**"AI can be the bridge between innovation and sustainability."**

*This project demonstrates how artificial intelligence can be harnessed to address real-world sustainability challenges, making climate action accessible to communities of all sizes.* 
