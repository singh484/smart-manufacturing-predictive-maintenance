import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="IoT Predictive Maintenance Dashboard", layout="wide")

# --- USER BROWSE & UPLOAD FILE WIDGET ---
st.sidebar.header("📁 Data Management")
uploaded_file = st.sidebar.file_uploader("Browse and upload your CSV file", type=["csv"])

@st.cache_data
def load_data(uploaded_file):
    # 1. Check if user uploaded a file via the browser interface
    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
    else:
        # 2. Fallback to default relative path if nothing uploaded yet
        default_path = "../dataset/predictive_maintenance.csv"
        if os.path.exists(default_path):
            raw_df = pd.read_csv(default_path)
        else:
            raw_df = pd.DataFrame()
            
    # Load model predictions if available
    test_pred_path = "../models/test_predictions.csv"
    test_df = pd.read_csv(test_pred_path) if os.path.exists(test_pred_path) else pd.DataFrame()
    
    return raw_df, test_df

# Load data into the app
raw_df, test_df = load_data(uploaded_file)

# App Title & Overview Header
st.title("🏭 Industrial IoT Predictive Maintenance Dashboard")
st.markdown("Monitor telemetry data, machine health diagnostics, and predictive maintenance schedules in real time.")

if not raw_df.empty:
    st.success(f"✅ Dataset loaded successfully! Total records: {raw_df.shape[0]:,}")
    
    # Quick preview metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Machines / Records", f"{raw_df.shape[0]:,}")
    col2.metric("Failure Rate", f"{(raw_df['Target'].mean() * 100):.2f}%" if 'Target' in raw_df.columns else "N/A")
    col3.metric("Total Features", raw_df.shape[1])
    
    st.subheader("📊 Dataset Preview")
    st.dataframe(raw_df.head())
else:
    st.warning("⚠️ Please browse and upload your `predictive_maintenance.csv` file using the sidebar panel on the left to view the dashboard.")