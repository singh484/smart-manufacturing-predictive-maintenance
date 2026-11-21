import pandas as pd
import tkinter as tk
from tkinter import filedialog

def load_and_audit_data():
    # Initialize hidden tkinter window for file browsing
    root = tk.Tk()
    root.attributes('-topmost', True)  # Forces the file browser window to pop up in front
    root.withdraw()                    # Hides the main tkinter background box

    print("📁 Opening file browser... Please select your 'predictive_maintenance.csv' file.")
    
    # Opens the file explorer browser window
    file_path = filedialog.askopenfilename(
        title="Select predictive_maintenance.csv file",
        filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
    )

    # Check if user cancelled or selected a file
    if not file_path:
        print("❌ [ERROR] No file selected. Operation cancelled.")
        return

    print(f"✅ Successfully loaded file from: {file_path}")

    # Load dataset using pandas
    df = pd.read_csv(file_path)

    # Display Industrial Data Quality Audit (Handbook Standards)
    print("\n" + "="*50)
    print("=== INDUSTRIAL DATA QUALITY AUDIT ===")
    print("="*50)
    print(f"Total Rows (Telemetry Records): {df.shape[0]:,}")
    print(f"Total Features (Columns): {df.shape[1]}")
    
    print("\n[1] Missing Values Check:")
    print(df.isnull().sum())
    
    print("\n[2] Target Class Distribution (0 = Normal, 1 = Failure):")
    print(df['Target'].value_counts())
    
    print("\n[3] Failure Type Breakdown:")
    print(df['Failure Type'].value_counts())

    print("\n[4] First 5 Rows Preview:")
    print(df.head())
    print("="*50)

if __name__ == "__main__":
    load_and_audit_data()