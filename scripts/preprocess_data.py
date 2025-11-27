import pandas as pd
import numpy as np
import os

# ----------------- Configuration -----------------
RAW_DATA_PATH = os.path.join('data', 'raw_bank_reviews.csv')
CLEAN_DATA_PATH = os.path.join('data', 'clean_bank_reviews.csv')
# -------------------------------------------------

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans, normalizes, and selects final columns for the dataset."""
    
    initial_count = len(df)
    print(f"Starting preprocessing with {initial_count} records.")

    # 1. Column Selection and Renaming (to meet the final column names)
    df_clean = df.rename(columns={
        'content': 'review',
        'score': 'rating',
        'at': 'date',
        'bank_name': 'bank'
    })
    
    # Select only the required final columns: review, rating, date, bank, source
    df_clean = df_clean[['review', 'rating', 'date', 'bank', 'source']]

    # 2. Duplicate Handling
    print("Removing duplicates...")
    # Drop duplicates based on the key identifying columns
    df_clean.drop_duplicates(subset=['review', 'rating', 'bank'], inplace=True)
    duplicates_removed = initial_count - len(df_clean)
    print(f"Removed {duplicates_removed} duplicate records.")

    # 3. Missing Data Handling (KPI: <5% missing data)
    print("Handling missing values...")
    
    # Drop rows where the critical 'review' text is missing
    df_clean.dropna(subset=['review'], inplace=True)
    
    # Check total missing data percentage for the KPI
    total_cells = df_clean.size
    missing_cells = df_clean.isnull().sum().sum()
    missing_percentage = (missing_cells / total_cells) * 100
    
    print(f"Total missing data percentage: {missing_percentage:.2f}%")
    if missing_percentage >= 5.0:
         print("WARNING: Missing data KPI (< 5%) might not be met for some columns.")


    # 4. Date Normalization (to YYYY-MM-DD format)
    print("Normalizing 'date' column...")
    try:
        # Convert to datetime objects
        df_clean['date'] = pd.to_datetime(df_clean['date'])
        # Format to YYYY-MM-DD string format
        df_clean['date'] = df_clean['date'].dt.strftime('%Y-%m-%d')
        print("Date normalization successful.")
    except Exception as e:
        print(f"Error during date normalization, keeping original format: {e}")
        
    final_count = len(df_clean)
    print(f"\nPreprocessing complete. Final record count: {final_count}")
    return df_clean

if __name__ == '__main__':
    if not os.path.exists(RAW_DATA_PATH):
        print(f"ERROR: Raw data file not found at {RAW_DATA_PATH}.")
        print("Please run 'scrape_reviews.py' first.")
    else:
        df_raw = pd.read_csv(RAW_DATA_PATH)
        df_cleaned = preprocess_data(df_raw)
        
        # Save the final clean CSV dataset (KPI)
        df_cleaned.to_csv(CLEAN_DATA_PATH, index=False)
        print(f"\nCleaned data successfully saved to: {CLEAN_DATA_PATH}")
