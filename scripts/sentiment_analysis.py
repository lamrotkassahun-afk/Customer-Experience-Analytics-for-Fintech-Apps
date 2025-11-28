import pandas as pd
from transformers import pipeline
import os
import time
import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ----------------- Configuration -----------------
# Ensure these paths match your actual structure
CLEAN_DATA_PATH = os.path.join('data', 'clean_bank_reviews.csv')
SENTIMENT_RESULTS_PATH = os.path.join('data', 'sentiment_analysis_results.csv')
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
# -------------------------------------------------

def analyze_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies a pre-trained DistilBERT model to classify the sentiment of reviews.
    """
    logging.info(f"Starting sentiment analysis for {len(df)} reviews.")

    if df.empty:
        logging.error("DataFrame is empty. Analysis aborted.")
        return df

    # Check for the required column
    if 'review' not in df.columns:
        logging.error("Required column 'review' not found in DataFrame. Check preprocessing step.")
        return df
        
    # --- Model Loading ---
    try:
        logging.info(f"Loading sentiment model: {MODEL_NAME}...")
        # Use device=0 for GPU acceleration if available, otherwise default to CPU
        # Using a try/except to catch dependency errors (e.g., missing torch/tensorflow)
        sentiment_pipeline = pipeline(
            "sentiment-analysis", 
            model=MODEL_NAME
        )
        logging.info("Model loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading model or running pipeline. Did you run 'pip install transformers torch'?: {e}")
        return df

    start_time = time.time()
    
    # --- Analysis Application ---
    try:
        # The pipeline processes the reviews in batches efficiently.
        results = sentiment_pipeline(df['review'].tolist())
    except Exception as e:
        logging.error(f"Error during sentiment processing: {e}")
        return df
    
    end_time = time.time()
    
    # --- Results Integration ---
    try:
        # Extract results and add them to the DataFrame
        df['sentiment_label'] = [r['label'] for r in results]
        df['sentiment_score'] = [r['score'] for r in results]
    except Exception as e:
        logging.error(f"Error integrating results into DataFrame: {e}")
        return df


    logging.info(f"Sentiment analysis completed in {end_time - start_time:.2f} seconds.")

    # 4. Aggregation and Initial Analysis (Optional but helpful check)
    print("\n--- Initial Sentiment Aggregation by Bank ---")
    if 'bank' in df.columns:
        sentiment_agg = df.groupby('bank')['sentiment_label'].value_counts(normalize=True).mul(100).unstack(fill_value=0).round(2)
        print(sentiment_agg)
    else:
        logging.warning("Column 'bank' not found for aggregation. Skipping print summary.")
    
    return df

if __name__ == '__main__':
    logging.info("--- Starting Sentiment Analysis Script ---")
    
    if not os.path.exists(CLEAN_DATA_PATH):
        logging.error(f"FATAL ERROR: Clean data file not found at {CLEAN_DATA_PATH}.")
        logging.error("Please ensure the file 'clean_bank_reviews.csv' exists in the 'data' directory and run preprocessing first.")
    else:
        try:
            df_clean = pd.read_csv(CLEAN_DATA_PATH)
            
            # Run sentiment analysis
            df_results = analyze_sentiment(df_clean)
            
            # Check if new columns were successfully added before saving
            if 'sentiment_label' in df_results.columns and not df_results.empty:
                # Save the final results with new columns
                df_results.to_csv(SENTIMENT_RESULTS_PATH, index=False)
                logging.info(f"\nSUCCESS: Sentiment analysis results saved to: {SENTIMENT_RESULTS_PATH}")
                logging.info(f"New columns added: {['sentiment_label', 'sentiment_score']}")
            else:
                logging.error("\nSAVE FAILURE: Sentiment columns were not added. Check logs for earlier errors.")

        except pd.errors.EmptyDataError:
             logging.error(f"Error: The file {CLEAN_DATA_PATH} is empty or invalid.")
        except Exception as e:
            logging.error(f"An unexpected error occurred during execution: {e}")