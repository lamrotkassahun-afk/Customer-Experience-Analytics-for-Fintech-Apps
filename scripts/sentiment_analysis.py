import pandas as pd
from transformers import pipeline
import os
import time

# ----------------- Configuration -----------------
CLEAN_DATA_PATH = os.path.join('data', 'clean_bank_reviews.csv')
SENTIMENT_RESULTS_PATH = os.path.join('data', 'sentiment_analysis_results.csv')
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
# -------------------------------------------------

def analyze_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies a pre-trained DistilBERT model to classify the sentiment of reviews.
    """
    print(f"Loading data from: {CLEAN_DATA_PATH}")
    if df.empty:
        print("Error: DataFrame is empty. Exiting.")
        return df

    # Initialize the sentiment analysis pipeline using the recommended model
    try:
        print(f"Loading sentiment model: {MODEL_NAME}...")
        # Use device=0 for GPU acceleration if available, otherwise default to CPU
        sentiment_pipeline = pipeline(
            "sentiment-analysis", 
            model=MODEL_NAME
        )
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Ensure 'torch' or 'tensorflow' and 'transformers' are installed correctly.")
        return df

    start_time = time.time()
    
    # 1. Apply the sentiment pipeline to the 'review' column
    # The pipeline processes the reviews in batches efficiently.
    results = sentiment_pipeline(df['review'].tolist())
    
    end_time = time.time()
    
    # 2. Extract results and add them to the DataFrame
    df['sentiment_label'] = [r['label'] for r in results]
    df['sentiment_score'] = [r['score'] for r in results]

    print(f"Sentiment analysis completed for {len(df)} reviews in {end_time - start_time:.2f} seconds.")

    # 3. Aggregation and Initial Analysis (Optional but helpful check)
    print("\n--- Initial Sentiment Aggregation by Bank ---")
    sentiment_agg = df.groupby('bank')['sentiment_label'].value_counts(normalize=True).mul(100).unstack(fill_value=0).round(2)
    print(sentiment_agg)
    
    print("\n--- Initial Aggregation by Rating ---")
    rating_agg = df.groupby('rating')['sentiment_label'].value_counts(normalize=True).mul(100).unstack(fill_value=0).round(2)
    print(rating_agg)

    return df

if __name__ == '__main__':
    if not os.path.exists(CLEAN_DATA_PATH):
        print(f"ERROR: Clean data file not found at {CLEAN_DATA_PATH}.")
        print("Please run 'scrape_reviews.py' and 'preprocess_data.py' first.")
    else:
        df_clean = pd.read_csv(CLEAN_DATA_PATH)
        
        # Run sentiment analysis
        df_results = analyze_sentiment(df_clean)
        
        # Save the final results with new columns
        df_results.to_csv(SENTIMENT_RESULTS_PATH, index=False)
        print(f"\nSentiment analysis results successfully saved to: {SENTIMENT_RESULTS_PATH}")