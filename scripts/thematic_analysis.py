import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import os
import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ----------------- Configuration -----------------
SENTIMENT_RESULTS_PATH = os.path.join('data', 'sentiment_analysis_results.csv')
THEMATIC_RESULTS_PATH = os.path.join('data', 'thematic_analysis_results.csv')
NUM_CLUSTERS = 7  # Number of themes to identify across all banks (you can adjust this)
# -------------------------------------------------

def get_top_terms_per_cluster(vectorizer, model, n_terms=5):
    """Prints the top N terms for each cluster."""
    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names_out()
    
    print("\n--- Top Terms Per Thematic Cluster ---")
    
    cluster_themes = {}
    for i in range(NUM_CLUSTERS):
        # Extract the top terms for the current cluster
        top_n_terms = [terms[ind] for ind in order_centroids[i, :n_terms]]
        
        # Format for printing and storage
        theme_key = f"Theme {i+1}"
        cluster_themes[theme_key] = ", ".join(top_n_terms)
        
        print(f"Cluster {i+1}: {cluster_themes[theme_key]}")

    return cluster_themes

def perform_thematic_analysis():
    """
    Loads data, filters for negative sentiment, vectorizes, clusters,
    and saves the results.
    """
    logging.info("--- Starting Thematic Analysis and Clustering ---")
    
    if not os.path.exists(SENTIMENT_RESULTS_PATH):
        logging.error(f"FATAL ERROR: Sentiment results file not found at {SENTIMENT_RESULTS_PATH}.")
        logging.error("Please ensure you have successfully run 'sentiment_analysis.py' first.")
        return

    try:
        # 1. Load Data
        df = pd.read_csv(SENTIMENT_RESULTS_PATH)
        logging.info(f"Data loaded successfully. Total reviews: {len(df)}")

        # 2. Filter for Negative Reviews (Pain Points)
        df_negative = df[df['sentiment_label'] == 'NEGATIVE'].copy()
        logging.info(f"Filtered for negative reviews. N={len(df_negative)}")

        if df_negative.empty:
            logging.warning("No negative reviews found. Analysis skipped.")
            # Still save the original dataframe for consistency
            df.to_csv(THEMATIC_RESULTS_PATH, index=False)
            return

        # 3. Feature Extraction (TF-IDF Vectorization)
        # We use TfidfVectorizer to turn text reviews into numerical features,
        # weighting important keywords highly.
        logging.info("Starting TF-IDF Vectorization...")
        vectorizer = TfidfVectorizer(
            max_df=0.85,          # Ignore terms that appear in too many reviews
            min_df=5,             # Ignore terms that appear in too few reviews
            ngram_range=(1, 2)    # Consider single words and two-word phrases (e.g., 'login failed')
        )
        X = vectorizer.fit_transform(df_negative['review'].fillna(''))
        logging.info(f"Vectorization complete. Matrix shape: {X.shape}")
        # [Image of TF-IDF formula]

        # 4. Clustering (K-Means)
        logging.info(f"Starting K-Means Clustering with K={NUM_CLUSTERS}...")
        # K-Means partitions the data into K distinct clusters (themes)
        model = KMeans(n_clusters=NUM_CLUSTERS, init='k-means++', max_iter=100, n_init=1, random_state=42)
        model.fit(X)
        
        # Get labels and map them to the negative reviews
        cluster_labels = model.labels_
        df_negative.loc[:, 'theme_id'] = cluster_labels

        # 5. Interpret and Print Themes
        theme_mapping = get_top_terms_per_cluster(vectorizer, model, n_terms=6)

        # 6. Merge Results back into the full dataset
        # Initialize a new column for all reviews
        df['theme_id'] = None
        
        # Update only the negative reviews with their theme ID
        df.update(df_negative['theme_id'])
        
        # Convert theme_id to integer (or keep None for positive reviews)
        df['theme_id'] = df['theme_id'].fillna(-1).astype(int)
        
        logging.info("Clustering and results merge complete.")
        
        # 7. Save Final Data
        df.to_csv(THEMATIC_RESULTS_PATH, index=False)
        logging.info(f"\nSUCCESS: Thematic analysis results saved to: {THEMATIC_RESULTS_PATH}")
        logging.info(f"The final dataset contains the new column: 'theme_id'")
        
        # Print a breakdown of the negative themes by bank
        print("\n--- Theme Breakdown (Negative Reviews Only) ---")
        theme_breakdown = df_negative.groupby('bank')['theme_id'].value_counts().unstack(fill_value=0)
        # Rename columns to be more readable
        theme_breakdown.columns = [f"Theme {i+1}" for i in theme_breakdown.columns]
        print(theme_breakdown)


    except Exception as e:
        logging.error(f"An unexpected error occurred during thematic analysis: {e}")

if __name__ == '__main__':
    perform_thematic_analysis()