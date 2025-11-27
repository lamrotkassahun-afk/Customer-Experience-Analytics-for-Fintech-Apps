import pandas as pd
from google_play_scraper import Sort, reviews_all
from tqdm import tqdm
import os
from datetime import datetime

# ----------------- Configuration -----------------
BANK_APPS = {
    'CBE': 'com.combanketh.mobilebanking',
    'BOA': 'com.boa.apollo',
    'Dashen': 'com.dashen.dashensuperapp'
}
N_REVIEWS_PER_BANK = 500  # Target >= 400 per bank (500 for safety)
RAW_DATA_PATH = os.path.join('data', 'raw_bank_reviews.csv')
# -------------------------------------------------

def scrape_reviews(app_id: str, bank_name: str, count: int) -> list:
    """Scrapes a specified number of reviews for a given app ID."""
    print(f"-> Starting scrape for {bank_name} ({app_id})...")
    
    # reviews_all is used to handle pagination and retrieve all available reviews
    # up to the specified limit.
    result = reviews_all(
        app_id,
        lang='en',             # Focus on English reviews
        country='et',          # Scrape from the Ethiopian store
        sort=Sort.NEWEST,      # Sort by newest
        filter_score_with=None # Get all scores (1-5)
    )

    # Limit the result to the desired count (N_REVIEWS_PER_BANK)
    reviews_data = result[:count]
    
    processed_reviews = []
    for review in reviews_data:
        processed_reviews.append({
            'bank_name': bank_name,
            'content': review['content'],
            'score': review['score'],
            'at': review['at'].strftime('%Y-%m-%d %H:%M:%S'),
            'source': 'Google Play Store', 
            # Include other scraped fields for completeness
            'review_id': review['reviewId'],
            'thumbsUpCount': review['thumbsUpCount'],
        })
    
    print(f"-> Successfully scraped {len(processed_reviews)} reviews for {bank_name}.")
    return processed_reviews

if __name__ == '__main__':
    all_reviews = []
    
    # Use tqdm to show a progress bar
    for bank_name, app_id in tqdm(BANK_APPS.items(), desc="Total Scraping Progress"):
        try:
            reviews = scrape_reviews(app_id, bank_name, N_REVIEWS_PER_BANK)
            all_reviews.extend(reviews)
        except Exception as e:
            print(f"Error scraping {bank_name}: {e}")

    # Convert to DataFrame and save raw data
    df_raw = pd.DataFrame(all_reviews)

    print(f"\nTotal raw reviews collected: {len(df_raw)}")
    df_raw.to_csv(RAW_DATA_PATH, index=False)
    print(f"Raw data saved to: {RAW_DATA_PATH}")
