import psycopg2
import pandas as pd
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- CONFIGURATION (MUST BE EDITED BY USER) ---
# NOTE: This configuration must match the credentials used in load_to_postgres.py
DB_CONFIG = {
    'host': 'localhost',
    'database': 'bank_reviews', 
    'user': 'postgres',    # <--- CHANGE THIS (e.g., 'postgres')
    'password': 'postgres', # <--- CHANGE THIS (Your password)
    'port': 5432
}
TABLE_NAME = 'fintech_reviews'
# -----------------------------------------------

def run_analytical_queries():
    """
    Connects to PostgreSQL, executes key analytical queries, and prints the results.
    """
    conn = None
    results = {}

    try:
        logging.info("--- Starting Task 4: Analytical Query Execution ---")
        
        # Connect to Database
        conn = psycopg2.connect(**DB_CONFIG)
        logging.info(f"Successfully connected to target database '{DB_CONFIG['database']}'.")

        # --- Query 1: Overall Summary (Average Rating & Review Count per Bank) ---
        query_1_sql = f"""
        SELECT 
            bank, 
            COUNT(id) AS total_reviews,
            ROUND(AVG(rating), 2) AS average_rating,
            SUM(CASE WHEN sentiment_label = 'POSITIVE' THEN 1 ELSE 0 END) AS positive_count,
            SUM(CASE WHEN sentiment_label = 'NEGATIVE' THEN 1 ELSE 0 END) AS negative_count
        FROM {TABLE_NAME}
        GROUP BY bank
        ORDER BY average_rating DESC;
        """
        query_1_df = pd.read_sql(query_1_sql, conn)
        results['overall_summary'] = query_1_df.to_dict('records')
        logging.info("\n[1] Overall Performance Summary (Bank, Rating, Sentiment):")
        logging.info(query_1_df.to_string(index=False))

        # --- Query 2: Theme-Sentiment Breakdown for Pain Points (NEGATIVE Reviews) ---
        # This query identifies which themes drive the most negative sentiment,
        # helping pinpoint the most critical issues.
        query_2_sql = f"""
        SELECT 
            bank, 
            theme_id,
            COUNT(id) AS negative_review_count,
            ROUND(AVG(sentiment_score), 4) AS avg_negative_score
        FROM {TABLE_NAME}
        WHERE sentiment_label = 'NEGATIVE'
        GROUP BY bank, theme_id
        HAVING COUNT(id) >= 10 -- Focus on themes with significant negative volume
        ORDER BY negative_review_count DESC
        LIMIT 10; -- Top 10 negative themes overall
        """
        query_2_df = pd.read_sql(query_2_sql, conn)
        results['top_10_negative_themes'] = query_2_df.to_dict('records')
        logging.info("\n[2] Top 10 Themes Driving Negative Sentiment (Pain Points):")
        logging.info(query_2_df.to_string(index=False))

        # --- Query 3: Rating Distribution Analysis (Count of each rating per Bank) ---
        query_3_sql = f"""
        SELECT 
            bank, 
            rating, 
            COUNT(id) AS rating_count
        FROM {TABLE_NAME}
        GROUP BY bank, rating
        ORDER BY bank, rating;
        """
        query_3_df = pd.read_sql(query_3_sql, conn)
        results['rating_distribution'] = query_3_df.to_dict('records')
        logging.info("\n[3] Rating Distribution (Count of 1-star, 2-star, etc., per Bank):")
        logging.info(query_3_df.to_string(index=False))

        logging.info("\nTask 4 (Analytical Queries) is complete.")
        
        # Save results to a JSON file for the next step (Visualization)
        with open('data/analytical_results.json', 'w') as f:
            json.dump(results, f, indent=4)
        logging.info("Analytical results saved to data/analytical_results.json.")


    except psycopg2.Error as e:
        logging.error(f"Database Error: {e}")
        logging.error("ACTION REQUIRED: Ensure your PostgreSQL server is running and DB_CONFIG (user, password) is correct.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
    finally:
        if conn:
            conn.close()
            logging.info("Database connection closed.")

if __name__ == '__main__':
    run_analytical_queries()