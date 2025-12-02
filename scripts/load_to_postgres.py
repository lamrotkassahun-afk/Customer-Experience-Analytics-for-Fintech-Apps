import pandas as pd
import psycopg2
from psycopg2 import extras
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import os
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- CONFIGURATION (MUST BE EDITED BY USER) ---
DB_CONFIG = {
    'host': 'localhost',
    'database': 'bank_reviews', # <--- UPDATED DATABASE NAME
    'user': 'postgres',    # <--- CHANGE THIS (e.g., 'postgres')
    'password': 'postgres', # <--- CHANGE THIS (Your password)
    'port': 5432
}
CSV_PATH = os.path.join('data', 'thematic_analysis_results.csv')
TABLE_NAME = 'fintech_reviews'
# -----------------------------------------------

def create_database_if_not_exists(db_config):
    """
    Connects to the default 'postgres' database and attempts to create the target
    database if it does not already exist.
    """
    conn = None
    target_db = db_config['database']
    # Connect to the default 'postgres' system database
    system_db_config = db_config.copy()
    system_db_config['database'] = 'postgres' 

    try:
        logging.info(f"Attempting to connect to 'postgres' to check for database '{target_db}'...")
        conn = psycopg2.connect(**system_db_config)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT) # Required for CREATE DATABASE
        cursor = conn.cursor()

        # Check if the database exists
        # Safely handling database names to avoid SQL injection
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (target_db,))
        if cursor.fetchone() is None:
            logging.warning(f"Database '{target_db}' does not exist. Creating it now...")
            cursor.execute(f"CREATE DATABASE {target_db}")
            logging.info(f"Database '{target_db}' created successfully.")
        else:
            logging.info(f"Database '{target_db}' already exists. Proceeding.")
        return True
    
    except psycopg2.OperationalError as e:
        # This usually means invalid credentials or the PostgreSQL server is not running
        logging.error(f"FATAL: Cannot connect to PostgreSQL server using default credentials: {e}")
        logging.error("ACTION REQUIRED: Ensure your PostgreSQL server is running and DB_CONFIG (user, password) is correct.")
        return False
    except Exception as e:
        logging.error(f"An unexpected error occurred during database creation check: {e}")
        return False
    finally:
        if conn:
            conn.close()


def create_table_if_not_exists(conn):
    """Creates the fintech_reviews table if it doesn't already exist."""
    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
        id SERIAL PRIMARY KEY,
        bank VARCHAR(50) NOT NULL,
        rating INTEGER NOT NULL,
        review_date DATE,
        source VARCHAR(50),
        review_text TEXT NOT NULL,
        sentiment_label VARCHAR(10) NOT NULL,
        sentiment_score NUMERIC(5, 4) NOT NULL,
        theme_id INTEGER NOT NULL
    );
    """
    try:
        cursor = conn.cursor()
        cursor.execute(create_table_sql)
        conn.commit()
        logging.info(f"Table '{TABLE_NAME}' ensured to exist.")
    except psycopg2.Error as e:
        logging.error(f"Error creating table {TABLE_NAME}: {e}")
        conn.rollback()
        raise # Re-raise to stop ingestion

def load_data_to_postgres():
    """
    Connects to PostgreSQL, loads the analyzed CSV data, and inserts it
    into the 'fintech_reviews' table using batch insertion.
    """
    logging.info("--- Starting Task 3: PostgreSQL Data Ingestion ---")
    
    # 0. Check and Create Database
    if not create_database_if_not_exists(DB_CONFIG):
        return # Stop if the database couldn't be created/reached

    conn = None
    try:
        # 1. Load Data
        df = pd.read_csv(CSV_PATH)
        logging.info(f"Data loaded successfully from CSV. Total reviews to upload: {len(df)}")

        # 2. Data Cleaning for DB Insertion
        if 'review' in df.columns and 'review_text' not in df.columns:
            df.rename(columns={'review': 'review_text'}, inplace=True)
        
        df['rating'] = df['rating'].fillna(0).astype(int)
        df['theme_id'] = df['theme_id'].fillna(-1).astype(int)
        # Attempt to handle potential mixed date formats with dayfirst=True
        df['review_date'] = pd.to_datetime(df['date'], errors='coerce', dayfirst=True).dt.date 
        
        # 3. Connect to Target Database and Create Table
        conn = psycopg2.connect(**DB_CONFIG)
        logging.info(f"Successfully connected to target database '{DB_CONFIG['database']}'.")
        
        create_table_if_not_exists(conn)

        # Prepare list of tuples for insertion
        columns = ['bank', 'rating', 'review_date', 'source', 'review_text', 
                   'sentiment_label', 'sentiment_score', 'theme_id']
        data_to_insert = [tuple(row[col] for col in columns) for index, row in df.iterrows()]
        
        # 4. Define SQL INSERT statement
        insert_query = f"""
        INSERT INTO {TABLE_NAME} ({', '.join(columns)}) 
        VALUES %s
        """
        
        # 5. Execute Batch Insertion
        start_time = time.time()
        
        extras.execute_values(
            conn.cursor(), 
            insert_query, 
            data_to_insert, 
            template=None, 
            page_size=1000
        )
        
        conn.commit()
        end_time = time.time()
        
        logging.info(f"\nSUCCESS: All {len(data_to_insert)} reviews ingested into PostgreSQL table: {TABLE_NAME}")
        logging.info(f"Insertion complete in {end_time - start_time:.2f} seconds.")
        logging.info("Task 3 (Database Storage) is complete.")

    except psycopg2.Error as e:
        logging.error(f"Database Error: {e}")
        logging.error("ACTION REQUIRED: Check your DB_CONFIG (user, password, host) and ensure the table schema is correct.")
        if conn:
            conn.rollback()
    except Exception as e:
        logging.error(f"An unexpected error occurred during ingestion: {e}")
    finally:
        if conn:
            conn.close()
            logging.info("Database connection closed.")

if __name__ == '__main__':
    # Place this file in your scripts directory and run it
    load_data_to_postgres()