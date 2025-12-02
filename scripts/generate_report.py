import json
import matplotlib.pyplot as plt
import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- CONFIGURATION ---
JSON_PATH = os.path.join('data', 'analytical_results.json')
REPORT_DIR = 'reports'
# ---------------------

def create_reports():
    """
    Loads analytical data and generates three key visualizations for the report.
    """
    if not os.path.exists(REPORT_DIR):
        os.makedirs(REPORT_DIR)
        logging.info(f"Created report directory: {REPORT_DIR}")

    try:
        logging.info("--- Starting Task 4: Report Generation and Visualization ---")
        
        # 1. Load Data
        with open(JSON_PATH, 'r') as f:
            data = json.load(f)
        
        logging.info("Analytical results loaded successfully.")
        
        # Convert lists of dictionaries back to DataFrames
        df_summary = pd.DataFrame(data['overall_summary'])
        df_themes = pd.DataFrame(data['top_10_negative_themes'])
        df_ratings = pd.DataFrame(data['rating_distribution'])

        # --- 2. Visualization 1: Overall Performance Summary (Avg Rating vs. Volume) ---
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Bar chart for Total Reviews (Primary Y-axis)
        bars = ax1.bar(df_summary['bank'], df_summary['total_reviews'], color='darkblue', alpha=0.7, label='Total Reviews')
        ax1.set_xlabel('Bank')
        ax1.set_ylabel('Total Reviews (Volume)', color='darkblue')
        ax1.tick_params(axis='y', labelcolor='darkblue')

        # Line plot for Average Rating (Secondary Y-axis)
        ax2 = ax1.twinx()
        ax2.plot(df_summary['bank'], df_summary['average_rating'], color='red', marker='o', linestyle='--', linewidth=2, label='Average Rating (1-5)')
        ax2.set_ylabel('Average Rating', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_ylim(2.5, 5.0) # Set reasonable limits for rating
        
        fig.suptitle('Overall CX Performance: Review Volume & Average Rating')
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Add a title and legend
        fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
        plt.title('')

        summary_path = os.path.join(REPORT_DIR, 'summary_performance.png')
        plt.savefig(summary_path)
        plt.close()
        logging.info(f"Generated visualization: {summary_path}")


        # --- 3. Visualization 2: Negative Themes (Pain Points) ---
        plt.figure(figsize=(12, 7))
        
        # Combine bank and theme_id for the label
        df_themes['theme_label'] = df_themes['bank'] + ' (Theme ' + df_themes['theme_id'].astype(str) + ')'
        
        plt.bar(df_themes['theme_label'], df_themes['negative_review_count'], color='orange')
        plt.xlabel('Bank and Theme ID')
        plt.ylabel('Count of Negative Reviews (Volume)')
        plt.title('Top 10 Themes Driving Negative Sentiment (Pain Points)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        themes_path = os.path.join(REPORT_DIR, 'negative_themes.png')
        plt.savefig(themes_path)
        plt.close()
        logging.info(f"Generated visualization: {themes_path}")


        # --- 4. Visualization 3: Rating Distribution ---
        # Pivot table for stacked bar chart: rows=bank, columns=rating, values=count
        df_pivot = df_ratings.pivot(index='bank', columns='rating', values='rating_count').fillna(0)
        
        # Stacked bar chart to show distribution
        df_pivot.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='viridis')
        
        plt.title('Customer Rating Distribution by Bank')
        plt.xlabel('Bank')
        plt.ylabel('Total Count of Ratings')
        plt.xticks(rotation=0)
        plt.legend(title='Rating', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        ratings_path = os.path.join(REPORT_DIR, 'rating_distribution.png')
        plt.savefig(ratings_path)
        plt.close()
        logging.info(f"Generated visualization: {ratings_path}")

        logging.info("Task 4 (Visualization and Reporting) is complete.")

    except FileNotFoundError:
        logging.error(f"Error: Analytical results file not found at {JSON_PATH}. Run analyze_from_postgres.py first.")
    except Exception as e:
        logging.error(f"An unexpected error occurred during visualization: {e}")

if __name__ == '__main__':
    create_reports()