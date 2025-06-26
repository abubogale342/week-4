import pandas as pd
from preprocess_amharic_text import preprocess_amharic_text
from pathlib import Path
import os

def clean_and_save_data():
    # Read the CSV file
    input_path = 'data/telegram_data.csv'
    output_dir = 'data/processed'
    output_path = os.path.join(output_dir, 'cleaned_telegram_data.csv')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("Reading data...")
    df = pd.read_csv(input_path)
    
    # Remove rows where Message is null
    print("\nRemoving rows with null messages...")
    df_cleaned = df.dropna(subset=['Message']).copy()
    
    # Preprocess the messages
    print("Preprocessing Amharic text (this may take a while)...")
    df_cleaned['Processed_Message'] = df_cleaned['Message'].apply(
        lambda x: preprocess_amharic_text(x) if pd.notnull(x) else ""
    )
    
    # Save the cleaned data
    print(f"\nSaving cleaned data to {output_path}...")
    df_cleaned.to_csv(output_path, index=False)
    
    # Display final dataset information
    print("\n=== Cleaned Dataset Information ===")
    print(f"Number of rows after cleaning: {len(df_cleaned)}")
    print(f"Number of columns: {len(df_cleaned.columns)}")
    print("\nFirst 5 processed messages:")
    pd.set_option('display.max_colwidth', 500)
    print(df_cleaned[['Message', 'Processed_Message']].head().to_string())
    
    print(f"\nCleaned data saved to: {output_path}")

if __name__ == "__main__":
    clean_and_save_data()
