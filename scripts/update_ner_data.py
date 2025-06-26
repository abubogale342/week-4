import pandas as pd
import re
from typing import List, Tuple
import random

def load_processed_data(csv_path: str) -> pd.DataFrame:
    """Load the processed Telegram data from CSV file."""
    try:
        df = pd.read_csv(csv_path)
        if 'Processed_Message' not in df.columns:
            print("Error: 'Processed_Message' column not found in the CSV file.")
            return pd.DataFrame()
        print(f"Loaded {len(df)} processed messages from {csv_path}")
        return df
    except Exception as e:
        print(f"Error loading processed CSV file: {e}")
        return pd.DataFrame()

def extract_entities(text: str) -> List[Tuple[str, str]]:
    """Extract entities from text and return list of (word, tag) tuples."""
    if not isinstance(text, str) or not text.strip():
        return []
    
    # Common patterns for entity extraction
    price_patterns = [
        (r'(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*ብር\b', 'B-PRICE', 'I-PRICE'),
        (r'ዋጋ[\s:፡]*\s*(\d+)\s*ብር', 'B-PRICE', 'I-PRICE'),
        (r'(\d+)\s*ብር', 'B-PRICE', 'I-PRICE'),
    ]
    
    # Initialize words and tags
    words = text.split()
    tags = ['O'] * len(words)
    
    # Function to mark entities
    def mark_entity(start_idx: int, end_idx: int, tag_type: str, i_tag: str):
        tags[start_idx] = f'B-{tag_type}'
        for i in range(start_idx + 1, end_idx):
            tags[i] = f'{i_tag}'
    
    # Apply entity patterns
    for pattern, b_tag, i_tag in price_patterns:
        for match in re.finditer(pattern, text):
            matched_text = match.group(0)
            matched_words = matched_text.split()
            
            # Find the position of the match in the word list
            for i in range(len(words) - len(matched_words) + 1):
                if ' '.join(words[i:i+len(matched_words)]) == matched_text:
                    mark_entity(i, i + len(matched_words), b_tag, i_tag)
                    break
    
    # Simple heuristic for product names (words in all caps or with emoji)
    for i, word in enumerate(words):
        if word.isupper() and len(word) > 1:
            tags[i] = 'B-PRODUCT'
        elif '📌' in word:
            tags[i] = 'B-PRODUCT'
    
    # Simple heuristic for locations (words after location indicators)
    location_indicators = ['📍', 'አድራሻ', 'ላይ', 'በ']
    for i, word in enumerate(words[:-1]):
        if word in location_indicators:
            tags[i+1] = 'B-LOCATION'
    
    return list(zip(words, tags))

def generate_ner_data(df: pd.DataFrame, output_path: str, sample_size: int = 50):
    """Generate NER data and save to file."""
    # Sample messages if needed
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            message = row['Processed_Message']  # Use the preprocessed message
            if not isinstance(message, str) or not message.strip():
                continue
                
            # Extract entities
            labeled_tokens = extract_entities(message)
            
            # Write to file in CONLL format
            for word, tag in labeled_tokens:
                f.write(f"{word}\t{tag}\n")
            f.write("\n")  # Empty line between messages
    
    print(f"Generated NER data saved to {output_path}")

if __name__ == "__main__":
    # Configuration
    CSV_PATH = "../data/processed/cleaned_telegram_data.csv"  # Updated path
    OUTPUT_PATH = "../data/labeled_telegram_product_price_location.txt"
    SAMPLE_SIZE = 50
    
    # Load data and generate NER data
    df = load_processed_data(CSV_PATH)
    if not df.empty:
        generate_ner_data(df, OUTPUT_PATH, SAMPLE_SIZE)
    else:
        print("No data loaded. Please check the CSV file path.")
