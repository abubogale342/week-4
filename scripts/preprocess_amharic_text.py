import unicodedata
import re
from typing import Callable, List, Optional

from etnltk.lang.am import (
    remove_english_chars,
    remove_arabic_chars,
    remove_ethiopic_punct,
    remove_punct,
    remove_special_characters,
    normalize,
    remove_emojis,
    clean_amharic
)

def preprocess_amharic_text(
    text: str,
    keep_abbrev: bool = False,
    custom_pipeline: Optional[List[Callable[[str], str]]] = None,
    normalize_unicode: bool = True,
    remove_english: bool = True,
    remove_arabic: bool = True,
    remove_punctuation: bool = True,
    remove_emojis_flag: bool = True,
    remove_urls: bool = True,
    min_word_length: int = 2,
    keep_numbers: bool = True
) -> str:
    """
    Preprocess Amharic text with options for cleaning and normalization.
    
    Args:
        text: Input text to preprocess
        keep_abbrev: Whether to keep abbreviations (default: False)
        custom_pipeline: List of custom cleaning functions to apply
        normalize_unicode: Whether to normalize Unicode characters (default: True)
        remove_english: Whether to remove English characters (default: True)
        remove_arabic: Whether to remove Arabic characters (default: True)
        remove_punctuation: Whether to remove punctuation (default: True)
        remove_emojis_flag: Whether to remove emojis (default: True)
        remove_urls: Whether to remove URLs (default: True)
        min_word_length: Minimum word length to keep (default: 2)
        keep_numbers: Whether to preserve numbers and numeric patterns (default: True)
        
    Returns:
        Preprocessed text
    """
    """
    Preprocess Amharic text with various cleaning options.
    
    Args:
        text: Input text to preprocess
        keep_abbrev: Whether to keep abbreviations
        custom_pipeline: List of additional cleaning functions to apply
        normalize_unicode: Whether to normalize Unicode characters
        remove_english: Whether to remove English characters
        remove_arabic: Whether to remove Arabic characters
        remove_punctuation: Whether to remove punctuation
        remove_emojis_flag: Whether to remove emojis
        remove_urls: Whether to remove URLs
        min_word_length: Minimum length of words to keep
        keep_numbers: Whether to preserve numbers in the text
        
    Returns:
        Preprocessed text as a string, or empty string on error
    """
    if not isinstance(text, str):
        return ""

    if not text.strip():
        return ""

    try:
        # Make a copy of the text to work with
        processed_text = text
        
        # Clean up URLs and mentions if requested
        if remove_urls:
            # Only remove website URLs, keep other text including phone numbers
            processed_text = re.sub(r'https?://\S+|www\.\S+', '', processed_text)
            # Preserve @mentions that might be part of phone numbers
            processed_text = re.sub(r'(?<![0-9])@\w+', '', processed_text)  # Only remove @mentions not part of numbers
        
        # Apply etnltk's clean_amharic with default pipeline
        processed_text = clean_amharic(
            processed_text,
            abbrev=keep_abbrev,
            pipeline=[]  # We'll apply our pipeline separately
        )
        
        # Normalize Unicode if requested
        if normalize_unicode:
            processed_text = unicodedata.normalize('NFC', processed_text)
            
        # Replace all whitespace characters (including non-breaking spaces) with regular spaces
        processed_text = ' '.join(processed_text.split())
        
        # Remove emojis and special symbols if requested
        if remove_emojis_flag:
            processed_text = remove_emojis(processed_text)
        
        # Apply language-specific cleaning
        if remove_english:
            processed_text = remove_english_chars(processed_text)
        
        if remove_arabic:
            processed_text = remove_arabic_chars(processed_text)
        
        # Remove special characters and normalize
        processed_text = remove_special_characters(processed_text)
        # Remove punctuation if requested
        if remove_punctuation:
            # Keep only Amharic letters, numbers, and basic punctuation
            # First, preserve numbers if needed
            if keep_numbers:
                # Find all numbers and their positions
                number_matches = list(re.finditer(r'\d+', processed_text))
                # Replace numbers with placeholders
                number_placeholders = {}
                for i, match in enumerate(number_matches):
                    placeholder = f'__NUMBER_{i}__'
                    number_placeholders[placeholder] = match.group(0)
                    processed_text = processed_text[:match.start()] + placeholder + processed_text[match.end():]
            
            # Apply punctuation removal
            processed_text = re.sub(
                r'[^\s\u1200-\u137F0-9\-\.,!?]', 
                ' ', 
                processed_text
            )
            
            # Restore numbers if they were preserved
            if keep_numbers:
                for placeholder, number in number_placeholders.items():
                    processed_text = processed_text.replace(placeholder, number)
        
        # Apply custom pipeline if provided
        if custom_pipeline:
            for func in custom_pipeline:
                processed_text = func(processed_text)

        # Final cleanup
        processed_text = re.sub(r'\s+', ' ', processed_text).strip()  # Remove extra whitespace
        
        # Remove short words and clean up
        words = processed_text.split()
        processed_text = ' '.join(word for word in words if len(word) >= min_word_length)
        
        # Process numbers and price patterns
        if keep_numbers:
            # First, normalize all whitespace to single spaces
            processed_text = ' '.join(processed_text.split())
            
            # Initialize placeholders dictionary
            price_placeholders = {}
            
            # Define price patterns to match
            price_patterns = [
                # Pattern 1: Numbers with commas/decimals followed by ብር (with optional separators)
                (r'(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*[\u1366-\u1368\s]*ብር\b', 
                 lambda m, ph=price_placeholders: f' __PRICE_{len(ph)}__ '),
                
                # Pattern 2: Simple numbers followed by ብር (no separators)
                (r'(\d+)\s*ብር\b', 
                 lambda m, ph=price_placeholders: f' __PRICE_{len(ph)}__ '),
                
                # Pattern 3: ዋጋ followed by numbers (with various separators)
                (r'ዋጋ[\s\u1366-\u1368:፡]*\s*(\d+)', 
                 lambda m, ph=price_placeholders: f'ዋጋ __PRICE_{len(ph)}__'),
                
                # Pattern 4: Currency symbols with numbers
                (r'([$€£¥])\s*(\d+)', 
                 lambda m, ph=price_placeholders: f'__PRICE_{len(ph)}__'),
                
                # Pattern 5: Numbers with currency symbols after
                (r'(\d+)\s*([$€£¥])', 
                 lambda m, ph=price_placeholders: f'__PRICE_{len(ph)}__'),
                
                # Pattern 6: Numbers with Ethiopic number indicators
                (r'(\d+)\s*[\u1366-\u1368]', 
                 lambda m, ph=price_placeholders: f'__PRICE_{len(ph)}__')
            ]
            
            # First pass: Find and replace all price patterns
            for pattern, replacement in price_patterns:
                while True:
                    match = re.search(pattern, processed_text)
                    if not match:
                        break
                    placeholder = replacement(match)
                    original = match.group(0)
                    price_placeholders[placeholder.strip()] = original
                    processed_text = processed_text[:match.start()] + placeholder + processed_text[match.end():]
            
            # Normalize spaces after replacement
            processed_text = ' '.join(processed_text.split())
            
            # Step 2: Protect standalone numbers
            number_placeholders = {}
            number_pattern = r'\b\d+\b'
            
            def number_replacer(match):
                placeholder = f' __NUMBER_{len(number_placeholders)}__ '
                number_placeholders[placeholder.strip()] = match.group(0)
                return placeholder
                
            processed_text = re.sub(number_pattern, number_replacer, processed_text)
            
            # Combine all placeholders
            all_placeholders = {**price_placeholders, **number_placeholders}
            
            # Process each token
            tokens = processed_text.split()
            processed_tokens = []
            
            for token in tokens:
                # Check if this is a placeholder
                if token in all_placeholders:
                    processed_tokens.append(all_placeholders[token])
                    continue
                    
                # Check if token contains Amharic characters or is a number
                has_amharic = any('\u1200' <= char <= '\u137F' for char in token)
                has_digits = any(char.isdigit() for char in token)
                
                # Keep tokens that:
                # 1. Are placeholders (handled above)
                # 2. Contain Amharic characters
                # 3. Are numbers (if keep_numbers is True)
                # 4. Are common Amharic punctuation
                if (has_amharic or 
                    (has_digits and keep_numbers) or 
                    token in ['፣', '።', '፡', '፤', '፥', '፦', '፧', '፨']):
                    processed_tokens.append(token)
            
            # Reconstruct the text
            processed_text = ' '.join(processed_tokens)
            
            # Post-processing: Ensure spaces around numbers are preserved
            # This helps with cases like "550ብር" -> "550 ብር"
            if keep_numbers:
                # Add space between numbers and Amharic letters
                processed_text = re.sub(r'(\d+)([\u1200-\u137F])', r'\1 \2', processed_text)
                # Add space between Amharic letters and numbers
                processed_text = re.sub(r'([\u1200-\u137F])(\d+)', r'\1 \2', processed_text)
            
            # Normalize spaces again after processing
            processed_text = ' '.join(processed_text.split())

        return processed_text

    except Exception as e:
        print(f"Error preprocessing text: {e}")
        return ""
