import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple
import re

class VendorAnalyzer:
    def __init__(self, data_path: str):
        """Initialize the analyzer with the path to the processed data."""
        self.df = pd.read_csv(data_path)
        self.vendor_metrics = {}
    
    def extract_price(self, text: str) -> float:
        """Extract price from text using regex."""
        if not isinstance(text, str):
            return None
            
        # Look for price patterns like '1000 ብር' or 'ብር 1000'
        price_patterns = [
            r'(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*ብር',  # 1,000 ብር
            r'ብር\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)',  # ብር 1,000
            r'\$(\d+(?:\.\d+)?)',  # $1000
        ]
        
        for pattern in price_patterns:
            match = re.search(pattern, text)
            if match:
                price_str = match.group(1).replace(',', '')
                try:
                    return float(price_str)
                except (ValueError, AttributeError):
                    continue
        return None
    
    def analyze_vendor(self, vendor_name: str) -> Dict:
        """Analyze a single vendor's data."""
        vendor_data = self.df[self.df['Channel Title'] == vendor_name].copy()
        
        if vendor_data.empty:
            return None
        
        # Calculate metrics
        metrics = {
            'total_posts': len(vendor_data),
            'avg_views': vendor_data['Views'].mean(),
            'max_views': vendor_data['Views'].max(),
            'top_post': vendor_data.loc[vendor_data['Views'].idxmax(), 'Message'],
            'posting_frequency': self._calculate_posting_frequency(vendor_data),
            'prices': []
        }
        
        # Extract prices
        for msg in vendor_data['Message'].dropna():
            price = self.extract_price(msg)
            if price is not None:
                metrics['prices'].append(price)
        
        # Calculate price metrics
        if metrics['prices']:
            metrics['avg_price'] = np.mean(metrics['prices'])
            metrics['min_price'] = min(metrics['prices'])
            metrics['max_price'] = max(metrics['prices'])
        else:
            metrics['avg_price'] = metrics['min_price'] = metrics['max_price'] = None
        
        # Calculate lending score (custom formula)
        metrics['lending_score'] = self._calculate_lending_score(metrics)
        
        return metrics
    
    def _calculate_posting_frequency(self, vendor_data) -> float:
        """Calculate average posts per week."""
        if 'Date' not in vendor_data.columns:
            return 0
            
        vendor_data['Date'] = pd.to_datetime(vendor_data['Date'])
        date_range = (vendor_data['Date'].max() - vendor_data['Date'].min()).days
        
        if date_range == 0:
            return len(vendor_data)
        return (len(vendor_data) * 7) / date_range  # Posts per week
    
    def _calculate_lending_score(self, metrics: Dict) -> float:
        """Calculate a custom lending score (0-100)."""
        score = 0
        
        # Normalize metrics (assuming some reasonable ranges)
        posting_freq = min(metrics.get('posting_frequency', 0) / 10, 1)  # Max 10 posts/week = 100%
        avg_views = min(metrics.get('avg_views', 0) / 1000, 1)  # Max 1000 views = 100%
        
        # Weighted score (adjust weights as needed)
        score = (
            (posting_freq * 0.3) +  # Posting frequency (30% weight)
            (avg_views * 0.5) +      # Engagement (50% weight)
            (0.2 if metrics.get('avg_price') and metrics['avg_price'] > 1000 else 0.1)  # Higher price = better (20% weight)
        ) * 100  # Scale to 0-100
        
        return min(max(score, 0), 100)  # Ensure score is between 0-100
    
    def generate_scorecard(self, top_n: int = 10) -> pd.DataFrame:
        """Generate a vendor scorecard."""
        vendors = self.df['Channel Title'].unique()
        scorecard = []
        
        for vendor in vendors[:top_n]:
            metrics = self.analyze_vendor(vendor)
            if metrics:
                scorecard.append({
                    'Vendor': vendor,
                    'Avg. Views/Post': round(metrics.get('avg_views', 0), 1),
                    'Posts/Week': round(metrics.get('posting_frequency', 0), 1),
                    'Avg. Price (ETB)': round(metrics.get('avg_price', 0), 2) if metrics.get('avg_price') else 'N/A',
                    'Lending Score': round(metrics.get('lending_score', 0), 1)
                })
        
        return pd.DataFrame(scorecard).sort_values('Lending Score', ascending=False)

if __name__ == "__main__":
    # Example usage
    analyzer = VendorAnalyzer("../data/processed/cleaned_telegram_data.csv")
    scorecard = analyzer.generate_scorecard()
    print("\nVendor Scorecard:")
    print(scorecard.to_string(index=False))
