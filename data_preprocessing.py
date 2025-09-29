import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_and_prepare(path):
    """
    Loads and preprocesses data for a recommendation system.
    
    Returns:
      - df: Original row-based DataFrame with cleaned data.
      - df_products: A single-row product table grouped by product_id.
      - user_item_matrix: A user x product matrix.
    """
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"Error: The file at '{path}' was not found.")
        return None, None, None, None

    required_cols = ['user_id', 'product_id', 'rating', 'review_content', 'product_name', 'category']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: The CSV file must contain the following columns: {required_cols}")
        return None, None, None, None

    df['user_id'] = df['user_id'].astype(str)
    df['product_id'] = df['product_id'].astype(str)
    df = df[~df['user_id'].str.contains(',')]
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df = df.dropna(subset=['rating'])
    df['review_content'] = df['review_content'].astype(str).fillna("")
    df['category'] = df['category'].astype(str).fillna("")
    if 'rating_count' in df.columns:
        df['rating_count'] = df['rating_count'].astype(str).str.replace(',', '').str.replace('K', '000').astype(float)
    df['product_name'] = df['product_name'].astype(str)
    
    # Veriyi eğitim ve test setlerine ayırma
    if len(df['user_id'].unique()) > 100:
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['user_id'])
    else:
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    df_products = train_df.groupby('product_id').agg(
        product_name=('product_name', 'first'),
        category=('category', 'first'),
        review_content=('review_content', lambda x: " ".join(x.astype(str))),
        avg_rating=('rating', 'mean')
    ).reset_index()

    df_products['text'] = df_products['review_content'] + " " + df_products['category']

    user_item_matrix = train_df.pivot_table(
        index='user_id', 
        columns='product_id', 
        values='rating', 
        fill_value=0.0
    )

    return train_df, test_df, df_products, user_item_matrix