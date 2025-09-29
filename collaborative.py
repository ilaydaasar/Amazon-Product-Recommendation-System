from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd


class CFModel:
    def __init__(self, n_components=50):
        self.n_components = n_components
        self.svd = None
        self.embeddings = None
        self.item_index = None

    def fit(self, user_item_matrix):
        """
        Fits the SVD model on the user-item interaction matrix.

        Args:
            user_item_matrix (pd.DataFrame): A DataFrame with users as index and items as columns.
        """
        n_items = user_item_matrix.shape[1]
        # Ensure n_components is not larger than the number of items minus 1
        n_comp = min(self.n_components, n_items - 1) if n_items > 1 else 1
        self.svd = TruncatedSVD(n_components=n_comp, random_state=42)
        self.embeddings = self.svd.fit_transform(user_item_matrix)
        
        # We need the user-item matrix's column index for future use
        self.item_index = user_item_matrix.columns

    def recommend(self, user_id, user_item_matrix, df_products, top_n=5):
        """
        Recommends products for a specific user based on collaborative filtering.

        Args:
            user_id: The ID of the user to recommend for.
            user_item_matrix (pd.DataFrame): The original user-item matrix.
            df_products (pd.DataFrame): The product information DataFrame.
            top_n (int): The number of top recommendations to return.

        Returns:
            pd.DataFrame: A DataFrame of recommended products with their CF scores.
        """
        if user_id not in user_item_matrix.index:
            return pd.DataFrame()
        
        # Get the row index and embeddings for the target user
        user_idx = user_item_matrix.index.get_loc(user_id)
        user_vec = self.embeddings[user_idx].reshape(1, -1)
        
        # Calculate cosine similarity between the target user and all other users
        sims = cosine_similarity(user_vec, self.embeddings).flatten()
        
        # Find the indices of the top N most similar users (excluding the user themselves)
        # Using a slice like [1:11] gets the top 10 most similar users
        top_users_indices = np.argsort(sims)[::-1][1:11]
        similar_users = user_item_matrix.index[top_users_indices]
        
        # Get the average ratings/interactions of similar users for each product
        rec_scores = user_item_matrix.loc[similar_users].mean().sort_values(ascending=False)
        
        # Identify items the target user has already interacted with (interaction > 0)
        # The loc and boolean indexing combination is very robust here
        already_interacted = user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] > 0].index
        
        # Filter out products the user has already interacted with
        rec_scores = rec_scores[~rec_scores.index.isin(already_interacted)]
        
        # Get the product IDs of the top N recommended items
        top_items = rec_scores.head(top_n).index
        
        # Prepare the final DataFrame with product details and CF scores
        recommendations = df_products.set_index('product_id').loc[top_items].reset_index()
        recommendations['cf_score'] = rec_scores.loc[top_items].values
        
        return recommendations