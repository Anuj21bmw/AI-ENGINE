import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

class CollaborativeFiltering:
    def __init__(self, user_data):
        """
        Initialize Collaborative Filtering model.
        """
        self.user_data = user_data
        self.interaction_matrix = None
        self.similarity_matrix = None

    def build_interaction_matrix(self):
        """
        Create a user-item interaction matrix.
        """
        self.interaction_matrix = self.user_data.pivot_table(
            index='user_id', columns='item_id', values='interaction', fill_value=0
        )
        return self.interaction_matrix

    def calculate_similarity(self):
        """
        Compute item-item similarity matrix using cosine similarity.
        """
        sparse_matrix = csr_matrix(self.interaction_matrix.values)
        self.similarity_matrix = cosine_similarity(sparse_matrix.T)
        return pd.DataFrame(
            self.similarity_matrix,
            index=self.interaction_matrix.columns,
            columns=self.interaction_matrix.columns
        )

    def recommend(self, user_id, top_n=3):
        """
        Generate top-N recommendations for a given user.
        """
        if self.similarity_matrix is None:
            raise ValueError("Similarity matrix not computed. Call calculate_similarity() first.")

        user_interactions = self.interaction_matrix.loc[user_id]

        scores = self.similarity_matrix.dot(user_interactions.values)
        scores /= np.array([np.abs(self.similarity_matrix).sum(axis=1)])

        item_scores = pd.Series(scores, index=self.interaction_matrix.columns)
        return item_scores[user_interactions == 0].sort_values(ascending=False).head(top_n)
