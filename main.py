from src.utils import load_data # type: ignore
from src.recommendation import CollaborativeFiltering # type: ignore

def main():
    # Load data
    user_data, item_data = load_data('data/users.csv', 'data/items.csv')

    # Build Collaborative Filtering
    cf_model = CollaborativeFiltering(user_data)
    interaction_matrix = cf_model.build_interaction_matrix()
    print("Interaction Matrix:\n", interaction_matrix)

    similarity_matrix = cf_model.calculate_similarity()
    print("Item Similarity Matrix:\n", similarity_matrix)

    # Recommend items for a user
    user_id = 1
    recommendations = cf_model.recommend(user_id)
    print(f"Recommendations for User {user_id}:\n", recommendations)

if __name__ == "__main__":
    main()
