import argparse
import torch
from .recommender import LearningPathRecommender


def main():
    """
    Main entry point for the recommendation engine.
    """
    parser = argparse.ArgumentParser(description='Personalized Learning Path Recommender')
    parser.add_argument('--user_id', type=int, required=True, help='User ID')
    parser.add_argument('--mode', choices=['train', 'recommend'], default='recommend', help='Mode of operation')
    parser.add_argument('--n_users', type=int, default=1000, help='Number of users')
    parser.add_argument('--n_items', type=int, default=500, help='Number of content items')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    recommender = LearningPathRecommender(args.n_users, args.n_items, device)

    if args.mode == 'train':
        # Create synthetic training data
        batch_data = [
            {'user_id': 1, 'content_id': 10, 'performance_score': 85, 'engagement_duration': 1200, 'difficulty_level': 3, 'cognitive_style': 0},
            {'user_id': 2, 'content_id': 25, 'performance_score': 70, 'engagement_duration': 900, 'difficulty_level': 2, 'cognitive_style': 1},
            {'user_id': 3, 'content_id': 40, 'performance_score': 90, 'engagement_duration': 1500, 'difficulty_level': 4, 'cognitive_style': 2}
        ]
        recommender.train_policy(batch_data, epochs=1)
        print('Training complete')
    else:
        # Create context for recommendation
        context = {
            'performance_score': 75,
            'engagement_duration': 1000,
            'difficulty_level': 3,
            'cognitive_style': 0
        }
        recommendations = recommender.recommend(args.user_id, context)
        print(f'Recommended content IDs: {recommendations}')

if __name__ == '__main__':
    main()
