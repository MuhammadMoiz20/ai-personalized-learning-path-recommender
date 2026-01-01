import torch
import numpy as np
from typing import List, Dict, Any
from .models import LatentFactorModel, PolicyNetwork, KnowledgeTracer


class LearningPathRecommender:
    """
    Main class for personalized learning path recommendation.
    """
    def __init__(self, n_users: int, n_items: int, device='cpu'):
        self.device = device
        self.latent_model = LatentFactorModel(n_users, n_items).to(device)
        self.policy_network = PolicyNetwork().to(device)
        self.knowledge_tracer = KnowledgeTracer(n_skills=10, n_actions=5)
        self.user_states = {}

    def recommend(self, user_id: int, context: Dict[str, Any]) -> List[int]:
        """
        Recommend next content items based on user context.
        """
        # Extract features from context
        features = torch.tensor([
            context.get('performance_score', 0.0),
            context.get('engagement_duration', 0.0),
            context.get('difficulty_level', 3.0),
            context.get('cognitive_style', 0.0)  # 0=visual, 1=auditory, 2=kinesthetic
        ], dtype=torch.float32).unsqueeze(0).to(self.device)

        # Get policy recommendation
        with torch.no_grad():
            probs = self.policy_network(features)
            top_k = torch.topk(probs, k=3).indices.tolist()

        # Filter by user state
        recommended = [item_id for item_id in top_k]
        return recommended

    def update_user_state(self, user_id: int, content_id: int, performance: float, engagement: float):
        """
        Update user state based on interaction.
        """
        if user_id not in self.user_states:
            self.user_states[user_id] = torch.zeros(10, dtype=torch.float32)

        # Update knowledge state
        state = self.user_states[user_id]
        action = torch.tensor([content_id % 5], dtype=torch.float32)  # Simplified action
        reward = torch.tensor([performance], dtype=torch.float32)
        self.knowledge_tracer.update(state, action, reward)

        # Update latent factors (simplified)
        # Note: LatentFactorModel does not have update_user_state method, so we skip it
        # This is a placeholder for future implementation
        pass

    def train_policy(self, batch_data: List[Dict[str, Any]], epochs=1):
        """
        Train policy network using curriculum learning.
        """
        self.policy_network.train()
        optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=0.001)

        for epoch in range(epochs):
            for data in batch_data:
                features = torch.tensor([
                    data['performance_score'],
                    data['engagement_duration'],
                    data['difficulty_level'],
                    data['cognitive_style']
                ], dtype=torch.float32).unsqueeze(0).to(self.device)

                target = torch.tensor([data['content_id']], dtype=torch.long).to(self.device)

                optimizer.zero_grad()
                probs = self.policy_network(features)
                loss = F.cross_entropy(probs, target)
                loss.backward()
                optimizer.step()

        self.policy_network.eval()

    def get_user_skill_level(self, user_id: int) -> torch.Tensor:
        """
        Get current skill level for user.
        """
        return self.knowledge_tracer.predict(self.user_states[user_id], torch.zeros(5))