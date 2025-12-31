import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentFactorModel(nn.Module):
    """
    Latent factor model for user and content embeddings.
    """
    def __init__(self, n_users, n_items, latent_dim=16):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, latent_dim)
        self.item_embedding = nn.Embedding(n_items, latent_dim)
        self.latent_dim = latent_dim

    def forward(self, user_ids, item_ids):
        """
        Compute dot product between user and item embeddings.
        """
        user_embed = self.user_embedding(user_ids)
        item_embed = self.item_embedding(item_ids)
        return (user_embed * item_embed).sum(dim=1)


class PolicyNetwork(nn.Module):
    """
    Policy network for reinforcement learning.
    """
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class KnowledgeTracer:
    """
    Bayesian network for knowledge tracing.
    """
    def __init__(self, n_skills=10, n_actions=5):
        self.n_skills = n_skills
        self.n_actions = n_actions
        self.transition_probs = torch.randn(n_skills, n_skills)
        self.emission_probs = torch.randn(n_skills, n_actions)

    def predict(self, state, action):
        """
        Predict next state given current state and action.
        """
        return torch.sigmoid(self.transition_probs @ state + self.emission_probs @ action)

    def update(self, state, action, reward):
        """
        Update model parameters based on feedback.
        """
        # Simple update rule
        self.transition_probs += 0.01 * reward * state.unsqueeze(1)
        self.emission_probs += 0.01 * reward * action.unsqueeze(1)
        return self.transition_probs, self.emission_probs