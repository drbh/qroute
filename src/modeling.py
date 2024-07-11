import torch

from sentence_transformers import SentenceTransformer
import torch.nn as nn
import time

print("Loading embedding model...")
start_time = time.time()
encoder_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print(f"Model loaded in {time.time() - start_time:.2f} seconds")

# hyperparameters
# LEARNING_RATE = 0.001
LEARNING_RATE = 0.1  # changed for testing to learn faster
ENTROPY_WEIGHT = 0.01
VALUE_WEIGHT = 0.5

# state and action dims
STATE_DIM = 384  # embedding size
NUM_ACTIONS = 5  # num available models


class PolicyValueNetwork(nn.Module):
    def __init__(self, state_dim, num_actions):
        super(PolicyValueNetwork, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU()
        )
        self.policy_head = nn.Linear(64, num_actions)
        self.value_head = nn.Linear(64, 1)

    def forward(self, state):
        shared_output = self.shared_layers(state)
        policy_logits = self.policy_head(shared_output)
        value = self.value_head(shared_output)
        return policy_logits, value

    #
    def encode(self, input_string: str) -> torch.Tensor:
        embedding = encoder_model.encode([input_string])
        return torch.FloatTensor(embedding)

    def get_state_embedding(self, input_string: str) -> torch.Tensor:
        embedding = self.encode(input_string)
        flattened = embedding.squeeze(0)
        # round to 2 decimal places to simplify state space
        return torch.round(flattened * 100) / 100
