#%%
import torch
import torch.nn as nn
from torch.distributions import Normal
from torchinfo import summary as torch_summary
from math import exp

class Actor(nn.Module):
    """
    Actor. 
    """
    def __init__(self, n_states=3, n_actions=1):
        """
        Initialize the Actor.
        
        Args:
        - n_states (int): Number of state inputs.
        - n_actions (int): Number of action outputs.
        """
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(n_states, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        
        self.mu = nn.Sequential(
            nn.Linear(128, n_actions)
            )
        
        self.std = nn.Sequential(
            nn.Linear(128, n_actions),
            nn.Softplus()
            )

    def forward(self, state):
        """
        Forward pass through the network.
        """
        x = self.network(state)
        mu = self.mu(x)
        std = torch.clamp(self.std(x), min = exp(-20), max = exp(2))
        e = Normal(0, 1).sample(std.shape).to("cuda" if std.is_cuda else "cpu")
        x = mu + e * std
        action = torch.tanh(x)
        log_prob = Normal(mu, std).log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = torch.mean(log_prob, -1).unsqueeze(-1)
        return 2*action, log_prob
    
    
    
class Critic(nn.Module):
    """
    Critic. 
    """
    def __init__(self, n_states=3):
        """
        Initialize the Critic.
        
        Args:
        - n_states (int): Number of state inputs.
        """
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(n_states + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state, action):
        """
        Forward pass through the network.
        """
        x = torch.cat([state, action], dim = -1)
        return self.network(x)
    
    
    
    

if __name__ == "__main__":
    # Initialize and print the Actor model
    actor = Actor()
    print(actor)
    
    # Print the model summary
    print("\nModel Summary:")
    print(torch_summary(actor, (3, 3)))
    
    # Initialize and print the Critic model
    critic = Critic()
    print(critic)
    
    # Print the model summary
    print("\nModel Summary:")
    print(torch_summary(critic, ((3, 3), (3, 1))))
# %%
