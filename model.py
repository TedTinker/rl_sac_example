#%%
import torch
import torch.nn as nn
from torchinfo import summary as torch_summary

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
            nn.Linear(128, n_actions),
            nn.Tanh()
        )

    def forward(self, state):
        """
        Forward pass through the network.
        """
        return 2*self.network(state)
    
    
    
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
