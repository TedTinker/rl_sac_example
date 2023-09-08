import random
from collections import namedtuple, deque
import torch
import matplotlib.pyplot as plt



# Define Transition namedtuple representing one time-step
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayBuffer:
    """
    A buffer to store state transitions for reinforcement learning.
    """
    def __init__(self, capacity=10000):
        """Initialize a buffer with a given capacity."""
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, next_state, reward, done):
        """Store a state transition in the buffer."""
        self.memory.append(Transition(state, action, next_state, reward, done))

    def sample(self, batch_size=128):
        """Randomly sample a batch of transitions from the buffer."""
        transitions = self.memory if len(self.memory) < batch_size else random.sample(self.memory, batch_size)
        batch = Transition(*zip(*transitions))
        
        state = torch.cat(batch.state)
        action = torch.cat(batch.action)
        next_state = torch.cat(batch.next_state)
        reward = torch.cat(batch.reward)
        done = torch.cat(batch.done)
        
        return state, action, next_state, reward, done

    def __len__(self):
        """Return the number of transitions stored in the buffer."""
        return len(self.memory)
    
    

def plot_rewards(episode_rewards, episode_num):
    """Plot the rewards of episodes (light blue) and their rolling average (red) over 100 episodes."""
    rewards_t = torch.tensor(episode_rewards, dtype=torch.float)
    
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    plt.plot(rewards_t.numpy(), color = (0, 0, 1, .2))
    
    if len(rewards_t) >= 100:
        means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy(), color = (1, 0, 0, 1))
    
    plt.savefig("plots/{}/rewards.png".format(episode_num))
    plt.show()
    plt.close()




