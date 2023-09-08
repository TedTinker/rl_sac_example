#%% 
import torch
import torch.nn as nn
import torch.optim as optim
import os
import gym as gymnasium
import matplotlib.pyplot as plt

from utils import ReplayBuffer, plot_rewards
from model import Actor, Critic

# Constants and Settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TAU = 0.005
GAMMA = 0.99
EPISODES = 10000
TARGET_ENTROPY = -1
ALPHA = 1
LOG_ALPHA = torch.tensor([0.0], requires_grad = True)

# Model, buffer, and optimizer setup
buffer = ReplayBuffer()
actor = Actor().to(device)
critic = Critic().to(device)
target_critic = Critic().to(device)
target_critic.load_state_dict(critic.state_dict())

actor_opt = optim.Adam(actor.parameters(), lr=0.001)
critic_opt = optim.Adam(critic.parameters(), lr=0.001)
alpha_opt = optim.Adam([LOG_ALPHA], lr=0.001)
criterion = nn.SmoothL1Loss()

# Environment setup
env = gymnasium.make("Pendulum-v1", render_mode = "rgb_array")
episode_rewards = []



def train():
    """
    Train the model using experience from the replay buffer.
    """
    global ALPHA, LOG_ALPHA
    state, action, next_state, reward, done = buffer.sample()
    
    # Predicted values
    state_action_values = critic(state, action)
    
    # Target values
    with torch.no_grad():
        next_action, next_log_prob = actor(next_state)
        if(len(next_action.shape) == 1): next_action = next_action.unsqueeze(-1)
        next_log_prob = next_log_prob.squeeze(1)
        next_state_values = target_critic(next_state, next_action).squeeze(1)
        next_state_values *= ~done
        expected_state_action_values = (next_state_values - ALPHA * next_log_prob) * GAMMA + reward
        
    # Compute loss and optimize
    loss = criterion(state_action_values.squeeze(1), expected_state_action_values)
    critic_opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(critic.parameters(), 100)
    critic_opt.step()
    
    # New actions
    new_action, new_log_prob = actor(state)
    if(len(new_action.shape) == 1): new_action = new_action.unsqueeze(-1)
    
    # Evaluate new actions
    Q = critic(state.detach(), new_action) * ~done - ALPHA * new_log_prob
    Q = -Q
    
    # Optimize
    actor_opt.zero_grad()
    Q.sum().backward()
    torch.nn.utils.clip_grad_value_(actor.parameters(), 100)
    actor_opt.step()
    
    # Optimize alpha for target_entropy
    _, log_pis = actor(state.detach())
    alpha_loss = -(LOG_ALPHA * (log_pis + TARGET_ENTROPY)).mean()
    alpha_opt.zero_grad()
    alpha_loss.backward()
    torch.nn.utils.clip_grad_value_([LOG_ALPHA], 100)
    alpha_opt.step()
    ALPHA = torch.exp(LOG_ALPHA) 
    
    # Soft update target network: \bar{\theta} <- \tau\theta + (1 - \tau) \bar{\theta}
    target_critic_state_dict = target_critic.state_dict()
    critic_state_dict = critic.state_dict()
    for key in critic_state_dict:
        target_critic_state_dict[key] = critic_state_dict[key] * TAU + target_critic_state_dict[key] * (1 - TAU)
    target_critic.load_state_dict(target_critic_state_dict)



def run_episode(render = False, episode_num = 0):
    """
    Execute one episode of the environment.
    """
    state, _ = env.reset()
    state = torch.tensor(state, device=device).unsqueeze(0)
    t = 0
    done = False
    total_reward = 0

    while not done:
        #if render:   
        #    env.render()
        
        t += 1
        with torch.no_grad():
            action, _ = actor(state)
        next_state, reward, done, _, _ = env.step(action)
        next_state = torch.tensor(next_state, device=device).unsqueeze(0)
        total_reward += reward
        buffer.push(state, action, next_state, reward, torch.tensor(done).unsqueeze(0))
        state = next_state
        train()
        
        if done or t > 100:
            episode_rewards.append(total_reward)
            done = True
            

if __name__ == '__main__':
    for episode in range(EPISODES):
        print(episode, end = '... ')
        if episode % 25 == 0:
            try: os.mkdir('plots/{}'.format(episode))
            except: pass
            run_episode(render = True, episode_num = episode)
            plot_rewards(episode_rewards, episode)
        else:
            run_episode()
# %%
