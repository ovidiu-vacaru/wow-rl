from environment.environment import WowEnvironment
from api.injector import get_xyz_coords
from neural_network.policies import SimplePolicy
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import torch
import os

def reinforce_agent(number_of_episodes=1000, update_frequency=32, load_pretrained=False):
    environment = WowEnvironment()
    policy = SimplePolicy(4)
    
    # Load pretrained weights if specified and the file exists
    if load_pretrained and os.path.exists("model.pth"):
        policy.load_state_dict(torch.load("model.pth"))
    
    optimizer = Adam(policy.parameters(), lr=1e-3)
    
    for episode in range(1, number_of_episodes + 1):
        environment.starting_position()
        rewards = []
        log_probs = []

        for _ in range(100):
            action_probs = policy.forward(environment.state)
            m = Categorical(action_probs)
            action = m.sample().item()
            log_prob = m.log_prob(torch.tensor(action))
            reward, done = environment.step(action)
            rewards.append(reward)
            log_probs.append(log_prob)

            if done:
                break
        
        rewards_to_go = compute_rewards_to_go(rewards)
        policy_loss = compute_policy_loss(log_probs, rewards_to_go)
        
        update_policy(optimizer, policy_loss)
            
        print(f"Episode: {episode}, Total Reward: {sum(rewards)}")

    torch.save(policy.state_dict(), "model.pth")

def compute_rewards_to_go(rewards):
    rewards_to_go = []
    rtg = 0
    for r in reversed(rewards):
        rtg = r + rtg
        rewards_to_go.insert(0, rtg)
    return rewards_to_go

def compute_policy_loss(log_probs, rewards_to_go):
    # Using the computed baseline
    baseline = sum(rewards_to_go) / len(rewards_to_go)
    adjusted_rewards = [rtg - baseline for rtg in rewards_to_go]
    return torch.mean(torch.stack([-log_prob * adjusted_reward for log_prob, adjusted_reward in zip(log_probs, adjusted_rewards)]))

def update_policy(optimizer, policy_loss):
    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()
