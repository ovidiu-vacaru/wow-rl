import torch
from torch.optim import Adam
from torch.distributions import Categorical
from environment.environment import WowEnvironment
from neural_network.policies import SimplePolicy, CriticPolicy

def ppo(episodes, ppo_epochs=4, clip_epsilon=0.2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    env = WowEnvironment()
    actor = SimplePolicy(4).to(device)
    critic = CriticPolicy().to(device)
    GAMMA = 0.99

    optimizer_actor = Adam(actor.parameters(), lr=0.001)
    optimizer_critic = Adam(critic.parameters(), lr=0.001)
    
    mse_loss = torch.nn.MSELoss()

    for episode in range(1, episodes + 1):
        env.reset()
        state = env.state.to(device)
        stored_values = []
        log_probs = []
        rewards_list = []
        old_log_probs = []

        for step in range(100):
            action_probs = actor(state)
            m = Categorical(action_probs)
            action = m.sample()
            log_prob = m.log_prob(action)

            next_state, reward, done = env.step(action.item())
            next_state = next_state.to(device)

            log_probs.append(log_prob)
            old_log_probs.append(log_prob.detach())
            rewards_list.append(reward)
            stored_values.append((state, action, next_state))

            state = next_state

            if done:
                break

        rewards = torch.tensor(rewards_list, dtype=torch.float32).to(device)
        states, _, next_states = zip(*stored_values)
        states, next_states = torch.stack(states).squeeze(), torch.stack(next_states).squeeze()

        # Calculate advantages and TD targets outside of the PPO loop
        with torch.no_grad():
            next_state_values = critic(next_states).squeeze()
        if done:
            next_state_values[-1] = 0.0
        td_targets = rewards + GAMMA * next_state_values

        advantages = td_targets - critic(states).squeeze()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

        for _ in range(ppo_epochs):
            current_log_probs = torch.stack(log_probs)
            old_log_probs_stack = torch.stack(old_log_probs)
            ratio = (current_log_probs - old_log_probs_stack).exp()

            # PPO clipping
            surrogate_obj = ratio * advantages
            clipped_obj = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
            actor_loss = -torch.min(surrogate_obj, clipped_obj).mean()

            critic_loss = mse_loss(critic(states).squeeze(), td_targets)

            optimizer_actor.zero_grad()
            actor_loss.backward(retain_graph=True)
            optimizer_actor.step()

            optimizer_critic.zero_grad()
            critic_loss.backward(retain_graph=True)
            optimizer_critic.step()

        print(f"Episode: {episode}, Total Reward: {sum(rewards_list)}")
