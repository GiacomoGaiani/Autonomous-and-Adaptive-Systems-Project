import numpy as np
import tensorflow as tf
from tqdm import tqdm
import os

class PPOAgent:
    def __init__(self, envs, policy_net, value_net, learning_rate=1e-5, gamma=0.999, lam=0.95, clip_ratio=0.2, entropy_coef=0.01, value_loss_coef=0.5):
        self.envs = envs
        self.policy_net = policy_net
        self.value_net = value_net
        self.policy_optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.value_optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef

    def select_action(self, states, training=False):
        logits = self.policy_net(states, training=training)
        actions = tf.random.categorical(logits, 1)
        return actions.numpy().reshape(-1), logits

    def compute_advantages(self, rewards, values, next_values, dones):
        advantages = np.zeros_like(rewards)
        lastgaelam = 0
        for t in reversed(range(len(rewards))):
            if dones[t]:
                next_non_terminal = 0.0
                next_value = 0
            else:
                next_non_terminal = 1.0
                next_value = next_values[t]
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            advantages[t] = lastgaelam = delta + self.gamma * self.lam * next_non_terminal * lastgaelam
        returns = advantages + values
        return advantages, returns

    def update_policy(self, states, actions, advantages, returns, old_logits):
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        with tf.GradientTape() as tape:
            logits = self.policy_net(states, training=True)
            action_probs = tf.nn.softmax(logits)
            selected_action_probs = tf.reduce_sum(actions * action_probs, axis=1)
            old_action_probs = tf.reduce_sum(actions * tf.nn.softmax(old_logits), axis=1)
            ratios = selected_action_probs / old_action_probs

            clipped_ratios = tf.clip_by_value(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio)
            surrogate1 = ratios * advantages
            surrogate2 = clipped_ratios * advantages
            policy_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))

            entropy = -tf.reduce_mean(tf.reduce_sum(action_probs * tf.math.log(action_probs + 1e-8), axis=1))
            total_loss = policy_loss - self.entropy_coef * entropy

        grads = tape.gradient(total_loss, self.policy_net.trainable_variables)
        self.policy_optimizer.apply_gradients(zip(grads, self.policy_net.trainable_variables))

    def update_value(self, states, returns):
        with tf.GradientTape() as tape:
            values = tf.squeeze(self.value_net(states, training=True))
            value_loss = tf.reduce_mean((returns - values) ** 2)
        
        grads = tape.gradient(value_loss, self.value_net.trainable_variables)
        self.value_optimizer.apply_gradients(zip(grads, self.value_net.trainable_variables))

    def train(self, max_timesteps, batch_size=1024, update_epochs=8, normalize_rewards=[False,(0,10)]):
        train_rewards = [[] for _ in range(len(self.envs))]
        episode_lengths = [[] for _ in range(len(self.envs))]
        timesteps = 0
        pbar = tqdm(total=max_timesteps, desc="Training Progress", unit=" timestep")
        
        # Initialize the states for all environments
        states = [env.reset() for env in self.envs]
        episode_rewards = [0] * len(self.envs)
        episode_timesteps = [0] * len(self.envs)

        while timesteps < max_timesteps:
            batch_states, batch_actions, batch_rewards, batch_dones, batch_next_states, batch_old_logits = [], [], [], [], [], []

            # Collect experiences from all environments in parallel
            for _ in range(batch_size // len(self.envs)):
                states = np.array(states).astype(np.float32)
                actions, logits = self.select_action(states, training=True)
                next_states, rewards, dones = [], [], []

                for i, env in enumerate(self.envs):
                    next_state, reward, done, _ = env.step(actions[i])

                    if normalize_rewards[0]:
                        min_val, max_val = normalize_rewards[1]
                        reward = (reward - min_val) / (max_val - min_val)

                    # Store experiences for the current batch
                    batch_states.append(states[i])
                    batch_actions.append(tf.one_hot(actions[i], env.action_space.n))
                    batch_rewards.append(reward)
                    batch_dones.append(done)
                    batch_next_states.append(next_state)
                    batch_old_logits.append(logits.numpy()[i])

                    # Accumulate episode rewards and reset environment if done
                    episode_rewards[i] += reward
                    episode_timesteps[i] += 1
                    if done:
                        next_state = env.reset()
                        train_rewards[i].append(episode_rewards[i])
                        episode_lengths[i].append(episode_timesteps[i])
                        episode_rewards[i] = 0
                        episode_timesteps[i] = 0

                    next_states.append(next_state)

                # Update the current states
                states = next_states
                timesteps += len(self.envs)
                pbar.update(len(self.envs))

                if timesteps >= max_timesteps:
                    break

            # Train the policy and value networks using the current batch
            states = np.array(batch_states).astype(np.float32)
            next_states = np.array(batch_next_states).astype(np.float32)
            rewards = np.array(batch_rewards, dtype=np.float32)
            dones = np.array(batch_dones, dtype=np.float32)
            old_logits = np.array(batch_old_logits, dtype=np.float32)

            values = tf.squeeze(self.value_net(states, training=True)).numpy()
            next_values = tf.squeeze(self.value_net(next_states, training=True)).numpy()
            advantages, returns = self.compute_advantages(rewards, values, next_values, dones)

            for _ in range(update_epochs):
                self.update_policy(states, batch_actions, advantages, returns, old_logits)
                self.update_value(states, returns)


        pbar.close()

        return train_rewards, episode_lengths

    def test(self, envs, num_episodes):
        num_envs = len(envs)
        rewards = [[] for _ in range(num_envs)]
        episode_lengths = [[] for _ in range(num_envs)]
        pbar = tqdm(total=num_episodes * num_envs, desc="Testing Progress", unit="episode")

        states = [env.reset() for env in envs]
        episode_rewards = [0] * num_envs
        episode_timesteps = [0] * num_envs
        active_envs = list(range(num_envs))

        while active_envs:
            current_states = np.array([states[i] for i in active_envs], dtype=np.float32)
            
            actions, _ = self.select_action(current_states, training=False)
            
            for idx, env_idx in enumerate(active_envs):
                env = envs[env_idx]
                action = actions[idx]
                next_state, reward, done, _ = env.step(action)
                
                episode_rewards[env_idx] += reward
                episode_timesteps[env_idx] += 1
                states[env_idx] = next_state
                
                if done:
                    rewards[env_idx].append(episode_rewards[env_idx])
                    episode_lengths[env_idx].append(episode_timesteps[env_idx])
                    
                    states[env_idx] = env.reset()
                    episode_rewards[env_idx] = 0
                    episode_timesteps[env_idx] = 0
                    pbar.update(1)
                    
                    if len(rewards[env_idx]) >= num_episodes:
                        active_envs.remove(env_idx)
        
        pbar.close()
        return rewards, episode_lengths
    

    def save_models(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.policy_net.save(os.path.join(directory, 'policy_net.keras'))
        self.value_net.save(os.path.join(directory, 'value_net.keras'))
        print(f"Models saved to {directory}")


    def load_models(self, directory):
        policy_net_path = os.path.join(directory, 'policy_net.keras')
        value_net_path = os.path.join(directory, 'value_net.keras')

        if os.path.exists(policy_net_path) and os.path.exists(value_net_path):
            try:
                self.policy_net = tf.keras.models.load_model(policy_net_path)
                self.value_net = tf.keras.models.load_model(value_net_path)
                print('Models loaded successfully.')
            except Exception as e:
                print(f'Error loading models: {e}')
        else:
            print('ERROR: tried to load models from non-existing directory')

