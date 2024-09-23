import numpy as np
from tqdm import tqdm

class RandomAgent:
    def __init__(self, envs):
        self.envs = envs

    def select_action(self, state):
        return [env.action_space.sample() for env in self.envs]

    def test(self, envs, num_episodes=10):
        rewards = [[] for _ in range(len(envs))] 
        episode_lengths = [[] for _ in range(len(envs))]  
        pbar = tqdm(total=num_episodes * len(envs), desc="Testing Progress", unit=" episode")

        states = [env.reset() for env in envs]
        episode_rewards = [0] * len(envs) 
        episode_timesteps = [0] * len(envs)

        while any(len(rewards[i]) < num_episodes for i in range(len(envs))):
            states = np.array(states)
            actions = self.select_action(states)
            next_states, dones = [], []

            for i, env in enumerate(envs):
                if len(rewards[i]) >= num_episodes:
                    continue

                next_state, reward, done, _ = env.step(actions[i])
                next_states.append(next_state)
                episode_rewards[i] += reward
                episode_timesteps[i] += 1

                if done:
                    next_state = env.reset()
                    rewards[i].append(episode_rewards[i])
                    episode_lengths[i].append(episode_timesteps[i])
                    episode_rewards[i] = 0 
                    episode_timesteps[i] = 0

                dones.append(done)

            states = next_states
            pbar.update(sum([done for done in dones]))

        pbar.close()
        return rewards, episode_lengths
