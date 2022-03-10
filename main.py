import os
from re import A, S
import gym
import torch
import numpy as np 
from agent import Agent


def main():
    env = gym.make("HalfCheetah-v2")
    seed = 0
    expl_noise = 0.1
    # Set seeds
    env.seed(seed)
    env.action_space.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])

    algo = Agent(state_dim, action_dim, max_action, discount=0.99, tau=0.005,\
                    policy_noise=0.2, noise_clip=0.5, policy_freq=2)

    start_timesteps = 25e3

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    for t in range(int(1e6)):
		
        episode_timesteps += 1

		# Select action randomly or according to policy
        if t < start_timesteps:
            action = env.action_space.sample()
        else:
            action = (
                algo.select_action(np.array(state))
                + np.random.normal(0, max_action * expl_noise, size=action_dim)
			).clip(-max_action, max_action)

		# Perform action
        next_state, reward, done, _ = env.step(action) 
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

		# Store data in replay buffer
        algo.cache(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

		# Train agent after collecting sufficient data
        if t >= start_timesteps:
            algo.learn()

        if done: 
			# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
			# Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1 

if __name__ == '__main__':
    main()
