import os
import gym
import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple
from agent import Agent
from plots import plot_results

env: "gym.env" = gym.make("MountainCar-v0")
logging: bool = bool(os.getenv('LOG'))

EPISODES = 1_500
SHOW_EVERY = 500

agent: Agent = Agent(
    env.observation_space.low,
    env.observation_space.high,
    env.action_space.n,
    env.goal_position,
    learning_rate=0.1,
    discount=0.99,
    epsilone=0.95,
    epsilone_decay=0.9,
    min_epsilone=0.1
)

results = []
for episode in range(EPISODES):
    state: Tuple[float, ...] = env.reset()
    done = False
    render = False

    if episode % SHOW_EVERY == 0 and episode != 0:
        print("Episode:", episode)
        # render = True

    log_rewards = 0
    while not done:
        action: int = agent.select_action(state)
        new_state, reward, done, _ = env.step(action)
        log_rewards += reward

        agent.train(state, new_state, action, reward, done, episode, logging)
        agent.update_epsilone()
        state = new_state

        if render and logging:
            env.render()

    results.append(log_rewards)

    if episode > 500 and logging:
        plot_results(results)

env.close()
plot_results(results, save=True)
