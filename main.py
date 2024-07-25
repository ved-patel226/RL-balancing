import gym
import random
from termcolor import cprint
import numpy as npfi
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory


class RL:
    def __init__(self):
        self.env = gym.make("CartPole-v0", render_mode="human")
        self.states = self.env.observation_space.shape[0]
        self.actions = self.env.action_space.n

        cprint(f"States: {self.states}", "green")
        cprint(f"Actions: {self.actions}", "green")

    def random(self):
        episodes = 10
        for episode in range(1, episodes + 1):
            state = self.env.reset()
            done = False
            score = 0

            while not done:
                self.env.render()
                action = random.choice([0, 1])
                n_state, reward, done, truncated, info = self.env.step(action)
                score += reward
            cprint(f"Episode:{episode} Score:{score}", "green", attrs=["bold"])

    def build_model(self, states, actions):
        model = Sequential()
        model.add(Flatten(input_shape=(1, states)))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(actions, activation="linear"))
        return model


def main() -> None:
    rl = RL()
    rl.random()


if __name__ == "__main__":
    main()
