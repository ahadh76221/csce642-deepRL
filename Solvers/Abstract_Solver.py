# Licensing Information:  You are free to use or extend this codebase for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) inform Guni Sharon at 
# guni@tamu.edu regarding your usage (relevant statistics is reported to NSF).
# The development of this assignment was supported by NSF (IIS-2238979).
# Contributors:
# The core code base was developed by Guni Sharon (guni@tamu.edu).

from abc import ABC, abstractmethod
from enum import Enum
import gymnasium as gym
import numpy as np
import time


class AbstractSolver(ABC):
    def __init__(self, env, eval_env, options):
        self.statistics = [0] * len(Statistics)
        self.env = env
        self.eval_env = eval_env
        self.options = options
        self.total_steps = 0
        self.render = False

    def init_stats(self):
        self.statistics[1:] = [0] * (len(Statistics) - 1)

    def sample(self, action_probs):
        """
        Sample a single action from a vector of action probabilities.

        Provided for you. Throughout this codebase an epsilon-greedy (or any
        stochastic) policy returns pi(.|s) as a probability vector, one entry
        per action. Use this helper when you need to act on that vector.

        Param:
            action_probs: a numpy array of length nA summing to 1
        Return:
            action: an int index sampled according to 'action_probs'
        """
        return int(np.random.choice(len(action_probs), p=action_probs))

    def step(self, action):
        """
        Take one step in the environment while keeping track of statistical information
        Param:
            action:
        Return:
            next_state: The next state
            reward: Immediate reward
            done: Is next_state terminal
            info: Gym transition information
        """
        next_state, reward, terminated, truncated, info = self.env.step(action)

        reward += self.calc_reward(next_state)

        # Update statistics
        self.statistics[Statistics.Rewards.value] += reward
        self.statistics[Statistics.Steps.value] += 1
        self.total_steps += 1

        return next_state, reward, terminated or truncated, info

    def calc_reward(self, state):
        # Create a new reward function for the CartPole domain that takes into account the degree of the pole
        try:
            domain = self.env.unwrapped.spec.id
        except:
            domain = self.env.name
        if domain == "CartPole-v1":
            # Gymnasium >=1.0 removed attribute pass-through on wrappers, so the
            # env's own thresholds must be read off .unwrapped.
            base = self.env.unwrapped
            x, x_dot, theta, theta_dot = state
            r1 = (base.x_threshold - abs(x)) / base.x_threshold - 0.8
            r2 = (
                base.theta_threshold_radians - abs(theta)
            ) / base.theta_threshold_radians - 0.5
            return r1 + r2
        return 0

    def run_greedy(self):
        """
        Run the greedy policy.
        """
        policy = self.create_greedy_policy()
        state, _ = self.eval_env.reset()
        if self.options.domain == "FlappyBird-v0":
            self.eval_env.render()
        rewards = 0
        steps = 0
        for _ in range(self.options.steps):
            action = policy(state)
            state, reward, done, _, _ = self.eval_env.step(action)
            if self.options.domain == "FlappyBird-v0":
                self.eval_env.render()
                time.sleep(1 / 30)
            rewards += reward
            steps += 1
            if done:
                break
        return rewards, steps

    def close(self):
        pass

    @abstractmethod
    def train_episode(self):
        pass

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def create_greedy_policy(self):
        pass

    @staticmethod
    def get_out_header():
        ans = "Domain,Solver"
        for s in Statistics:
            ans += "," + s.name
        return ans

    def plot(self, stats, smoothing_window=20, final=False):
        pass

    def get_stat(self):
        try:
            domain = self.env.unwrapped.spec.id
        except:
            domain = self.env.name
        ans = "{},{}".format(domain, str(self))
        for s in Statistics:
            ans += "," + str(self.statistics[s.value])
        return ans


class Statistics(Enum):
    Episode = 0
    Rewards = 1
    Steps = 2
