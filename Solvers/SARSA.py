# Licensing Information:  You are free to use or extend this codebase for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) inform Guni Sharon at 
# guni@tamu.edu regarding your usage (relevant statistics is reported to NSF).
# The development of this assignment was supported by NSF (IIS-2238979).
# Contributors:
# The core code base was developed by Guni Sharon (guni@tamu.edu).

from collections import defaultdict
import numpy as np
from Solvers.Abstract_Solver import AbstractSolver
from lib import plotting


class Sarsa(AbstractSolver):
    def __init__(self, env, eval_env, options):
        assert str(env.observation_space).startswith("Discrete"), (
            str(self) + " cannot handle non-discrete state spaces"
        )
        assert str(env.action_space).startswith("Discrete") or str(
            env.action_space
        ).startswith("Tuple(Discrete"), (
            str(self) + " cannot handle non-discrete action spaces"
        )
        super().__init__(env, eval_env, options)
        # The final action-value function.
        # A nested dictionary that maps state -> (action -> action-value).
        self.Q = defaultdict(lambda: np.zeros(env.action_space.n))

    def train_episode(self):
        """
        Run one episode of the SARSA algorithm: On-policy TD control.

        Use:
            self.env: OpenAI environment.
            self.epsilon_greedy(state): returns the epsilon-greedy action probabilities for 'state'
            self.sample(probs): samples an action from a probability vector
            self.options.steps: number of steps per episode
            self.options.gamma: Gamma discount factor.
            self.options.alpha: TD learning rate.
            self.Q[state][action]: q value for ('state', 'action')
            self.options.epsilon: Chance the sample a random action. Float betwen 0 and 1.

        """

        # Reset the environment
        state, _ = self.env.reset()
        ################################
        #   YOUR IMPLEMENTATION HERE   #
        ################################

        # Single episode on-policy SARSA loop following (Sutton & Barto, Sec. 6.4)
        gamma = self.options.gamma
        alpha = self.options.alpha

        # Choose A from S using epsilon-greedy policy
        epsilon_greedy_action_probabilities = self.epsilon_greedy(state)
        action = self.sample(epsilon_greedy_action_probabilities)

        for i in range(self.options.steps):
            next_state, reward, done, _ = self.step(action)

            # Choose A' from S' using epsilon-greedy policy
            next_state_epsilon_greedy_action_probabilities = self.epsilon_greedy(next_state)
            next_action = self.sample(next_state_epsilon_greedy_action_probabilities)

            if not done:
                # On-policy target: bootstrap from Q(S', A'), the action we will actually take next
                self.Q[state][action] += alpha * (reward + gamma * self.Q[next_state][next_action] - self.Q[state][action])
            else:
                # Terminal state has value 0, so the target is just the reward
                self.Q[state][action] += alpha * (reward - self.Q[state][action])
                break

            # S <- S', A <- A': carrying A' forward is what keeps the update on-policy
            state = next_state
            action = next_action

    def pull_updates(self):
        raise NotImplementedError

    def __str__(self):
        return "Sarsa"

    def create_greedy_policy(self):
        """
        Creates a greedy policy based on Q values.

        Returns:
            A function that takes a state as input and returns a greedy action.
        """

        def policy_fn(state):
            ################################
            #   YOUR IMPLEMENTATION HERE   #
            ################################
            return np.argmax(self.Q[state])

        return policy_fn

    def epsilon_greedy(self, state):
        """
        Compute the epsilon-greedy action probabilities for 'state', based on
        the current Q-values and epsilon.

        Note: this returns pi(.|s) as a vector, NOT a sampled action. Use
        self.sample(probs) when you need to act on it.

        Use:
            self.env.action_space.n: the size of the action space
            np.argmax(self.Q[state]): action with highest q value
        Returns:
            Probability of taking actions as a vector where each entry is the probability of taking that action
        """
        ################################
        #   YOUR IMPLEMENTATION HERE   #
        ################################

        # Every action gets a base probability of epsilon/number_of_actions;
        # The greedy action gets the remaining 1 - epsilon on top (Sutton & Barto, Sec. 5.4)
        epsilon = self.options.epsilon
        optimal_action = np.argmax(self.Q[state])
        action_space = self.env.action_space.n
        action_probabilities = np.full(action_space, epsilon/action_space)
        action_probabilities[optimal_action] += 1 - epsilon

        return action_probabilities

    def plot(self, stats, smoothing_window=20, final=False):
        plotting.plot_episode_stats(stats, smoothing_window, final=final)
