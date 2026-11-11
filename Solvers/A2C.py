# Licensing Information:  You are free to use or extend this codebase for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) inform Guni Sharon at 
# guni@tamu.edu regarding your usage (relevant statistics is reported to NSF).
# The development of this assignment was supported by NSF (IIS-2238979).
# Contributors:
# The core code was developed by Guni Sharon (guni@tamu.edu).
# The PyTorch code was developed by Sheelabhadra Dey (sheelabhadra@tamu.edu).

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.optim import Adam

from Solvers.Abstract_Solver import AbstractSolver
from lib import plotting


class ActorCriticNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes):
        super().__init__()
        sizes = [obs_dim] + hidden_sizes + [act_dim]
        self.layers = nn.ModuleList()
        # Shared layers
        for i in range(len(sizes) - 2):
            self.layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        # Actor head layers
        self.layers.append(nn.Linear(hidden_sizes[-1], act_dim))
        # Critic head layers
        self.layers.append(nn.Linear(hidden_sizes[-1], 1))

    def forward(self, obs):
        x = torch.cat([obs], dim=-1)
        for i in range(len(self.layers) - 2):
            x = F.relu(self.layers[i](x))
        # Actor head
        probs = F.softmax(self.layers[-2](x), dim=-1)
        # Critic head
        value = self.layers[-1](x)

        return torch.squeeze(probs, -1), torch.squeeze(value, -1)


class A2C(AbstractSolver):
    def __init__(self, env, eval_env, options):
        super().__init__(env, eval_env, options)
        # Create actor-critic network
        self.actor_critic = ActorCriticNetwork(
            env.observation_space.shape[0], env.action_space.n, self.options.layers
        )
        self.policy = self.create_greedy_policy()

        self.optimizer = Adam(self.actor_critic.parameters(), lr=self.options.alpha)

    def create_greedy_policy(self):
        """
        Creates a greedy policy.


        Returns:
            A function that takes an observation as input and returns a vector
            of action probabilities.
        """

        def policy_fn(state):
            state = torch.as_tensor(state, dtype=torch.float32)
            return torch.argmax(self.actor_critic(state)[0]).detach().numpy()

        return policy_fn

    def select_action(self, state):
        """
        Selects an action given state.

        Returns:
            The selected action (as an int)
            The probability of the selected action (as a tensor)
            The critic's value estimate (as a tensor)
        """
        state = torch.as_tensor(state, dtype=torch.float32)
        probs, value = self.actor_critic(state)

        probs_np = probs.detach().numpy()
        action = np.random.choice(len(probs_np), p=probs_np)

        return action, probs[action], value

    def update_actor_critic(self, advantage, prob, value):
        """
        Performs actor critic update.

        args:
            advantage: Advantage of the chosen action (tensor).
            prob: Probability associated with the chosen action (tensor).
            value: Critic's state value estimate (tensor).
        """
        # Compute loss
        actor_loss = self.actor_loss(advantage.detach(), prob).mean()
        critic_loss = self.critic_loss(advantage.detach(), value).mean()

        loss = actor_loss + critic_loss

        # Update actor critic
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train_episode(self):
        """
        Run a single episode of the A2C algorithm.

        Use:
            self.select_action(state): Sample an action from the policy.
            self.step(action): Perform an action in the env.
            self.options.gamma: Gamma discount factor.
            self.actor_critic(state): Returns the action probabilities and
                the critic's estimate at a given state.
            torch.as_tensor(state, dtype=torch.float32): Converts a numpy array
                'state' to a tensor.
            self.update_actor_critic(advantage, prob, value): Update actor critic. 
        """

        state, _ = self.env.reset()
        gamma = self.options.gamma
        for _ in range(self.options.steps):
            ################################
            #   YOUR IMPLEMENTATION HERE   #
            # Run update_actor_critic()    #
            # only ONCE at EACH step in    #
            # an episode.                  # 
            ################################

            # Online one-step actor-critic (Sutton & Barto, Sec. 13.5). Sample A from the current
            # policy; baseline is V(S) from the critic, kept as a tensor so the critic update
            # can backpropagate through it
            action, action_prob, baseline = self.select_action(state)
            next_state, reward, done, _ = self.step(action)

            # Bootstrap from the critic's estimate of V(S'). A terminal state has value 0,
            # so skip the forward pass and the advantage reduces to R - V(S)
            if done:
                estimate = 0
            else:
                # V(S') only feeds the TD target, which semi-gradient TD treats as a constant,
                # so skip building an autograd graph for it. update_actor_critic detaches the
                # advantage anyway, so this saves work without changing the update
                with torch.no_grad():
                    _, estimate = self.actor_critic(torch.as_tensor(next_state, dtype=torch.float32))

            # The one-step TD error is the advantage estimate: R + gamma * V(S') - V(S).
            # It is biased since it relies on the imperfect critic, but unlike REINFORCE
            # it is available after every step, so we update once per step
            advantage = reward + gamma * estimate - baseline

            # Pass V(S): the critic update moves the value of the state we left
            self.update_actor_critic(advantage, action_prob, baseline)

            # Stop collecting once the episode terminates or is truncated
            if done:
                break

            state = next_state

    def actor_loss(self, advantage, prob):
        """
        The policy gradient loss function.
        Note that you are required to define the Loss^PG
        which should be the integral of the policy gradient.

        args:
            advantage: Advantage of the chosen action.
            prob: Probability associated with the chosen action.

        Use:
            torch.log: Element-wise logarithm.

        Returns:
            The unreduced loss (as a tensor).
        """
        ################################
        #   YOUR IMPLEMENTATION HERE   #
        ################################)

        # Autodiff of A * log pi(a|s) gives the policy gradient A * grad log pi
        # (Sutton & Barto, Sec. 13.5). The optimizer minimizes while policy gradient
        # ascends, so negate it. The advantage arrives detached and the mean is taken
        # by the caller (update_actor_critic)
        return -(advantage * torch.log(prob))

    def critic_loss(self, advantage, value):
        """
        The integral of the critic gradient

        args:
            advantage: Advantage of the chosen action.
            value: Critic's state value estimate.

        Returns:
            The unreduced loss (as a tensor).
        """
        ################################
        #   YOUR IMPLEMENTATION HERE   #
        ################################

        # Semi-gradient TD (Sutton & Barto, Sec. 13.5). The critic update direction is
        # delta * grad V(s). The advantage (delta) is detached so it acts as a constant,
        # meaning only V(s) carries a gradient. Negate so that minimizing this surrogate
        # moves V(s) along +delta * grad V(s). Its value is not a meaningful error
        # measure, only its gradient matters
        return -(advantage * value)

    def pull_updates(self):
        raise NotImplementedError

    def __str__(self):
        return "A2C"

    def plot(self, stats, smoothing_window=20, final=False):
        plotting.plot_episode_stats(stats, smoothing_window, final=final)
